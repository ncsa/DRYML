"""Immutable tensor and opaque-value specs, batch metadata and spec-tree helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, TypeAlias

import numpy as np

from .dtype import DType, normalize_dtype
from .backend import Backend, _native_backend, _try_discover_backend
from .utils.recurse import map_leaf_groups, map_leaves, iter_leaves
from .utils.types import is_namedtuple


class Dim(Enum):
    DYNAMIC = "dynamic"

    def __str__(self) -> str:
        return self.value


Dynamic = Dim.DYNAMIC
DimLike = int | Dim
AxisName = str | None
TreeKey: TypeAlias = str | int


class Layout(Enum):
    DENSE = "dense"
    RAGGED = "ragged"
    SPARSE = "sparse"
    PYTHON = "python"

    def __str__(self) -> str:
        return self.value


BatchLike = int | Dim | None

class BatchMode(Enum):
    batched="batched"
    element="element"

def _normalize_dim(dim: DimLike) -> DimLike:
    if isinstance(dim, bool):
        raise TypeError("Shape dimensions cannot be bool.")
    if isinstance(dim, int):
        if dim < 0:
            raise ValueError(f"Shape dimensions must be >= 0, got {dim}.")
        return dim
    if isinstance(dim, Dim):
        return dim
    raise TypeError(f"Invalid dimension {dim!r}.")


def _normalize_shape(shape: tuple[DimLike, ...] | list[DimLike] | None) -> tuple[DimLike, ...] | None:
    if shape is None:
        return None
    return tuple(_normalize_dim(d) for d in shape)


def _normalize_batch(batch: BatchLike) -> BatchLike:
    if batch is None:
        return None
    return _normalize_dim(batch)


def _normalize_axis_names(
    axis_names: tuple[AxisName, ...] | list[AxisName] | None,
    rank: int | None,
    field_name: str,
) -> tuple[AxisName, ...] | None:
    if axis_names is None:
        return None
    if rank is None:
        raise ValueError(f"{field_name} cannot be set when rank is unknown.")
    axis_names = tuple(axis_names)
    if len(axis_names) != rank:
        raise ValueError(f"{field_name} must have length {rank}, got {len(axis_names)}.")
    for name in axis_names:
        if name is not None and not isinstance(name, str):
            raise TypeError(f"{field_name} entries must be str or None.")
    return axis_names


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """
    Backend-independent tensor interface spec.

    Parameters
    ----------
    dtype:
        Canonical semantic dtype.

    shape:
        Sample shape, excluding batch. None means unknown rank.

    batch:
        Batch axis contract:
            None      -> unbatched
            Dynamic   -> batched, unknown batch size
            int       -> batched, fixed batch size

    layout:
        Dense / ragged / sparse / python-object style container.

    axis_names:
        Optional names for sample axes.

    batch_axis_name:
        Optional name for the batch axis.

    backend:
        Optional name for the backend used for this tensor

    ``dtype="object", layout="python"`` describes an opaque Python value,
    without a Python class or content constraint. Its sample shape is ``()`` or
    unknown and it has no tensor backend or sparse/ragged metadata. An explicit
    batch contains opaque values in a list or tuple. Dense NumPy object arrays
    retain their ordinary tensor shape/backend semantics.
    """
    dtype: DType | str | Any
    shape: tuple[DimLike, ...] | list[DimLike] | None = ()
    batch: BatchLike = None
    backend: Backend | None = None
    layout: Layout = Layout.DENSE
    axis_names: tuple[AxisName, ...] | list[AxisName] | None = None
    batch_axis_name: AxisName = "batch"
    ragged_rank: int | None = None
    row_splits_dtype: DType | str | Any | None = None
    sparse_format: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))
        object.__setattr__(self, "shape", _normalize_shape(self.shape))
        object.__setattr__(self, "batch", _normalize_batch(self.batch))

        if not isinstance(self.layout, Layout):
            object.__setattr__(self, "layout", Layout(self.layout))

        rank = None if self.shape is None else len(self.shape)
        object.__setattr__(
            self,
            "axis_names",
            _normalize_axis_names(self.axis_names, rank, "axis_names"),
        )

        if self.batch_axis_name is not None and not isinstance(self.batch_axis_name, str):
            raise TypeError("batch_axis_name must be str or None.")

        if self.batch is None and self.batch_axis_name == "batch":
            object.__setattr__(self, "batch_axis_name", None)

        if self.row_splits_dtype is not None:
            object.__setattr__(self, "row_splits_dtype", normalize_dtype(self.row_splits_dtype))

        if self.backend is not None:
            if not isinstance(self.backend, Backend):
                object.__setattr__(self, "backend", Backend(self.backend))
        if self.is_python_object:
            if self.shape not in ((), None):
                raise ValueError("Opaque Python object specs do not declare sample axes.")
            if self.backend is not None:
                raise ValueError("Opaque Python object specs do not declare a tensor backend.")
            if any(value is not None for value in (self.ragged_rank, self.row_splits_dtype, self.sparse_format)):
                raise ValueError("Opaque Python object specs cannot have sparse or ragged metadata.")

    @property
    def is_python_object(self) -> bool:
        """Return whether this spec denotes opaque, class-agnostic Python values.

        Only canonical object dtype with Python layout qualifies; dense object
        arrays remain tensor leaves. This metadata check inspects no runtime
        value and has no side effects.
        """
        return self.dtype.kind == "object" and self.layout is Layout.PYTHON

    @property
    def rank(self) -> int | None:
        return None if self.shape is None else len(self.shape)

    @property
    def full_shape(self) -> tuple[DimLike, ...] | None:
        if self.shape is None:
            return None
        if self.batch is None:
            return self.shape
        return (self.batch, *self.shape)

    @property
    def full_rank(self) -> int | None:
        shape = self.full_shape
        return None if shape is None else len(shape)

    @property
    def batched(self) -> bool:
        return self.batch is not None

    def framework_shape(self, *, include_batch: bool = True) -> tuple[DimLike, ...] | None:
        if include_batch:
            return self.full_shape
        return self.shape

    def with_batch(self, batch: int | Dim = Dynamic, axis_name: str | None = "batch") -> TensorSpec:
        return replace(self, batch=batch, batch_axis_name=axis_name)

    def without_batch(self) -> TensorSpec:
        return replace(self, batch=None, batch_axis_name=None)

    def with_dtype(self, dtype: DType | str | Any) -> TensorSpec:
        return replace(self, dtype=dtype)

    def with_shape(self, shape: tuple[DimLike, ...] | list[DimLike] | None) -> TensorSpec:
        return replace(self, shape=shape)

    def compatible_with_shape(self, shape: tuple[int, ...] | list[int], *, include_batch: bool = True) -> bool:
        expected = self.framework_shape(include_batch=include_batch)
        if expected is None:
            return True
        shape = tuple(shape)
        if len(shape) != len(expected):
            return False
        return all(e is Dynamic or e == a for e, a in zip(expected, shape))

    def __repr__(self) -> str:
        cls_str = f"{type(self).__name__}"

        var_strs = []
        var_strs.append(self.dtype.name)
        var_strs.append(f"shape={self.shape}")
        if self.batch is not None:
            var_strs.append(f"batch={self.batch}")
        if self.backend is not None:
            var_strs.append(f"backend={self.backend}")
        if self.layout is not Layout.DENSE:
            var_strs.append(f"layout={self.layout}")
        if self.ragged_rank is not None:
            var_strs.append(f"ragged_rank={self.ragged_rank}")
        if self.row_splits_dtype is not None:
            var_strs.append(f"row_splits_dtype={self.row_splits_dtype}")
        if self.sparse_format is not None:
            var_strs.append(f"sparse_format={self.sparse_format}")

        return f"{cls_str}({','.join(var_strs)})"

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, TensorSpec):
            return NotImplemented

        return (
            self.dtype == rhs.dtype and
            self.shape == rhs.shape and
            self.batch == rhs.batch and
            self.layout == rhs.layout and
            self.axis_names == rhs.axis_names and
            self.batch_axis_name == rhs.batch_axis_name and
            self.ragged_rank == rhs.ragged_rank and
            self.row_splits_dtype == rhs.row_splits_dtype and
            self.sparse_format == rhs.sparse_format)

    def equal_exact(self, other: object) -> bool:
        """Compare all normalized TensorSpec fields, including backend.

        Parameters
        ----------
        other:
            Object to compare with this specification.

        Returns
        -------
        bool
            ``True`` when ``other`` is a TensorSpec with identical normalized
            dtype, shape, batch, backend, layout, axis, ragged, and sparse
            metadata; otherwise ``False``. Unlike ``==``, this comparison is
            backend-sensitive and never raises for non-TensorSpec values.
        """
        return TensorSpec.__eq__(self, other) is True and self.backend is other.backend

    def __hash__(self) -> int:
        return hash((
            self.dtype,
            self.shape,
            self.batch,
            self.layout,
            self.axis_names,
            self.batch_axis_name,
            self.ragged_rank,
            self.row_splits_dtype,
            self.sparse_format,
        ))

    def __stable_leaf_bytes__(self):
        return str(self).encode("utf-8")


SpecTree: TypeAlias = (
    TensorSpec
    | dict[TreeKey, "SpecTree"]
    | tuple["SpecTree", ...]
    | list["SpecTree"]
)


def _python_object_batch_size(value: Any) -> int:
    """Read only the outer list/tuple length of an explicitly declared object batch."""
    if not isinstance(value, (list, tuple)):
        raise TypeError("Opaque Python object batches must be lists or tuples.")
    return len(value)


def _python_object_spec(*, batch=None, batch_axis_name="batch") -> TensorSpec:
    """Describe an opaque value domain without accepting an instance or class."""
    return TensorSpec("object", layout=Layout.PYTHON, batch=batch,
                      batch_axis_name=batch_axis_name if batch is not None else None)


def is_spec_tree(x: Any) -> bool:
    if isinstance(x, TensorSpec):
        return True

    if isinstance(x, dict):
        return all(isinstance(k, (str, int)) and is_spec_tree(v) for k, v in x.items())

    if isinstance(x, tuple):
        return all(is_spec_tree(v) for v in x)

    if isinstance(x, list):
        return all(is_spec_tree(v) for v in x)

    return False


def iter_specs(spec: SpecTree) -> Iterator[TensorSpec]:
    yield from iter_leaves(spec, pred=lambda x: isinstance(x, TensorSpec))


def map_spec_tree(spec: SpecTree, fn: Callable[[TensorSpec], TensorSpec]) -> SpecTree:
    return map_leaves(spec, fn, pred=lambda x: isinstance(x, TensorSpec))


def _same_spec_structure(a: Any, b: Any) -> bool:
    if isinstance(a, TensorSpec) and isinstance(b, TensorSpec):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return tuple(a.keys()) == tuple(b.keys()) and all(_same_spec_structure(a[k], b[k]) for k in a)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same_spec_structure(x, y) for x, y in zip(a, b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_spec_structure(x, y) for x, y in zip(a, b))
    return False


def assert_same_spec_structure(*specs: SpecTree) -> None:
    if not specs:
        raise ValueError("At least one spec is required.")
    base = specs[0]
    for spec in specs[1:]:
        if not _same_spec_structure(base, spec):
            raise ValueError("Spec trees do not have the same structure.")


def batch_spec_tree(spec: SpecTree, batch=Dynamic, axis_name: str | None = "batch") -> SpecTree:
    return map_spec_tree(spec, lambda s: s.with_batch(batch=batch, axis_name=axis_name))


def unbatch_spec_tree(spec: SpecTree) -> SpecTree:
    return map_spec_tree(spec, lambda s: s.without_batch())


def spec_tree_is_batched(spec_tree: SpecTree) -> bool:
    batches = {spec.batched for spec in iter_specs(spec_tree)}
    return len(batches) == 1 and batches.pop()


def batch_from_spec_tree(spec_tree: SpecTree):
    batches = {spec.batch for spec in iter_specs(spec_tree) if spec.batched}
    if not batches:
        return None
    return batches.pop() if len(batches) == 1 else Dynamic


def fake_from_spec_tree(spec_tree: SpecTree):
    """Construct a NumPy placeholder tree for supported tensor declarations.

    Args:
        spec_tree: TensorSpec tree whose ranks and NumPy dtypes are available.

    Returns:
        The same container structure with zero-filled arrays. Dynamic dimensions
        become one; unbatched leaves keep the existing leading singleton sample
        convention.

    Raises:
        ValueError: If a rank is unknown or an opaque Python object would need
            invented content.
        TypeError: If a declared dtype has no NumPy representation.

    Side Effects:
        Allocates NumPy arrays; never instantiates arbitrary Python object types.
    """
    def leaf_sample(spec: TensorSpec):
        if spec.is_python_object:
            raise ValueError("Cannot synthesize sample content for an opaque Python object spec.")
        if spec.shape is None:
            raise ValueError("Cannot create a fake sample from an unknown-rank TensorSpec.")
        sample_shape = tuple(1 if dim is Dynamic else int(dim) for dim in spec.shape)
        shape = spec.full_shape if spec.batched else (1, *sample_shape)
        if spec.batched:
            shape = tuple(1 if dim is Dynamic else int(dim) for dim in shape)
        return np.zeros(shape, dtype=spec.dtype.np())

    return map_leaves(spec_tree, leaf_sample, pred=lambda x: isinstance(x, TensorSpec))


def maybe_unbatch_output_spec(output_spec: SpecTree, input_spec: SpecTree) -> SpecTree:
    if spec_tree_is_batched(input_spec):
        return output_spec
    return unbatch_spec_tree(output_spec)


def match_input_batch(spec: TensorSpec, input_spec: SpecTree) -> TensorSpec:
    batch = batch_from_spec_tree(input_spec)
    if batch is None:
        return spec
    return spec.with_batch(batch=batch)


def merge_tensor_specs(specs: Sequence[TensorSpec]) -> TensorSpec:
    """Merge same-position TensorSpecs, marking varying dimensions dynamic."""
    specs = list(specs)
    if not specs:
        raise ValueError("At least one TensorSpec is required.")

    ranks = [spec.rank for spec in specs]
    if len(set(ranks)) > 1:
        raise ValueError("Inconsistent TensorSpec ranks.")

    def dynamic_if_mixed(values):
        values = set(values)
        if len(values) > 1:
            return Dynamic
        return values.pop()

    def uniform_value(values):
        values = set(values)
        if len(values) > 1:
            raise ValueError("TensorSpecs have inconsistent non-shape metadata.")
        return values.pop()

    rank = ranks[0]
    shape = None
    if rank is not None:
        shape = tuple(
            dynamic_if_mixed(spec.shape[dim] for spec in specs)
            for dim in range(rank)
        )

    backends = [spec.backend for spec in specs if spec.backend is not None]
    backend = uniform_value(backends) if backends else None

    return TensorSpec(
        dtype=uniform_value(spec.dtype for spec in specs),
        shape=shape,
        batch=dynamic_if_mixed(spec.batch for spec in specs),
        backend=backend,
        layout=uniform_value(spec.layout for spec in specs),
        ragged_rank=uniform_value(spec.ragged_rank for spec in specs),
        row_splits_dtype=uniform_value(spec.row_splits_dtype for spec in specs),
        sparse_format=uniform_value(spec.sparse_format for spec in specs),
        axis_names=uniform_value(spec.axis_names for spec in specs),
        batch_axis_name=uniform_value(spec.batch_axis_name for spec in specs),
    )


def merge_spec_trees(specs: Sequence[SpecTree]) -> SpecTree:
    """Merge same-structured SpecTrees leaf-wise."""
    return map_leaf_groups(specs, merge_tensor_specs)


def as_tensor_spec(x: Any, *args, require_consistent_backend=True, **kwargs):
    """Infer a logical SpecTree from tensors, Python scalars and opaque objects.

    Args:
        x: One value or a dict/list/tuple/namedtuple value tree. Unknown Python
            leaves become object/Python-layout specs without recording their
            classes or inspecting content. Existing TensorSpecs pass through.
        *args: Positional options forwarded to native backend adapters.
        require_consistent_backend: Require one consistent native tensor backend;
            opaque leaves contribute no backend, rather than a conflicting one.
        **kwargs: Adapter options, including batched and batch_axis_name. With
            batched=True, a nonempty list/tuple of opaque leaves is one opaque
            batch. Other containers retain ordinary tree structure. Ambiguous
            object-valued containers can always use an explicit object spec.

    Returns:
        A normalized SpecTree with numerical or opaque leaves. Variable opaque
        batch lengths can be merged by detect_spec_tree into Dynamic metadata.

    Raises:
        TypeError: For unsupported native adapters/dtypes, detector failures,
            invalid Python-object options, or unsupported mapping keys.
        ValueError: For conflicting native backends, incompatible batching, or
            an empty tree with no inferable leaves.

    Side Effects:
        Invokes registered native spec adapters when recognized; optional native
        families need their existing adapters rather than being relabeled opaque.
        No opaque content, equality, hash, shape or array protocol is inspected.
        Input objects are not retained in the returned metadata.
    """
    batched = kwargs.get("batched", False)
    batch_axis_name = kwargs.get("batch_axis_name", "batch")

    def leaf(value, *, as_batch):
        if isinstance(value, TensorSpec):
            return value
        backend = _try_discover_backend(value)
        if backend is not None:
            options = dict(kwargs, batched=as_batch)
            return backend.as_tensor_spec(value, *args, **options)
        if _native_backend(value) is not None:
            raise TypeError("Native tensor spec inference requires its registered backend adapter.")
        if args or set(kwargs) - {"batched", "batch_axis_name"}:
            raise TypeError("Python value spec inference supports only batched and batch_axis_name options.")
        if np.isscalar(value):
            if as_batch:
                raise ValueError("A Python scalar has no physical batch axis.")
            array = np.asarray(value)
            return TensorSpec(array.dtype, shape=array.shape)
        if as_batch:
            raise ValueError("Opaque object batches require an outer list or tuple.")
        return _python_object_spec()

    def infer(value):
        if isinstance(value, dict):
            if not all(isinstance(key, (str, int)) for key in value):
                raise TypeError("SpecTree mapping keys must be str or int.")
            return {key: infer(child) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            if batched and value and not is_namedtuple(value) and all(
                    not isinstance(child, (dict, tuple, list)) for child in value):
                candidates = tuple(leaf(child, as_batch=False) for child in value)
                if all(spec.is_python_object for spec in candidates):
                    return _python_object_spec(batch=len(value), batch_axis_name=batch_axis_name)
            items = tuple(infer(child) for child in value)
            return type(value)(*items) if is_namedtuple(value) else list(items) if isinstance(value, list) else items
        return leaf(value, as_batch=batched)

    result = infer(x)
    leaves = tuple(iter_specs(result))
    if not leaves:
        raise ValueError("No values with an inferable specification.")
    backends = {spec.backend for spec in leaves if spec.backend is not None}
    if require_consistent_backend and len(backends) > 1:
        raise ValueError(f"Found values from multiple backends: {backends}")
    return result


@dataclass(frozen=True, slots=True)
class SpecHint:
    batch_mode: BatchMode = "element"
    samples: int = 2

    @staticmethod
    def build(d: dict[str,str|int]|str|int|SpecHint|None):
        if d is None:
            return SpecHint()
        if isinstance(d, SpecHint):
            return d
        if isinstance(d, str):
            return SpecHint(batch_mode=d)
        elif isinstance(d, int):
            return SpecHint(samples=d)
        elif isinstance(d, dict):
            return SpecHint(**d)
        raise TypeError(f"Cannot build SpecHint from {type(d).__name__}.")

    def __post_init__(self):
        if not isinstance(self.batch_mode, BatchMode):
            object.__setattr__(self, "batch_mode", BatchMode(self.batch_mode))


def detect_spec_tree(
    samples: Iterable[Any],
    hint: SpecHint | dict[str, str | int] | str | int | None = None,
) -> SpecTree:
    """Infer and merge logical metadata from a bounded number of sample values.

    Args:
        samples: Iterable of numerical, opaque Python, or structured values.
        hint: SpecHint or its dict/string/count shorthand. Defaults to two
            element samples; batched mode treats nonempty lists/tuples of opaque
            leaves as object batches and merges differing lengths to Dynamic.

    Returns:
        A SpecTree retaining structure and common dtype/layout/backend facts,
        with varying dimensions merged to Dynamic and no opaque values/classes.

    Raises:
        TypeError: For invalid hints, structural keys, or native adapter/dtypes.
        ValueError: For insufficient samples or incompatible spec metadata.
        Exception: Propagates sample iterator and registered adapter failures.

    Side Effects:
        Advances the iterable by hint.samples values and invokes recognized
        native adapters. Explicit specs avoid this sampling step entirely.
    """
    hint = SpecHint.build(hint)
    batched = hint.batch_mode is BatchMode.batched
    it = iter(samples)
    specs = []
    for _ in range(hint.samples):
        try:
            sample = next(it)
        except StopIteration as e:
            raise ValueError(
                f"Cannot infer a SpecTree from fewer than {hint.samples} samples."
            ) from e
        specs.append(as_tensor_spec(sample, batched=batched))

    return merge_spec_trees(specs)
