from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, TypeAlias

from .dtype import DType, normalize_dtype
from .backend import discover_backend, Backend
from .utils.recurse import map_leaves, iter_leaves


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


def as_tensor_spec(x: Any, *args, require_consistent_backend=True, **kwargs):
    # Tensor Specs should have a consistent backend.
    all_backends = set(map(discover_backend, iter_leaves(x)))
    if len(all_backends) == 0:
        raise ValueError("No values with a backend?")

    if require_consistent_backend and len(all_backends) > 1:
        raise ValueError(f"Found values from multiple backends: {all_backends}")

    backend = all_backends.pop()

    return backend.as_tensor_spec(x, *args, **kwargs)


@dataclass(frozen=True, slots=True)
class SpecHint:
    batch_mode: BatchMode = "element"
    samples: int = 2

    @staticmethod
    def build(d: dict[str,str|int]|str|int):
        if isinstance(d, str):
            return SpecHint(batch_mode=d)
        elif isinstance(d, int):
            return SpecHint(samples=d)
        elif isinstance(d, dict):
            return SpecHint(**d)

    def __post_init__(self):
        if not isinstance(self.batch_mode, BatchMode):
            object.__setattr__(self, "batch_mode", BatchMode(self.batch_mode))
