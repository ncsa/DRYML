from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from multiprocessing.sharedctypes import Value
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
class TensorLeafSpec:
    dtype: DType | str | Any
    shape: tuple[DimLike, ...] | None = ()
    layout: Layout = Layout.DENSE

    # Optional stuff
    axis_names: tuple[AxisName, ...] | None = None
    ragged_rank: int | None = None
    row_splits_dtype: DType | None = None
    sparse_format: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))
        object.__setattr__(self, "shape", _normalize_shape(self.shape))

        rank = None if self.shape is None else len(self.shape)
        object.__setattr__(
            self,
            "axis_names",
            _normalize_axis_names(self.axis_names, rank, "axis_names"),
        )

        if self.row_splits_dtype is not None:
            object.__setattr__(self, "row_splits_dtype", normalize_dtype(self.row_splits_dtype))

        if not isinstance(self.layout, Layout):
            raise TypeError("layout must be a Layout enum value.")

    @property
    def rank(self) -> int | None:
        return None if self.shape is None else len(self.shape)

    def with_dtype(self, dtype):
        return replace(self, dtype=dtype)

    def with_shape(self, shape):
        return replace(self, shape=shape)

    def __repr__(self) -> str:
        cls_str = f"{type(self).__name__}"

        var_strs = []
        var_strs.append(self.dtype.name)
        var_strs.append(f"shape={self.shape}")
        if self.layout is not Layout.DENSE:
            var_strs.append(f"layout={self.layout}")
        if self.ragged_rank is not None:
            var_strs.append(f"ragged_rank={self.ragged_rank}")
        if self.row_splits_dtype is not None:
            var_strs.append(f"row_splits_dtype={self.row_splits_dtype}")
        if self.sparse_format is not None:
            var_strs.append(f"sparse_format={self.sparse_format}")

        return f"{cls_str}({",".join(var_strs)})"

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, TensorLeafSpec):
            raise NotImplementedError(f"TensorSpec comparison not implemented for this type ({type(rhs)})")

        return (
            self.dtype == rhs.dtype and
            self.shape == rhs.shape and
            self.layout == rhs.layout and
            self.ragged_rank == rhs.ragged_rank and
            self.row_splits_dtype == rhs.row_splits_dtype and
            self.sparse_format == rhs.sparse_format)


LeafSpecTree: TypeAlias = (
    TensorLeafSpec
    | dict[str, "LeafSpecTree"]
    | tuple["LeafSpecTree", ...]
    | list["LeafSpecTree"]
)


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
    spec: LeafSpecTree
    batch: BatchLike = None
    backend: Backend | None = None
    batch_axis_name: AxisName = "batch"

    def __post_init__(self):
        object.__setattr__(self, "batch", _normalize_batch(self.batch))
        if self.batch_axis_name is not None and not isinstance(self.batch_axis_name, str):
            raise TypeError("batch_axis_name must be str or None.")

        if self.batch is None and self.batch_axis_name != "batch":
            # optional: either allow this or force None when unbatched
            pass

        if self.backend is not None:
            if not isinstance(self.backend, Backend):
                object.__setattr__(self, "backend", Backend(self.backend))

    @property
    def batched(self) -> bool:
        return self.batch is not None

    def with_batch(self, batch: int | Dim = Dynamic, axis_name: str | None = "batch") -> TensorSpec:
        return replace(self, batch=batch, batch_axis_name=axis_name)
    def without_batch(self) -> TensorSpec:
        return replace(self, batch=None)

    def __repr__(self) -> str:
        cls_str = f"{type(self).__name__}"

        var_strs = []
        var_strs.append(f"{self.spec!r}")
        if self.batch is not None:
            var_strs.append(f"batch={self.batch}")
        if self.backend is not None:
            var_strs.append(f"backend={self.backend}")

        return f"{cls_str}({",".join(var_strs)})"

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, TensorSpec):
            raise NotImplementedError(f"TensorSpec comparison not implemented for this type ({type(rhs)})")

        return (
            self.dtype == rhs.dtype and
            self.shape == rhs.shape and
            self.batch == rhs.batch )

    def __stable_leaf_bytes__(self):
        return str(self).encode("utf-8")


SpecTree: TypeAlias = (
    TensorSpec
    | dict[str, "SpecTree"]
    | tuple["SpecTree", ...]
    | list["SpecTree"]
)


def is_spec_tree(x: Any) -> bool:
    if isinstance(x, TensorSpec):
        return True

    if isinstance(x, dict):
        return all(isinstance(k, str) and is_spec_tree(v) for k, v in x.items())

    if isinstance(x, tuple):
        return all(is_spec_tree(v) for v in x)

    if isinstance(x, list):
        return all(is_spec_tree(v) for v in x)

    return False


def batch_spec_tree(spec: SpecTree, batch=Dynamic, axis_name: str | None = "batch") -> SpecTree:
    return map_leaves(spec, lambda s: s.with_batch(batch=batch, axis_name=axis_name))


def unbatch_spec_tree(spec: SpecTree) -> SpecTree:
    return map_leaves(spec, lambda s: s.without_batch())


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
        if not isinstance(self, BatchMode):
            object.__setattr__(self, "batch_mode", BatchMode(self.batch_mode))
