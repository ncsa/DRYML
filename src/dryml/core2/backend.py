from __future__ import annotations


from typing import TYPE_CHECKING, Any
from enum import Enum

import numpy as np
from dryml.core2.utils.types import is_numpy
from dryml.core2.utils.recurse import iter_leaves
from dryml.core2.dtype import normalize_dtype

if TYPE_CHECKING:
    from dryml.core2.tensor_spec import TensorSpec, SpecTree


class Backend(Enum):
    numpy = "numpy"
    tf = "tf"
    torch = "torch"
    jax = "jax"

    def __str__(self) -> str:
        return self.value

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, rhs):
        if isinstance(rhs, str):
            return self.value == rhs
        else:
            return self.value == rhs.value

    @property
    def dtype(self):
        return backend_dtype_method_map[self]

    @property
    def as_tensor_spec(self):
        return backend_as_tensor_spec_method_map[self]


backend_map = {
    "numpy": Backend.numpy,
    "tf": Backend.tf,
    "torch": Backend.torch,
    "jax": Backend.jax
}


backend_testers = {
    Backend.numpy: is_numpy,
}


numpy_check = lambda: True


backend_existence_testers = {
    Backend.numpy: numpy_check,
}


def available_backends():
    all_backends = [
        Backend.jax,
        Backend.torch,
        Backend.tf,
        Backend.numpy]

    backends = []
    for backend in all_backends:
        if backend in backend_existence_testers:
            if backend_existence_testers[backend]():
                backends.append(backend)

    return backends


def discover_backend(x: Any, _backends:list[Backend]|None=None) -> Backend:
    if _backends is None:
        _backends = available_backends()

    for backend in _backends:
        if backend_testers[backend](x):
            return backend

    raise TypeError("Unable to discover backend")


def discover_backends(*args, _backends: list[Backend]|None=None, **kwargs) -> set[Backend]:
    if _backends is None:
        _backends = available_backends()

    arg_backends = set(map(lambda v: discover_backends(v, _backends=_backends), iter_leaves(args)))
    kwarg_backends = set(map(lambda v: discover_backends(v, _backends=_backends), iter_leaves(kwargs)))

    return arg_backends.union(kwarg_backends)


def numpy_dtype(x: Any) -> DType:
    """
    Convert a NumPy dtype-like object, ndarray, or scalar value
    to a DRYML DType.
    """
    if hasattr(x, "dtype"):
        x = x.dtype

    np_dtype = np.dtype(x)
    return normalize_dtype(np_dtype.name)


def dims_to_np(shape):
    """
    Convert a DRYML shape to a NumPy shape tuple.

    NumPy does not support symbolic/dynamic dimensions for array creation,
    so Dynamic dimensions are rejected here.
    """
    if shape is None:
        raise ValueError("NumPy does not support unknown-rank tensor specs.")

    out = []
    for d in shape:
        if d is Dynamic:
            raise ValueError(
                "NumPy does not support dynamic dimensions in backend tensor specs."
            )
        out.append(int(d))
    return tuple(out)


def _tensor_spec_np(self, *, include_batch: bool = True):
    """
    Convert a DRYML TensorSpec to a minimal NumPy backend spec.

    Since NumPy has no native TensorSpec object, this returns a
    `(shape, dtype)` pair.
    """
    from dryml.core2.tensor_spec import Layout
    if self.layout is not Layout.DENSE:
        raise TypeError(f"Unsupported NumPy layout: {self.layout}")

    shape = self.framework_shape(include_batch=include_batch)
    np_shape = dims_to_np(shape)
    np_dtype = self.dtype.np()

    return (np_shape, np_dtype)


def _np_shape_to_dryml(shape: Any) -> tuple[int | object, ...] | None:
    """
    Convert a NumPy shape-like object to DRYML shape form.

    NumPy arrays always have known rank and concrete integer dimensions,
    so this is mostly a normalization helper.
    """
    if shape is None:
        return None

    return tuple(int(d) for d in shape)


def _split_batch(
    shape: tuple[int | object, ...] | None,
    *,
    batched: bool,
) -> tuple[tuple[int | object, ...] | None, int | object | None]:
    if not batched:
        return shape, None

    if shape is None:
        raise ValueError(
            "Cannot set batched=True when the NumPy shape has unknown rank."
        )

    if len(shape) == 0:
        raise ValueError(
            "Cannot set batched=True for a rank-0 NumPy value."
        )

    return shape[1:], shape[0]


def numpy_as_tensor_spec(
    x: SpecTree,
    *,
    batched: bool = False,
    batch_axis_name: str | None = "batch",
) -> SpecTree:
    from dryml.core2.tensor_spec import Layout
    """
    Convert a NumPy ndarray or NumPy scalar value to a DRYML TensorSpec.

    Parameters
    ----------
    batched:
        NumPy arrays do not intrinsically identify a "batch axis".
        If True, interpret the leading axis as batch.
    """
    def leaf_to_spec(x: Any) -> TensorSpec:
        if isinstance(x, np.ndarray):
            shape = _np_shape_to_dryml(x.shape)
            sample_shape, batch = _split_batch(shape, batched=batched)

            return TensorSpec(
                dtype=dtype(x.dtype),
                shape=sample_shape,
                batch=batch,
                layout=Layout.DENSE,
                batch_axis_name=batch_axis_name if batch is not None else None,
                backend="numpy",
            )

        if isinstance(x, np.generic):
            shape = ()
            sample_shape, batch = _split_batch(shape, batched=batched)

            return TensorSpec(
                dtype=dtype(x.dtype),
                shape=sample_shape,
                batch=batch,
                layout=Layout.DENSE,
                batch_axis_name=batch_axis_name if batch is not None else None,
                backend="numpy",
            )

        # Allow plain Python scalar values too, since np.asarray handles them cleanly.
        if np.isscalar(x):
            arr = np.asarray(x)
            shape = _np_shape_to_dryml(arr.shape)
            sample_shape, batch = _split_batch(shape, batched=batched)

            return TensorSpec(
                dtype=dtype(arr.dtype),
                shape=sample_shape,
                batch=batch,
                layout=Layout.DENSE,
                batch_axis_name=batch_axis_name if batch is not None else None,
                backend="numpy",
            )

        raise TypeError(f"Unsupported NumPy spec/value type: {type(x).__name__}")

    return map_leaves(x, leaf_to_spec)


backend_dtype_method_map = {
    Backend.numpy: numpy_dtype
}


backend_as_tensor_spec_method_map = {
    Backend.numpy: numpy_as_tensor_spec
}
