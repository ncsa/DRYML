from typing import Any
import numpy as np
from dryml.core.utils.recurse import map_leaves
from dryml.core.tensor_spec import Dynamic, TensorSpec, SpecTree


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
    from dryml.core.tensor_spec import Layout
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


def as_tensor_spec(
    x: SpecTree,
    *,
    batched: bool = False,
    batch_axis_name: str | None = "batch",
) -> SpecTree:
    from dryml.core.tensor_spec import Layout
    """
    Convert a NumPy ndarray or NumPy scalar value to a DRYML TensorSpec.

    Parameters
    ----------
    batched:
        NumPy arrays do not intrinsically identify a "batch axis".
        If True, interpret the leading axis as batch.
    """
    def leaf_to_spec(x: Any) -> TensorSpec:
        from dryml.core.tensor_spec import TensorSpec
        from dryml.core.dtype import dtype
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
