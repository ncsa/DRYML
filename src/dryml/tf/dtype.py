from __future__ import annotations

from typing import Any

from dryml.core2.dtype import DType, normalize_dtype
from dryml.core2.tensor_spec import Dynamic, Layout, TensorSpec


def _dtype_tf(self):
    import tensorflow as tf

    table = {
        "bool": tf.bool,
        "int8": tf.int8,
        "int16": tf.int16,
        "int32": tf.int32,
        "int64": tf.int64,
        "uint8": tf.uint8,
        "uint16": tf.uint16,
        "uint32": tf.uint32,
        "uint64": tf.uint64,
        "float16": tf.float16,
        "float32": tf.float32,
        "float64": tf.float64,
        "bfloat16": tf.bfloat16,
        "complex64": tf.complex64,
        "complex128": tf.complex128,
        "string": tf.string,
    }
    try:
        return table[self.name]
    except KeyError:
        raise TypeError(f"Unsupported TensorFlow dtype: {self.name}")


def _tf_shape_to_dryml(shape: Any) -> tuple[int | object, ...] | None:
    """
    Convert a tf.TensorShape-like object to DRYML shape form.

    Unknown rank -> None
    Unknown dim  -> Dynamic
    """
    try:
        dims = shape.as_list()
    except ValueError:
        return None

    out = []
    for d in dims:
        out.append(Dynamic if d is None else int(d))
    return tuple(out)


def _split_batch(
    shape: tuple[int | object, ...] | None,
    *,
    assume_batched: bool,
) -> tuple[tuple[int | object, ...] | None, int | object | None]:
    if not assume_batched:
        return shape, None

    if shape is None:
        raise ValueError(
            "Cannot set assume_batched=True when the TensorFlow shape has unknown rank."
        )

    if len(shape) == 0:
        raise ValueError(
            "Cannot set assume_batched=True for a rank-0 TensorFlow spec/value."
        )

    return shape[1:], shape[0]


def dtype(x: Any) -> DType:
    """
    Convert a TensorFlow dtype-like object, TensorFlow spec, or TensorFlow value
    to a DRYML DType.
    """
    import tensorflow as tf  # type: ignore

    if hasattr(x, "dtype"):
        x = x.dtype

    tf_dtype = tf.dtypes.as_dtype(x)
    return normalize_dtype(tf_dtype.name)
