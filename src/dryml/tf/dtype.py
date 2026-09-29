from __future__ import annotations

from typing import Any

from dryml.core.dtype import DType, normalize_dtype


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
