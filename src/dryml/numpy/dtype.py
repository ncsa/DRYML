from typing import Any
import numpy as np
from dryml.core.dtype import DType, normalize_dtype


def _dtype_np(self):
    import numpy as np

    mapping = {
        "bool": np.bool_,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
        "bytes": np.bytes_,
        "str": np.str_,
    }

    if self.name == "bfloat16":
        if hasattr(np, "bfloat16"):
            return np.bfloat16
        raise TypeError("NumPy bfloat16 is not available in this environment.")

    try:
        return mapping[self.name]
    except KeyError as e:
        raise TypeError(
            f"Can't convert DType {self.name!r} to a NumPy dtype."
        ) from e


def dtype(x: Any) -> DType:
    """
    Convert a NumPy dtype-like object, ndarray, or scalar value
    to a DRYML DType.
    """
    if hasattr(x, "dtype"):
        x = x.dtype

    np_dtype = np.dtype(x)
    return normalize_dtype(np_dtype.name)
