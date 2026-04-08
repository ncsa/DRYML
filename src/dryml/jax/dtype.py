from dryml.core2.dtype import DType, normalize_dtype
from typing import Any

def _dtype_jax(self):
    import numpy as np
    try:
        return np.dtype(self.name)
    except TypeError as e:
        raise TypeError(f"Unsupported JAX dtype: {self.name}") from e


def dtype(x: Any) -> DType:
    """
    Convert a JAX dtype-like object, jax array, or ShapeDtypeStruct
    to a DRYML DType.
    """
    import jax.numpy as jnp  # type: ignore

    if hasattr(x, "dtype"):
        x = x.dtype

    try:
        jax_dtype = jnp.dtype(x)
    except Exception as e:
        raise TypeError(f"Unsupported JAX dtype-like object: {x!r}") from e

    return normalize_dtype(jax_dtype.name)
