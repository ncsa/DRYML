import importlib.util
from typing import Any


def is_jax_available() -> bool:
    return importlib.util.find_spec("jax") is not None


def is_jax_value(x: Any) -> bool:
    import jax
    return isinstance(x, (jax.Array, jax.ShapeDtypeStruct))
