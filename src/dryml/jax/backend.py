import importlib.util
import sys
from typing import Any


def is_jax_available() -> bool:
    try:
        return importlib.util.find_spec("jax") is not None
    except (ImportError, ValueError):
        return False


def is_jax_value(x: Any) -> bool:
    jax = sys.modules.get("jax")
    if jax is None:
        return False
    return isinstance(x, (jax.Array, jax.ShapeDtypeStruct))
