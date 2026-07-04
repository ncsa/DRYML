import importlib.util
import sys
from typing import Any


def is_jax_available() -> bool:
    return importlib.util.find_spec("jax") is not None


def is_jax_value(x: Any) -> bool:
    jax = sys.modules.get("jax")
    if jax is None:
        return False
    return isinstance(x, (jax.Array, jax.ShapeDtypeStruct))
