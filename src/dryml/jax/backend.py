"""Passive JAX value detection and installed runtime-adapter availability."""

from typing import Any
import importlib.util
import sys


def is_jax_available() -> bool:
    """Return installed-adapter availability without requiring a training context."""
    return importlib.util.find_spec("jax") is not None


def is_jax_value(x: Any) -> bool:
    """Recognize loaded JAX values without activating JAX during discovery."""
    jax = sys.modules.get("jax")
    if jax is None:
        return False
    value_types = (vars(jax).get(name) for name in ("Array", "ShapeDtypeStruct"))
    return any(isinstance(value_type, type) and isinstance(x, value_type) for value_type in value_types)
