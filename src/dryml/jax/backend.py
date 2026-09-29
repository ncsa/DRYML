from typing import Any
from dryml.context import check_context, ContextError


def is_jax_available() -> bool:
    try:
        check_context('jax')
        return True
    except ContextError:
        pass
    return False


def is_jax_value(x: Any) -> bool:
    import jax
    return isinstance(x, (jax.Array, jax.ShapeDtypeStruct))
