from typing import Any

from dryml.context import check_context, ContextError


def is_tf_available():
    try:
        check_context('tf')
        return True
    except ContextError:
        pass
    return False


def is_tf_value(x: Any) -> bool:
    import tensorflow as tf
    return tf.is_tensor(x)
