from typing import Any
import importlib.util
import sys


def is_tf_available():
    try:
        return importlib.util.find_spec("tensorflow") is not None
    except (ImportError, ValueError):
        return False


def is_tf_value(x: Any) -> bool:
    tf = sys.modules.get("tensorflow")
    if tf is None:
        return False
    return tf.is_tensor(x) or isinstance(x, tf.TypeSpec)
