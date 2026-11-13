from typing import Any
import importlib.util
import sys


def is_tf_available():
    return importlib.util.find_spec("tensorflow") is not None


def is_tf_value(x: Any) -> bool:
    """Recognize dense/ragged values and TypeSpecs from an already-loaded TensorFlow.

    Args:
        x: Runtime value to classify without optional framework activation.

    Returns:
        False if TensorFlow is not loaded, otherwise its tensor/type-spec test.
        No tensor content or row partitions are read.
    """
    tf = sys.modules.get("tensorflow")
    if tf is None:
        return False
    return tf.is_tensor(x) or isinstance(x, (tf.TypeSpec, tf.RaggedTensor))
