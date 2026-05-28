from dryml.context import context_check, NoContextError, \
    WrongContextError, ContextIncompatibilityError
from typing import Any

expected_context_errors = (NoContextError, WrongContextError, ContextIncompatibilityError)

def signature_discovery(obj: Any, **kwargs):
    try:
        context_check('tf')
        from .tf.utils import tf_signature_discovery
        return tf_signature_discovery(obj, **kwargs)
    except expected_context_errors:
        pass

    raise ValueError("Unable to guess a signature based on the object.")
