from dryml.context.tf.context import TFComputeContext
from dryml.context import register_context_manager


register_context_manager('tf', TFComputeContext)


__all__ = [
    TFComputeContext,
]
