from dryml.context import check_context
check_context('tf')

from dryml.data.tf.dataset import TFDataset

__all__ = [
    TFDataset,
]
