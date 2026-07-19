"""TensorFlow data namespace.

Legacy ``TFDataset`` has been removed. Use ``dryml.data.TFDSAdapter`` for
TensorFlow Datasets sources and core ``dryml.data`` transforms for batching,
mapping, shuffling, and conversion.
"""

from .cache import TensorFlowCacheView


__all__ = ["TensorFlowCacheView"]
