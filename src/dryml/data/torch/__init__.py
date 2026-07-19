"""PyTorch data namespace.

Legacy ``TorchDataset`` wrappers have been removed. Use
``dryml.data.TorchDatasetAdapter`` for PyTorch dataset sources and core
``dryml.data`` transforms for batching, mapping, shuffling, and conversion.
"""

from .cache import TorchCacheView


__all__ = ["TorchCacheView"]
