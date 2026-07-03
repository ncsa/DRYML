def __getattr__(name):
    if name == "TFDataset":
        from dryml.data.tf.dataset import TFDataset
        return TFDataset
    raise AttributeError(name)

__all__ = [
    "TFDataset",
]
