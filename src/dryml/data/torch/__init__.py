def __getattr__(name):
    if name in {"TorchDataset", "TorchIterableDatasetWrapper"}:
        from dryml.data.torch.dataset import TorchDataset, TorchIterableDatasetWrapper
        return {"TorchDataset": TorchDataset, "TorchIterableDatasetWrapper": TorchIterableDatasetWrapper}[name]
    if name == "transforms":
        import dryml.data.torch.transforms as transforms
        return transforms
    raise AttributeError(name)

__all__ = [
    "TorchDataset",
    "TorchIterableDatasetWrapper",
    "transforms",
]
