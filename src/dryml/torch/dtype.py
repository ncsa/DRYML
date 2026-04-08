from __future__ import annotations
from dryml.core2.dtype import DType, normalize_dtype


def _dtype_torch(self):
    import torch

    table = {
        "bool": torch.bool,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    try:
        return table[self.name]
    except KeyError:
        raise TypeError(f"Unsupported PyTorch dtype: {self.name}")


def dtype(x: Any) -> DType:
    """
    Convert a torch.dtype, torch.Tensor, or TorchTensorSpec to a DRYML DType.
    """
    if hasattr(x, "dtype"):
        x = x.dtype

    if isinstance(x, str):
        name = x.removeprefix("torch.")
        return normalize_dtype(name)

    s = str(x)
    if not s.startswith("torch."):
        raise TypeError(f"Unsupported PyTorch dtype-like object: {x!r}")

    return normalize_dtype(s.removeprefix("torch."))
