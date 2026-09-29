from typing import Any
import importlib.util
import sys


def is_torch_available():
    return importlib.util.find_spec("torch") is not None


def is_torch_value(x: Any) -> bool:
    from .spec import TorchTensorSpec
    torch = sys.modules.get("torch")
    if torch is None:
        return isinstance(x, TorchTensorSpec)
    spec = getattr(torch, "__spec__", None)
    if getattr(spec, "_initializing", False):
        return isinstance(x, TorchTensorSpec)
    tensor_type = getattr(torch, "__dict__", {}).get("Tensor")
    return (tensor_type is not None and isinstance(x, tensor_type)) or isinstance(x, TorchTensorSpec)
