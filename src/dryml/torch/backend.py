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
    return torch.is_tensor(x) or isinstance(x, TorchTensorSpec)
