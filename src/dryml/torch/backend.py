from typing import Any

from dryml.context import check_context, ContextError


def is_torch_available():
    try:
        check_context('torch')
        return True
    except ContextError:
        pass
    return False


def is_torch_value(x: Any) -> bool:
    import torch
    from .spec import TorchTensorSpec
    return torch.is_tensor(x) or isinstance(x, TorchTensorSpec)
