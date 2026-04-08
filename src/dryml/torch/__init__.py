from .convert import install as _install
_install()

from .spec import TorchTensorSpec
from .dtype import dtype
from .tensor_spec import tensor_spec

from dryml.context import context_check
context_check('torch')

__all__ = ["dtype", "tensor_spec", "TorchTensorSpec"]
