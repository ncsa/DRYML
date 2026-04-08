# Check we have the right context
from dryml.context import check_context
check_context('torch')

from .convert import install as _install
_install()

from .spec import TorchTensorSpec
from .dtype import dtype
from .tensor_spec import tensor_spec

__all__ = ["dtype", "tensor_spec", "TorchTensorSpec"]
