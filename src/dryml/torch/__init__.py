# Check we have the right context
from dryml.context import check_context
check_context('torch')

from dryml.core2.utils.classes import install_method
from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import TensorSpec

from .spec import TorchTensorSpec
from .dtype import dtype, _dtype_torch
from .tensor_spec import tensor_spec, _tensor_spec_torch


def _install() -> None:
    install_method(DType, "torch", _dtype_torch)
    install_method(TensorSpec, "torch", _tensor_spec_torch)
_install()


__all__ = ["dtype", "tensor_spec", "TorchTensorSpec"]
