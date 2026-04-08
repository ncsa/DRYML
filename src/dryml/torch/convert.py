from dryml.core2._backend import install_backend_method
from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import TensorSpec
from .dtype import _dtype_torch
from .tensor_spec import _tensor_spec_torch


def install() -> None:
    install_backend_method(DType, "torch", _dtype_torch)
    install_backend_method(TensorSpec, "torch", _tensor_spec_torch)
