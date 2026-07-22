from dryml.core.utils.classes import install_method
from dryml.core.dtype import DType
from dryml.core.tensor_spec import TensorSpec
from dryml.core.backend import Backend

from .spec import TorchTensorSpec
from .dtype import dtype, _dtype_torch
from .tensor_spec import as_tensor_spec, _tensor_spec_torch
from .backend import is_torch_value, is_torch_available


def _install() -> None:
    try:
        install_method(DType, "torch", _dtype_torch)
        install_method(TensorSpec, "torch", _tensor_spec_torch)
    except RuntimeError:
        # methods already installed, so we'll exit here.
        return

    from dryml.core.backend import backend_testers, backend_existence_testers
    backend_testers[Backend.torch] = is_torch_value
    backend_existence_testers[Backend.torch] = is_torch_available


_install()


__all__ = ["dtype", "as_tensor_spec", "TorchTensorSpec"]
