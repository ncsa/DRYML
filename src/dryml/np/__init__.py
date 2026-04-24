# Check we have the right context
from dryml.core2.utils.classes import install_method

from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import TensorSpec
from dryml.core2.backend import Backend

from .dtype import dtype, _dtype_np
from .tensor_spec import as_tensor_spec, _tensor_spec_np


def _install() -> None:
    try:
        install_method(DType, "np", _dtype_np)
        install_method(TensorSpec, "np", _tensor_spec_np)
    except RuntimeError:
        # methods already installed, so we'll exit here.
        return

    from dryml.core2.backend import backend_testers, backend_existence_testers, \
        backend_dtype_method_map, backend_as_tensor_spec_method_map
    backend_testers[Backend.numpy] = lambda: True
    backend_existence_testers[Backend.numpy] = lambda: True
    backend_dtype_method_map[Backend.numpy] = dtype
    backend_as_tensor_spec_method_map[Backend.numpy] = as_tensor_spec


_install()


__all__ = ["dtype", "as_tensor_spec"]
