# Check we have the right context
from dryml.core.utils.classes import install_method

from dryml.core.dtype import DType
from dryml.core.tensor_spec import TensorSpec
from dryml.core.backend import Backend

from .dtype import dtype, _dtype_np
from .tensor_spec import as_tensor_spec, _tensor_spec_np


def _install() -> None:
    try:
        install_method(DType, "np", _dtype_np)
        install_method(TensorSpec, "np", _tensor_spec_np)
    except RuntimeError:
        # methods already installed, so we'll exit here.
        return

    from dryml.core.backend import backend_testers, backend_existence_testers
    from dryml.core.utils.types import is_numpy
    backend_testers[Backend.numpy] = is_numpy
    backend_existence_testers[Backend.numpy] = lambda: True


_install()


__all__ = ["dtype", "as_tensor_spec"]
