# Check we have the right context
from dryml.context import check_context
from dryml.core.utils.classes import install_method
check_context('jax')

from dryml.core.dtype import DType
from dryml.core.tensor_spec import TensorSpec
from dryml.core.backend import Backend

from .dtype import dtype, _dtype_jax
from .tensor_spec import as_tensor_spec, _tensor_spec_jax
from .backend import is_jax_available, is_jax_value


def _install() -> None:
    try:
        install_method(DType, "jax", _dtype_jax)
        install_method(TensorSpec, "jax", _tensor_spec_jax)
    except RuntimeError:
        # methods already installed, so we'll exit here.
        return

    from dryml.core.backend import backend_testers, backend_existence_testers
    backend_testers[Backend.jax] = is_jax_value
    backend_existence_testers[Backend.jax] = is_jax_available


_install()


__all__ = ["dtype", "as_tensor_spec"]
