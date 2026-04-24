# Check we have the right context
from dryml.context import check_context
from dryml.core2.utils.classes import install_method
check_context('jax')

from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import TensorSpec
from dryml.core2.backend import Backend

from .dtype import dtype, _dtype_jax
from .tensor_spec import as_tensor_spec, _tensor_spec_jax
from .backend import is_jax_available, is_jax_value


def _install() -> None:
    install_method(DType, "jax", _dtype_jax)
    install_method(TensorSpec, "jax", _tensor_spec_jax)

    from dryml.core2.backend import backend_testers, backend_existence_testers, \
        backend_dtype_method_map, backend_as_tensor_spec_method_map
    backend_testers[Backend.jax] = is_jax_value
    backend_existence_testers[Backend.jax] = is_jax_available
    backend_dtype_method_map[Backend.jax] = dtype
    backend_as_tensor_spec_method_map[Backend.jax] = as_tensor_spec


_install()


__all__ = ["dtype", "as_tensor_spec"]
