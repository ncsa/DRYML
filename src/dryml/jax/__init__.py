# Check we have the right context
from dryml.context import check_context
from dryml.core2.utils.classes import install_method
check_context('jax')

from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import TensorSpec

from .dtype import dtype, _dtype_jax
from .tensor_spec import tensor_spec, _tensor_spec_jax

def _install() -> None:
    install_method(DType, "jax", _dtype_jax)
    install_method(TensorSpec, "jax", _tensor_spec_jax)

_install()


__all__ = ["dtype", "tensor_spec"]
