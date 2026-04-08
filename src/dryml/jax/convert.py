from dryml.core2._backend import install_backend_method
from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import TensorSpec
from .dtype import _dtype_jax
from .tensor_spec import _tensor_spec_jax

def install() -> None:
    install_backend_method(DType, "jax", _dtype_jax)
    install_backend_method(TensorSpec, "jax", _tensor_spec_jax)
