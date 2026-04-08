from dryml.core2._backend import install_backend_method
from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import TensorSpec
from .dtype import _dtype_tf
from .tensor_spec import _tensor_spec_tf

def install() -> None:
    install_backend_method(DType, "tf", _dtype_tf)
    install_backend_method(TensorSpec, "tf", _tensor_spec_tf)
