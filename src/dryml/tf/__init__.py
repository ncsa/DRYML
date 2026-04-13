# Check we have the right context
from dryml.context import check_context
check_context('tf')

from dryml.core2.utils.classes import install_method

from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import TensorSpec

from .dtype import dtype, _dtype_tf
from .tensor_spec import tensor_spec, _tensor_spec_tf


def _install() -> None:
    install_method(DType, "tf", _dtype_tf)
    install_method(TensorSpec, "tf", _tensor_spec_tf)
_install()


__all__ = ["dtype", "tensor_spec"]
