# Check we have the right context
from dryml.context import check_context
check_context('tf')

from dryml.core2.utils.classes import install_method

from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import TensorSpec
from dryml.core2.backend import Backend

from .dtype import dtype, _dtype_tf
from .tensor_spec import tensor_spec, _tensor_spec_tf
from .backend import is_tf_available, is_tf_value


def _install() -> None:
    install_method(DType, "tf", _dtype_tf)
    install_method(TensorSpec, "tf", _tensor_spec_tf)

    from dryml.core2.backend import backend_testers
    from dryml.core2.backend import backend_existence_testers
    backend_testers[Backend.tf] = is_tf_value
    backend_existence_testers[Backend.tf] = is_tf_available


_install()


__all__ = ["dtype", "tensor_spec"]
