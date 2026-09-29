from dryml.core.utils.classes import install_method

from dryml.core.dtype import DType
from dryml.core.tensor_spec import TensorSpec
from dryml.core.backend import Backend

from .dtype import dtype, _dtype_tf
from .tensor_spec import as_tensor_spec, output_signature, _tensor_spec_tf
from .backend import is_tf_available, is_tf_value


def _install() -> None:
    try:
        install_method(DType, "tf", _dtype_tf)
        install_method(TensorSpec, "tf", _tensor_spec_tf)
    except RuntimeError:
        # Another import already installed these extension methods. Runtime
        # recognition still belongs to this optional plugin and must register.
        pass

    from dryml.core.backend import backend_testers, backend_existence_testers
    backend_testers[Backend.tf] = is_tf_value
    backend_existence_testers[Backend.tf] = is_tf_available


_install()


__all__ = ["dtype", "as_tensor_spec", "output_signature"]
