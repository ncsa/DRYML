# Check we have the right context
from dryml.context import check_context
check_context('jax')

from .convert import install as _install
from .dtype import dtype
from .tensor_spec import tensor_spec
_install()


__all__ = ["dtype", "tensor_spec"]
