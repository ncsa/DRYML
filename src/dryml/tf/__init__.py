from .convert import install as _install
from .dtype import dtype
from .tensor_spec import tensor_spec
_install()

from dryml.context import context_check
context_check('tf')


__all__ = ["dtype", "tensor_spec"]
