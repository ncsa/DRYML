"""Runtime annotation decorators.

These decorators declare runtime defaults only. They do not enter runtime,
activate bootstrap, mutate ``os.environ``, or import heavy frameworks.
"""

from dryml.annotations.runtime import default, runtime_default_fragment

__all__ = ["default", "runtime_default_fragment"]
