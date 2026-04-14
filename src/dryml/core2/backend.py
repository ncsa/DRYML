from typing import Any
from enum import Enum

from dryml.core2.utils.types import is_numpy, is_python_builtin_pod


class Backend(Enum):
    python = "python"
    numpy = "numpy"
    tf = "tf"
    torch = "torch"
    jax = "jax"

    def __str__(self) -> str:
        return self.value

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, rhs):
        if isinstance(rhs, str):
            return self.value == rhs
        else:
            return self.value == rhs.value

    @property
    def module(self):
        from dryml.context import ContextError
        import importlib
        if not backend_existence_testers[self]():
            raise ContextError("This backend is not available in this context.")
        return importlib.import_module(f"dryml.{self.value}")

    @property
    def dtype(self):
        return self.module.dtype

    @property
    def as_tensor_spec(self):
        return self.module.as_tensor_spec


backend_testers = {
    Backend.python: is_python_builtin_pod,
    Backend.numpy: is_numpy,
}


python_check = lambda: True
numpy_check = lambda: True


backend_existence_testers = {
    Backend.python: python_check,
    Backend.numpy: numpy_check,
}


def available_backends():
    all_backends = [
        Backend.jax,
        Backend.torch,
        Backend.tf,
        Backend.numpy,
        Backend.python]

    backends = []
    for backend in all_backends:
        if backend in backend_existence_testers:
            if backend_existence_testers[backend]():
                backends.append(backend)

    return backends


def discover_backend(x: Any, backends:list[Backend]|None=None) -> Backend:
    if backends is None:
        backends = available_backends()

    for backend in backends:
        if backend_testers[backend](x):
            return backend

    raise TypeError("Unable to discover backend")
