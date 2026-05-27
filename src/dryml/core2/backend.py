from __future__ import annotations


from typing import Any
from enum import Enum

from dryml.core2.utils.types import is_numpy
from dryml.core2.utils.recurse import iter_leaves


class Backend(Enum):
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


backend_map = {
    "numpy": Backend.numpy,
    "tf": Backend.tf,
    "torch": Backend.torch,
    "jax": Backend.jax
}


backend_testers = {
    Backend.numpy: is_numpy,
}


numpy_check = lambda: True


backend_existence_testers = {
    Backend.numpy: numpy_check,
}


def available_backends():
    all_backends = [
        Backend.jax,
        Backend.torch,
        Backend.tf,
        Backend.numpy]

    backends = []
    for backend in all_backends:
        if backend in backend_existence_testers:
            if backend_existence_testers[backend]():
                backends.append(backend)

    return backends


def discover_backend(x: Any, _backends:list[Backend]|None=None) -> Backend:
    if _backends is None:
        _backends = available_backends()

    for backend in _backends:
        if backend_testers[backend](x):
            return backend

    raise TypeError("Unable to discover backend")


def discover_backends(*args, _backends: list[Backend]|None=None, **kwargs) -> set[Backend]:
    if _backends is None:
        _backends = available_backends()

    def leaf_backend(v):
        try:
            return discover_backend(v, _backends=_backends)
        except TypeError:
            return None

    arg_backends = {leaf_backend(v) for v in iter_leaves(args)}
    kwarg_backends = {leaf_backend(v) for v in iter_leaves(kwargs)}
    arg_backends.discard(None)
    kwarg_backends.discard(None)

    return arg_backends.union(kwarg_backends)
