"""Closed backend identifiers and shared passive/runtime value discovery.

Discovery consults registered detectors by default. Runtime callers may explicitly
activate a native value's adapter; metadata planning never requests activation.
"""

from __future__ import annotations


from typing import Any
from enum import Enum

from dryml.core.utils.types import is_numpy
from dryml.core.utils.recurse import iter_leaves


class Backend(Enum):
    """Identify a supported tensor backend without activating its runtime.

    Members name NumPy, TensorFlow, Torch, and JAX. Enum construction rejects
    unknown spellings with ValueError. The module property lazily activates the
    existing plugin at runtime and raises ContextError for unavailable optional
    dependencies. Discovery consults registered detectors unless a runtime caller
    explicitly requests native adapter activation.
    """
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
        """Activate this backend's adapter at a runtime boundary.

        Returns:
            The registered ``dryml.<backend>`` plugin module.

        Raises:
            ContextError: If the optional dependency is unavailable. Import errors
            within an installed plugin otherwise propagate.

        Side Effects:
            Lazily imports the plugin and registers its existing runtime adapters.
            Passive discovery and metadata planning never call this property.
        """
        from dryml.context import ContextError
        import importlib
        module = importlib.import_module(f"dryml.{self.value}")
        tester = backend_existence_testers.get(self)
        if tester is None or not tester():
            raise ContextError(
                f"The {self.value!r} backend is not available in this context. "
                "Use an environment with its optional dependency installed."
            )
        return module

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


def _native_backend(value: object) -> Backend | None:
    """Return a native type-family hint without imports or runtime validation.

    Inherited module roots recognize user tensor subclasses, but unrelated names
    such as ``torch_helpers`` do not activate an adapter. A hint is not admission:
    discovery still uses registered detectors and conversion validates each leaf.
    """
    if is_numpy(value):
        return Backend.numpy
    for cls in type(value).__mro__:
        root = cls.__module__.split(".", 1)[0]
        if root == "tensorflow":
            return Backend.tf
        if root == "torch":
            return Backend.torch
        if root in ("jax", "jaxlib"):
            return Backend.jax
    return None


def discover_backend(
    x: Any, _backends: list[Backend] | None = None, *, activate_native: bool = False,
) -> Backend:
    """Identify one leaf through the existing registered backend detectors.

    Args:
        x: Runtime value or logical value understood by a registered detector.
        _backends: Optional restricted, ordered detector list; defaults to the
            available registered backends.
        activate_native: Runtime-only opt-in to lazily register a missing native
            NumPy/TensorFlow/Torch/JAX adapter suggested by the value's type family.
            Existing custom detectors are retained, and an explicit detector list
            still restricts which adapters can activate.

    Returns:
        The first backend whose detector accepts ``x``.

    Raises:
        TypeError: If no detector accepts the value.
        ContextError: If a requested native adapter's dependency is unavailable.
        ImportError: If activation fails inside an installed plugin.

    Side Effects:
        None by default. With ``activate_native=True``, may import the matching
        optional plugin and register its adapter. Does not copy/extract values or
        validate placement, gradients, or conversion eligibility.
    """
    candidates = available_backends() if _backends is None else _backends
    for backend in candidates:
        tester = backend_testers.get(backend)
        if tester is not None and tester(x):
            return backend

    if activate_native:
        native = _native_backend(x)
        if (native is not None and native not in backend_testers
                and (_backends is None or native in _backends)):
            native.module
            if backend_testers[native](x):
                return native

    raise TypeError("Unable to discover backend")


def discover_backends(
    *args, _backends: list[Backend] | None = None, activate_native: bool = False, **kwargs,
) -> set[Backend]:
    """Discover distinct backends in nested positional and keyword value trees.

    Args:
        *args: Values with dict/list/tuple/namedtuple containers.
        _backends: Optional restricted detector list, as in ``discover_backend``.
        activate_native: Runtime-only native adapter activation opt-in.
        **kwargs: Additional value trees; mapping keys are not inspected.

    Returns:
        A set of detected backends, ignoring leaves no detector recognizes.

    Raises:
        ContextError: If native activation requires an unavailable dependency.
        ImportError: If activation fails inside an installed plugin.

    Side Effects:
        Passive by default; the runtime opt-in can register missing native adapters
        before detection. Selected consumers still own runtime validation.
    """
    if _backends is None and not activate_native:
        _backends = available_backends()

    def leaf_backend(v):
        try:
            return discover_backend(v, _backends=_backends, activate_native=activate_native)
        except TypeError:
            return None

    arg_backends = {leaf_backend(v) for v in iter_leaves(args)}
    kwarg_backends = {leaf_backend(v) for v in iter_leaves(kwargs)}
    arg_backends.discard(None)
    kwarg_backends.discard(None)

    return arg_backends.union(kwarg_backends)
