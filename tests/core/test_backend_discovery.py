import numpy as np
import pytest

from dryml.core.backend import Backend, discover_backend, discover_backends


def test_discover_backends_for_single_numpy_arg():
    x = np.zeros((2, 3), dtype=np.float32)

    assert discover_backends(x) == {Backend.numpy}


def test_discover_backends_for_nested_numpy_args():
    x = {"a": np.zeros((2,), dtype=np.float32), "b": (np.ones((3,), dtype=np.float32),)}

    assert discover_backends(x) == {Backend.numpy}


def test_discover_backends_ignores_non_backend_metadata():
    x = {"a": np.zeros((2,), dtype=np.float32), "label": 1}

    assert discover_backends(x) == {Backend.numpy}


def test_runtime_discovery_preserves_custom_registered_detectors(monkeypatch):
    """Runtime recognition keeps custom detectors and nested mixed-backend facts."""
    import dryml.core.backend as backend

    native_type = type("Native", (), {"__module__": "torch"})
    value = native_type()
    detector = lambda item: item is value
    monkeypatch.setattr(backend, "backend_testers", {
        Backend.numpy: backend.is_numpy, Backend.torch: detector,
    })
    monkeypatch.setattr(backend, "backend_existence_testers", {
        Backend.numpy: lambda: True, Backend.torch: lambda: True,
    })
    monkeypatch.setattr(Backend, "module", property(lambda self: pytest.fail("custom detector replaced")))

    assert discover_backends({"x": (value, [np.ones(2)]), "metadata": 1},
                             activate_native=True) == {Backend.numpy, Backend.torch}
    assert backend.backend_testers[Backend.torch] is detector
    monkeypatch.setattr(backend, "backend_testers", {Backend.numpy: detector})
    assert discover_backend(value, activate_native=True) is Backend.numpy


def test_native_activation_is_opt_in_restricted_and_not_based_on_similar_names(monkeypatch):
    """Passive discovery, logical specs, and explicit restrictions activate nothing."""
    import dryml.core.backend as backend
    from dryml.core import TensorSpec

    monkeypatch.setattr(backend, "backend_testers", {Backend.numpy: backend.is_numpy})
    monkeypatch.setattr(backend, "backend_existence_testers", {Backend.numpy: lambda: True})
    monkeypatch.setattr(Backend, "module", property(lambda self: pytest.fail("unexpected activation")))
    native = type("Native", (), {"__module__": "torch"})()
    lookalike = type("Unrelated", (), {"__module__": "torch_helpers"})()

    assert discover_backends(native) == set()
    assert discover_backends(native, _backends=[Backend.numpy], activate_native=True) == set()
    assert discover_backends(lookalike, TensorSpec("float32", backend="torch"),
                             activate_native=True) == set()
    with pytest.raises(TypeError, match="Unable to discover backend"):
        discover_backend(native)


def test_runtime_native_hint_activates_once_and_requires_detector_acceptance(monkeypatch):
    """Inherited family hints load an adapter, but never substitute for its tester."""
    import importlib
    import dryml.core.backend as backend

    native_type = type("Native", (), {"__module__": "torch"})
    subclass = type("UserTensor", (native_type,), {"__module__": __name__})
    accepted, rejected = subclass(), native_type()
    monkeypatch.setattr(backend, "backend_testers", {Backend.numpy: backend.is_numpy})
    monkeypatch.setattr(backend, "backend_existence_testers", {Backend.numpy: lambda: True})
    imports = []

    def activate(name):
        imports.append(name)
        backend.backend_testers[Backend.torch] = lambda item: item is accepted
        backend.backend_existence_testers[Backend.torch] = lambda: True
        return object()

    monkeypatch.setattr(importlib, "import_module", activate)
    assert discover_backends({"x": (accepted, [accepted, rejected])},
                             activate_native=True) == {Backend.torch}
    assert discover_backends(rejected, activate_native=True) == set()
    assert imports == ["dryml.torch"]


def test_runtime_discovery_reports_unavailable_native_dependency(monkeypatch):
    """Unavailable activation names the backend instead of returning unknown traits."""
    import importlib
    import dryml.core.backend as backend
    from dryml.context import ContextError

    monkeypatch.setattr(backend, "backend_testers", {Backend.numpy: backend.is_numpy})
    monkeypatch.setattr(backend, "backend_existence_testers", {Backend.numpy: lambda: True})
    monkeypatch.setattr(importlib, "import_module", lambda name: object())
    value = type("Native", (), {"__module__": "torch"})()

    with pytest.raises(ContextError, match="'torch'.*not available.*optional dependency"):
        discover_backends(value, activate_native=True)
