"""Failure and cache-cleanup behavior for controlled fake framework imports."""

from __future__ import annotations

import importlib
import os
import sys
import uuid

import pytest

from dryml.runtime import FrameworkImportSafetyError, PublicationError, RuntimeMode, RuntimeState
from dryml.runtime import imports as runtime_imports
from dryml.runtime.frameworks import FrameworkRegistration, framework_registry
from dryml.runtime.publication import PublicationService


@pytest.fixture(autouse=True)
def _isolate_registry_freeze():
    """Keep synthetic registrations isolated from another test's frozen epoch."""
    with framework_registry._lock:
        registrations = dict(framework_registry._registrations)
        framework_registry._frozen = False
    yield
    with framework_registry._lock:
        framework_registry._registrations = registrations
        framework_registry._frozen = False


def test_post_import_failure_cleans_cache_and_fails_closed(tmp_path, monkeypatch):
    """A failure after module execution requires a fresh process."""
    name = "dryml_fake_" + uuid.uuid4().hex
    (tmp_path / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")

    class Adapter:
        def post_import(self, plan, module):
            raise RuntimeError("synthetic post failure")

    framework_registry.register(FrameworkRegistration(name, (name,), Adapter()))
    monkeypatch.syspath_prepend(str(tmp_path))
    service = PublicationService(environ=os.environ)
    service.initialize(RuntimeState())
    monkeypatch.setattr(runtime_imports, "publication", service)
    service.publish(RuntimeState(RuntimeMode.ORCHESTRATOR))
    with pytest.raises(RuntimeError, match="synthetic post failure"):
        importlib.import_module(name)
    assert name not in sys.modules
    assert service.current().health == "failed"


def test_pre_import_failure_does_not_leave_visibility_effects(tmp_path, monkeypatch):
    """A failing pre-import hook aborts before applying owned visibility."""
    name = "dryml_fake_" + uuid.uuid4().hex
    (tmp_path / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")

    class Adapter:
        def plan(self, runtime, visibility):
            return {"env_updates": dict(visibility.env_updates)}

        def apply_pre_import(self, plan):
            raise RuntimeError("synthetic pre failure")

    framework_registry.register(FrameworkRegistration(name, (name,), Adapter()))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    service = PublicationService(environ=os.environ)
    service.initialize(RuntimeState())
    monkeypatch.setattr(runtime_imports, "publication", service)
    service.publish(RuntimeState(RuntimeMode.ORCHESTRATOR))

    with pytest.raises(RuntimeError, match="synthetic pre failure"):
        importlib.import_module(name)
    assert name not in sys.modules
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert service.current().health == "healthy"


def test_late_watched_root_rejects_visibility_changing_publication(monkeypatch):
    """A pre-imported root is never retroactively brought under control."""
    service = PublicationService()
    service.initialize(RuntimeState())
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(PublicationError, match="restart"):
        service.publish(RuntimeState(RuntimeMode.ORCHESTRATOR))
    assert service.current().health == "healthy"


def test_trusted_reload_fails_without_repeating_framework_controls(tmp_path, monkeypatch):
    """Reload never silently repeats an adapter's irreversible post controls."""
    name = "dryml_fake_" + uuid.uuid4().hex
    (tmp_path / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")
    framework_registry.register(FrameworkRegistration(name, (name,), object()))
    monkeypatch.syspath_prepend(str(tmp_path))
    service = PublicationService(environ=os.environ)
    service.initialize(RuntimeState())
    monkeypatch.setattr(runtime_imports, "publication", service)
    service.publish(RuntimeState(RuntimeMode.ORCHESTRATOR))

    module = importlib.import_module(name)
    with pytest.raises(FrameworkImportSafetyError, match="reload"):
        importlib.reload(module)
    assert module.VALUE == 1
