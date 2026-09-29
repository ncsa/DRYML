"""PEP 451 fake-loader coverage for watched framework imports."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import uuid

import pytest

from dryml.runtime import RuntimeMode, RuntimeState, publication
from dryml.runtime.frameworks import FrameworkPostResult, FrameworkRegistration, framework_registry


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


def test_controlled_import_sees_visibility_and_finalizes_before_return(tmp_path, monkeypatch):
    """A raw root import observes pre-controls and returns finalized statuses."""
    name = "dryml_fake_" + uuid.uuid4().hex
    (tmp_path / f"{name}.py").write_text("import os\nSEEN = os.environ['CUDA_VISIBLE_DEVICES']\n", encoding="utf-8")

    class Adapter:
        def plan(self, runtime, visibility):
            return {"env_updates": dict(visibility.env_updates)}

        def post_import(self, plan, module):
            return FrameworkPostResult({"visibility": "visibility-enforced", "threading": "unsupported"})

    framework_registry.register(FrameworkRegistration(name, (name,), Adapter()))
    monkeypatch.syspath_prepend(str(tmp_path))
    publication.publish(RuntimeState(RuntimeMode.ORCHESTRATOR))
    before = publication.current().number
    module = importlib.import_module(name)
    assert module.SEEN == ""
    assert publication.current().number > before
    assert publication.current().statuses[f"{name}:visibility"] == "visibility-enforced"
    sys.modules.pop(name, None)


def test_descendant_from_import_and_custom_create_module_keep_loader_semantics(tmp_path, monkeypatch):
    """Wrapped package loaders retain normal descendants and custom creation."""
    name = "dryml_fake_" + uuid.uuid4().hex
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("from .child import VALUE\n", encoding="utf-8")
    (package / "child.py").write_text("VALUE = 3\n", encoding="utf-8")
    framework_registry.register(FrameworkRegistration(name, (name,), object()))
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    assert module.VALUE == 3
    assert name + ".child" in sys.modules
    sys.modules.pop(name + ".child", None)
    sys.modules.pop(name, None)


def test_descendant_import_does_not_repeat_root_controls(tmp_path, monkeypatch):
    """Nested modules delegate normally after their root controls ran once."""
    name = "dryml_fake_" + uuid.uuid4().hex
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("from .child import VALUE\n", encoding="utf-8")
    (package / "child.py").write_text("VALUE = 3\n", encoding="utf-8")

    class Adapter:
        def __init__(self):
            self.pre_imports = 0
            self.post_imports = 0

        def plan(self, runtime, visibility):
            return {"env_updates": dict(visibility.env_updates)}

        def apply_pre_import(self, plan):
            self.pre_imports += 1

        def post_import(self, plan, module_name):
            self.post_imports += 1
            return FrameworkPostResult({"visibility": "visibility-enforced"})

    adapter = Adapter()
    framework_registry.register(FrameworkRegistration(name, (name,), adapter))
    monkeypatch.syspath_prepend(str(tmp_path))
    publication.publish(RuntimeState(RuntimeMode.ORCHESTRATOR))

    assert importlib.import_module(name).VALUE == 3
    assert adapter.pre_imports == 1
    assert adapter.post_imports == 1
    sys.modules.pop(name + ".child", None)
    sys.modules.pop(name, None)


def test_custom_loader_create_module_is_delegated_under_preimport_controls(monkeypatch):
    """The wrapper preserves a loader-created module and PEP 451 callbacks."""
    name = "dryml_fake_" + uuid.uuid4().hex

    class Loader:
        def __init__(self):
            self.create_calls = 0
            self.exec_calls = 0

        def create_module(self, spec):
            self.create_calls += 1
            return None

        def exec_module(self, module):
            self.exec_calls += 1
            import os
            module.seen_visibility = os.environ["CUDA_VISIBLE_DEVICES"]

    class Delegate(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == name:
                return importlib.util.spec_from_loader(fullname, loader)
            return None

    loader = Loader()
    framework_registry.register(FrameworkRegistration(name, (name,), object()))
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path[:1], Delegate(), *sys.meta_path[1:]])
    publication.publish(RuntimeState(RuntimeMode.ORCHESTRATOR))

    module = importlib.import_module(name)
    assert module.seen_visibility == ""
    assert loader.create_calls == loader.exec_calls == 1
    sys.modules.pop(name, None)
