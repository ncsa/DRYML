"""Independent framework-control status publication contracts."""

from __future__ import annotations

import importlib
import subprocess
import sys
import uuid

import pytest

from dryml.runtime import FrameworkImportSafetyError, RuntimeMode, RuntimeState
from dryml.runtime import imports as runtime_imports
from dryml.runtime.frameworks import FrameworkImportPlan, FrameworkPostResult, FrameworkRegistration, framework_registry
from dryml.runtime.publication import PublicationService


def test_successive_same_epoch_finalizers_merge_statuses():
    """Finalizers advance generation without altering the runtime control epoch."""
    service = PublicationService()
    service.initialize(RuntimeState(RuntimeMode.ORCHESTRATOR))
    admission = service.admit_status_finalization()
    first = service.finalize_statuses(admission, {"torch:visibility": "visibility-enforced"})
    second = service.finalize_statuses(admission, {"torch:threading": "framework-configured"})
    assert second.number == first.number + 1
    assert second.metadata["control_epoch"] == admission.control_epoch
    assert dict(second.statuses) == {"torch:visibility": "visibility-enforced", "torch:threading": "framework-configured"}


@pytest.fixture(autouse=True)
def _isolate_registry_freeze():
    """Keep synthetic grouped registrations independent of active tests."""
    with framework_registry._lock:
        framework_registry._frozen = False
    yield
    with framework_registry._lock:
        framework_registry._frozen = False


def test_grouped_roots_share_one_plan_and_status_namespace(tmp_path, monkeypatch):
    """Two roots in one group cannot finalize with divergent adapter controls."""
    first = "dryml_fake_" + uuid.uuid4().hex
    second = "dryml_fake_" + uuid.uuid4().hex
    group = "dryml_group_" + uuid.uuid4().hex
    (tmp_path / f"{first}.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / f"{second}.py").write_text("VALUE = 2\n", encoding="utf-8")

    class Adapter:
        def __init__(self):
            self.modules = []

        def plan(self, runtime, visibility):
            return FrameworkImportPlan(visibility.env_updates)

        def post_import(self, plan, module_name):
            self.modules.append(module_name)
            return FrameworkPostResult({"visibility": "visibility-enforced"})

    service = PublicationService()
    service.initialize(RuntimeState())
    monkeypatch.setattr(runtime_imports, "publication", service)
    adapter = Adapter()
    framework_registry.register(FrameworkRegistration(group, (first, second), adapter))
    monkeypatch.syspath_prepend(str(tmp_path))
    service.publish(RuntimeState(RuntimeMode.ORCHESTRATOR))

    assert importlib.import_module(first).VALUE == 1
    assert importlib.import_module(second).VALUE == 2
    assert adapter.modules == [first, second]
    assert service.current().statuses == {f"{group}:visibility": "visibility-enforced"}
    sys.modules.pop(first, None)
    sys.modules.pop(second, None)


def test_grouped_roots_reject_conflicting_plans(tmp_path, monkeypatch):
    """A second group root cannot silently alter active adapter controls."""
    first = "dryml_fake_" + uuid.uuid4().hex
    second = "dryml_fake_" + uuid.uuid4().hex
    group = "dryml_group_" + uuid.uuid4().hex
    (tmp_path / f"{first}.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / f"{second}.py").write_text("VALUE = 2\n", encoding="utf-8")

    class Adapter:
        changed = False

        def plan(self, runtime, visibility):
            updates = dict(visibility.env_updates)
            if self.changed:
                updates["DRYML_FAKE_GROUP_PLAN"] = "changed"
            return FrameworkImportPlan(updates)

    service = PublicationService()
    service.initialize(RuntimeState())
    monkeypatch.setattr(runtime_imports, "publication", service)
    adapter = Adapter()
    framework_registry.register(FrameworkRegistration(group, (first, second), adapter))
    monkeypatch.syspath_prepend(str(tmp_path))
    service.publish(RuntimeState(RuntimeMode.ORCHESTRATOR))

    assert importlib.import_module(first).VALUE == 1
    adapter.changed = True
    with pytest.raises(FrameworkImportSafetyError, match="incompatible"):
        importlib.import_module(second)
    assert second not in sys.modules
    sys.modules.pop(first, None)


def test_torch_adapter_applies_declared_threads_before_reporting_success(tmp_path):
    """The built-in adapter cannot claim a control it did not invoke."""
    (tmp_path / "torch.py").write_text(
        "THREADS = []\n"
        "def set_num_threads(value): THREADS.append(value)\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def device_count(): return 0\n",
        encoding="utf-8",
    )
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from dryml.runtime import RuntimeContextSpec, RuntimeMode, RuntimeState, publication
from dryml.worlds import LocalResourceInventory
spec = RuntimeContextSpec(RuntimeMode.ORCHESTRATOR, framework={'torch': {'threads': 2}})
publication.publish(RuntimeState(RuntimeMode.ORCHESTRATOR, spec=spec), inventory=LocalResourceInventory((0,), {}))
import torch
assert torch.THREADS == [2]
assert publication.current().statuses['torch:threading'] == 'framework-configured'
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
