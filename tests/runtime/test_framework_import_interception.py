"""Synthetic PEP-451 regression coverage for framework import interception."""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import threading
import uuid

import pytest

import dryml.runtime as runtime
from dryml._framework_imports import coordinator
from dryml.runtime.frameworks import FrameworkBootstrapResult, FrameworkRegistration, framework_registry
from dryml.runtime.publication import FrameworkAdmission, MaterializationFence, PublicationError, PublicationFailedError, PublicationService, SessionGeneration


@pytest.fixture(autouse=True)
def _isolate_registry_freeze():
    """Keep synthetic activation cases process-isolated without a facade reset."""

    with framework_registry._lock:
        framework_registry._frozen = False
    try:
        yield
    finally:
        with framework_registry._lock:
            framework_registry._frozen = False


class _Adapter:
    name = "unused"

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def build_plan(self, spec, allocation, visibility):
        return FrameworkBootstrapResult(env_updates={"DRYML_U3_FAKE": self.name})

    def validate_before_import(self, result):
        self.calls.append("validate")

    def apply_pre_import(self, result, *, environ=None):
        (environ if environ is not None else __import__("os").environ).update(result.env_updates)

    def apply_post_import(self, result):
        self.calls.append("post")


def _module(tmp_path, body="VALUE = 1\n"):
    name = "dryml_fake_" + uuid.uuid4().hex
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")
    return name


def _register(name, calls):
    adapter = _Adapter(name, calls)
    framework_registry.register(FrameworkRegistration(name, (name,), adapter))
    return adapter


def _remove(name):
    for module_name in tuple(sys.modules):
        if module_name == name or module_name.startswith(name + "."):
            sys.modules.pop(module_name, None)


def test_python_mode_preserves_delegated_execution_and_releases_readers(tmp_path, monkeypatch):
    name = _module(tmp_path)
    calls = []
    _register(name, calls)
    monkeypatch.syspath_prepend(str(tmp_path))

    module = importlib.import_module(name)

    assert module.VALUE == 1
    assert calls == []
    assert coordinator.reader_count == 0
    assert not framework_registry._frozen
    _remove(name)


def test_creation_fence_allows_module_from_spec_then_one_execution(tmp_path, monkeypatch):
    name = _module(tmp_path)
    _register(name, [])
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    assert coordinator.reader_count == 0
    spec.loader.exec_module(module)
    assert module.VALUE == 1
    assert coordinator.reader_count == 0
    with pytest.raises(RuntimeError, match="repeated framework module creation"):
        importlib.util.module_from_spec(spec)
    _remove(name)


def test_controlled_raw_import_runs_post_stage_before_return(tmp_path, monkeypatch):
    name = _module(tmp_path, "import os\nSEEN = os.environ['DRYML_U3_FAKE']\n")
    calls = []
    adapter = _register(name, calls)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = runtime.RuntimeContextSpec.from_data(
        {"mode": "probe", "frameworks": {name: {}}, "device_visibility": {"policy": "none"}}
    )
    plan = runtime.build_runtime_bootstrap_plan(
        spec,
        runtime.NoAllocation,
        policy=runtime.FrameworkBootstrapPolicy((name,)),
        adapters={name: adapter},
    )

    with runtime.enter_runtime(runtime.RuntimeMode.PROBE, runtime.NoAllocation, spec):
        with runtime.activate_runtime_bootstrap(plan, adapters={name: adapter}):
            assert framework_registry._frozen
            module = importlib.import_module(name)
            assert module.SEEN == name
            assert calls == ["validate", "post"]
    assert coordinator.reader_count == 0
    _remove(name)


def test_tensorflow_leaf_observes_pre_import_visibility_and_publishes_controls(tmp_path):
    """The built-in leaf configures a fake TensorFlow before raw import returns."""

    package = tmp_path / "tensorflow"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import os\n"
        "SEEN_CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')\n"
        "class _Threading:\n"
        "    @staticmethod\n"
        "    def set_intra_op_parallelism_threads(value): pass\n"
        "    @staticmethod\n"
        "    def set_inter_op_parallelism_threads(value): pass\n"
        "class _Config:\n"
        "    threading = _Threading()\n"
        "    @staticmethod\n"
        "    def get_physical_devices(kind): return ()\n"
        "    @staticmethod\n"
        "    def set_visible_devices(devices, kind): pass\n"
        "    @staticmethod\n"
        "    def get_visible_devices(kind): return ()\n"
        "config = _Config()\n",
        encoding="utf-8",
    )
    script = """
import importlib
import sys
import dryml.runtime as runtime
from dryml.runtime.errors import FrameworkImportSafetyError
sys.path.insert(0, sys.argv[1])
spec = runtime.RuntimeContextSpec.from_data({'mode': 'probe', 'frameworks': {'tensorflow': {'num_threads': 2}}, 'device_visibility': {'policy': 'none'}})
plan = runtime.build_runtime_bootstrap_plan(spec, runtime.NoAllocation, policy=runtime.FrameworkBootstrapPolicy(('tensorflow',)))
with runtime.enter_runtime(runtime.RuntimeMode.PROBE, runtime.NoAllocation, spec):
    with runtime.activate_runtime_bootstrap(plan):
        module = importlib.import_module('tensorflow')
        assert module.SEEN_CUDA_VISIBLE_DEVICES == ''
statuses = runtime.publication.current().metadata['framework_statuses']
assert statuses['tensorflow:tensorflow:visibility'] == 'visibility-enforced'
assert statuses['tensorflow:tensorflow:threads'] == 'framework-configured'
assert statuses['tensorflow:tensorflow:process_memory'] == 'declarative'
"""
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_orchestrator_hides_visible_accelerators_before_fake_torch_module_execution(tmp_path):
    """Strict orchestration permits definition work after hiding raw Torch visibility."""

    (tmp_path / "torch.py").write_text(
        "import os\n"
        "SEEN_VISIBILITY = {name: os.environ.get(name) for name in (\n"
        "    'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',\n"
        "    'ROCR_VISIBLE_DEVICES', 'XLA_VISIBLE_DEVICES',\n"
        ")}\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def device_count(): return 0\n"
        "def set_num_threads(value): pass\n",
        encoding="utf-8",
    )
    script = """
import sys

sys.path.insert(0, sys.argv[1])
from dryml import session
from dryml.core import Definition, Object
from dryml.runtime import NoAllocation, active_runtime

session.set_mode('orchestrator')
assert active_runtime().allocation is NoAllocation
assert 'torch' not in sys.modules
import torch

class DefinitionOnly(Object):
    def __init__(self):
        raise AssertionError('definition construction must not run __init__')

assert isinstance(DefinitionOnly(), Definition)
assert torch.SEEN_VISIBILITY == {
    'CUDA_VISIBLE_DEVICES': '',
    'HIP_VISIBLE_DEVICES': '',
    'ROCR_VISIBLE_DEVICES': '',
    'XLA_VISIBLE_DEVICES': '',
}
"""
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="synthetic-cuda-0",
        HIP_VISIBLE_DEVICES="synthetic-hip-0",
        ROCR_VISIBLE_DEVICES="synthetic-rocr-0",
        XLA_VISIBLE_DEVICES="synthetic-xla-0",
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_mandatory_leaf_failure_prevents_caller_target_entry(tmp_path):
    """A fake TensorFlow lacking its visibility API poisons the controlled import."""

    package = tmp_path / "tensorflow"
    package.mkdir()
    (package / "__init__.py").write_text("class _Config: pass\nconfig = _Config()\n", encoding="utf-8")
    script = """
import importlib
import sys
import dryml.runtime as runtime
from dryml.runtime.errors import FrameworkImportSafetyError
sys.path.insert(0, sys.argv[1])
spec = runtime.RuntimeContextSpec.from_data({'mode': 'probe', 'frameworks': {'tensorflow': {}}, 'device_visibility': {'policy': 'none'}})
plan = runtime.build_runtime_bootstrap_plan(spec, runtime.NoAllocation, policy=runtime.FrameworkBootstrapPolicy(('tensorflow',)))
entered = False
with runtime.enter_runtime(runtime.RuntimeMode.PROBE, runtime.NoAllocation, spec):
    with runtime.activate_runtime_bootstrap(plan):
        try:
            importlib.import_module('tensorflow')
        except FrameworkImportSafetyError:
            pass
        else:
            raise AssertionError('mandatory visibility failure did not reject import')
        assert not entered
assert runtime.publication.current().health == 'failed'
"""
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_helper_import_does_not_repeat_raw_loader_post_stage(tmp_path, monkeypatch):
    name = _module(tmp_path)
    calls = []
    adapter = _register(name, calls)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = runtime.RuntimeContextSpec.from_data(
        {"mode": "probe", "frameworks": {name: {}}, "device_visibility": {"policy": "none"}}
    )
    plan = runtime.build_runtime_bootstrap_plan(
        spec,
        runtime.NoAllocation,
        policy=runtime.FrameworkBootstrapPolicy((name,)),
        adapters={name: adapter},
    )

    with runtime.enter_runtime(runtime.RuntimeMode.PROBE, runtime.NoAllocation, spec):
        with runtime.activate_runtime_bootstrap(plan, adapters={name: adapter}):
            importlib.import_module(name)
            runtime.import_configured_framework(name)
            runtime.apply_runtime_bootstrap_plan(plan, phase="post_import", adapters={name: adapter})

    assert calls == ["validate", "post"]
    assert coordinator.reader_count == 0
    _remove(name)


def test_exec_without_creation_fence_rejects_before_module_execution(tmp_path, monkeypatch):
    name = _module(tmp_path, "raise AssertionError('module body should not execute')\n")
    _register(name, [])
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.loader is not None

    with pytest.raises(PublicationError, match="requires prior PEP-451 module creation"):
        spec.loader.exec_module(type(sys)(name))
    assert coordinator.reader_count == 0
    _remove(name)


def test_controlled_descendant_keeps_its_fence_without_running_the_root_post_hook(tmp_path, monkeypatch):
    package = tmp_path / ("dryml_fake_" + uuid.uuid4().hex)
    package.mkdir()
    (package / "__init__.py").write_text("ROOT = True\n", encoding="utf-8")
    (package / "child.py").write_text("CHILD = True\n", encoding="utf-8")
    calls = []
    name = package.name
    adapter = _register(name, calls)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = runtime.RuntimeContextSpec.from_data({"mode": "probe", "frameworks": {name: {}}, "device_visibility": {"policy": "none"}})
    plan = runtime.build_runtime_bootstrap_plan(spec, runtime.NoAllocation, policy=runtime.FrameworkBootstrapPolicy((name,)), adapters={name: adapter})

    with runtime.enter_runtime(runtime.RuntimeMode.PROBE, runtime.NoAllocation, spec):
        with runtime.activate_runtime_bootstrap(plan, adapters={name: adapter}):
            importlib.import_module(name + ".child")

    assert calls == ["validate", "post", "validate"]
    _remove(name)


def test_materialization_fence_rejects_changed_lifecycle_fingerprint():
    """Execution cannot reuse a creation token for another adapter plan."""

    service = PublicationService()
    service.initialize(object())
    admission = FrameworkAdmission(0, 0, 1, "fake", "fake", "first")
    fence = MaterializationFence(admission, 1, None)

    with pytest.raises(PublicationError, match="materialization is stale"):
        service.validate_materialization(
            fence,
            FrameworkAdmission(0, 0, 1, "fake", "fake", "second"),
        )


def test_stale_post_creation_fence_poisoning_fails_closed(monkeypatch):
    import dryml.runtime.imports as lifecycle_imports

    service = PublicationService()
    service.initialize(object())
    monkeypatch.setattr(lifecycle_imports, "publication", service)
    admission = FrameworkAdmission(0, 0, 1, "fake", "fake", "plan")
    lifecycle = lifecycle_imports._Lifecycle(None, admission, True, object(), object())

    lifecycle_imports._poison(lifecycle, PublicationError("stale materialization"), after_creation=True)

    assert service.current().health == "failed"


def test_same_epoch_finalizers_merge_and_terminal_failure_is_monotonic():
    service = PublicationService()
    service.initialize(object())
    first = FrameworkAdmission(0, 0, 1, "first", "first", "plan")
    second = FrameworkAdmission(0, 0, 1, "second", "second", "plan")

    service.finalize_framework(first, {"first:visibility": "visibility-enforced"})
    merged = service.finalize_framework(second, {"second:visibility": "visibility-enforced"})
    assert dict(merged.metadata["framework_statuses"]) == {
        "first:visibility": "visibility-enforced",
        "second:visibility": "visibility-enforced",
    }

    failed = service.finalize_framework(second, {}, failure=RuntimeError("synthetic failure"))
    assert failed.health == "failed"
    with pytest.raises(PublicationFailedError, match="restart"):
        service.finalize_framework(first, {"first:threads": "framework-configured"})


def test_successful_peer_finalizer_completion_raises_after_terminal_failure():
    service = PublicationService()
    service.initialize(object())
    successful = FrameworkAdmission(0, 0, 1, "successful", "successful", "plan")
    failing = FrameworkAdmission(0, 0, 1, "failing", "failing", "plan")
    assert service.claim_framework_finalizer(successful, 1)
    service.finalize_framework(successful, {"successful:visibility": "visibility-enforced"})
    service.fail_framework(failing, RuntimeError("peer failed"))

    with pytest.raises(PublicationFailedError, match="restart"):
        service.complete_framework_finalizer(successful, 1)


def _install_facade_plan(monkeypatch, service, name, adapter):
    """Publish a synthetic controlled facade plan without optional imports."""

    import dryml.runtime.imports as lifecycle_imports
    import dryml.session.state as state

    result = adapter.build_plan(None, runtime.NoAllocation, None)
    bootstrap = type("Bootstrap", (), {"env_updates": {}, "framework_results": {name: result}})()
    monkeypatch.setattr(lifecycle_imports, "publication", service)
    monkeypatch.setattr(state, "publication", service)
    monkeypatch.setattr(state, "_bootstrap_plan", lambda _runtime: bootstrap)
    monkeypatch.setattr(state, "local_inventory", lambda: __import__("dryml.worlds", fromlist=["LocalResourceInventory"]).LocalResourceInventory((0, 1), {}, memory=None))
    return bootstrap


def test_loader_create_exec_gap_accepts_a_compatible_facade_update(tmp_path, monkeypatch):
    from dryml import session

    name = _module(tmp_path)
    calls = []
    adapter = _register(name, calls)
    service = PublicationService(environ={}, affinity_getter=lambda: {0, 1}, affinity_setter=lambda _cpus: None)
    service.initialize(runtime.RuntimeState(enforcement=runtime.RuntimeEnforcement.OFF))
    _install_facade_plan(monkeypatch, service, name, adapter)
    monkeypatch.syspath_prepend(str(tmp_path))
    session.set_mode("orchestrator")
    spec = importlib.util.find_spec(name)
    module = importlib.util.module_from_spec(spec)
    created_epoch = service.current().metadata["control_epoch"]

    session.require_env("dryml>=0")
    spec.loader.exec_module(module)

    assert module.VALUE == 1
    assert service.current().metadata["control_epoch"] == created_epoch
    assert calls == ["validate", "post"]
    assert coordinator.reader_count == 0
    _remove(name)


def test_loader_create_exec_gap_rejects_an_incompatible_facade_transition(tmp_path, monkeypatch):
    from dryml import session

    name = _module(tmp_path)
    adapter = _register(name, [])
    affinity = {0, 1}
    service = PublicationService(
        environ={},
        affinity_getter=lambda: affinity,
        affinity_setter=lambda cpus: (affinity.clear(), affinity.update(cpus)),
    )
    service.initialize(runtime.RuntimeState(enforcement=runtime.RuntimeEnforcement.OFF))
    _install_facade_plan(monkeypatch, service, name, adapter)
    monkeypatch.syspath_prepend(str(tmp_path))
    session.set_mode("orchestrator")
    spec = importlib.util.find_spec(name)
    module = importlib.util.module_from_spec(spec)

    session.manage(cpus=1)
    with pytest.raises(PublicationError, match="materialization is stale"):
        spec.loader.exec_module(module)

    assert service.current().health == "failed"
    assert coordinator.reader_count == 0
    _remove(name)


def test_loader_callback_facade_reentry_rejects_and_releases_reader(tmp_path, monkeypatch):
    from dryml import session

    name = _module(tmp_path)
    calls = []

    class ReenteringAdapter(_Adapter):
        def validate_before_import(self, result):
            calls.append("validate")
            session.require_env("dryml>=0")

    adapter = ReenteringAdapter(name, calls)
    framework_registry.register(FrameworkRegistration(name, (name,), adapter))
    service = PublicationService(environ={}, affinity_getter=lambda: {0, 1}, affinity_setter=lambda _cpus: None)
    service.initialize(runtime.RuntimeState(enforcement=runtime.RuntimeEnforcement.OFF))
    _install_facade_plan(monkeypatch, service, name, adapter)
    monkeypatch.syspath_prepend(str(tmp_path))
    session.set_mode("orchestrator")
    spec = importlib.util.find_spec(name)

    with pytest.raises(PublicationError, match="upgrade"):
        importlib.util.module_from_spec(spec)

    assert calls == ["validate"]
    assert service.current().health == "healthy"
    assert coordinator.reader_count == 0
    _remove(name)


def test_loader_callback_and_facade_writer_overlap_rejects_without_stranding_reader(tmp_path, monkeypatch):
    from dryml import session

    name = _module(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    class BlockingAdapter(_Adapter):
        def validate_before_import(self, result):
            entered.set()
            assert release.wait(timeout=2)

    adapter = BlockingAdapter(name, [])
    framework_registry.register(FrameworkRegistration(name, (name,), adapter))
    service = PublicationService(environ={}, affinity_getter=lambda: {0, 1}, affinity_setter=lambda _cpus: None)
    service.initialize(runtime.RuntimeState(enforcement=runtime.RuntimeEnforcement.OFF))
    _install_facade_plan(monkeypatch, service, name, adapter)
    monkeypatch.syspath_prepend(str(tmp_path))
    session.set_mode("orchestrator")
    spec = importlib.util.find_spec(name)
    created = []
    failures = []

    def create():
        try:
            created.append(importlib.util.module_from_spec(spec))
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=create)
    thread.start()
    assert entered.wait(timeout=2)
    with pytest.raises(PublicationError, match="import-busy"):
        session.require_env("dryml>=0")
    release.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert failures == []
    assert len(created) == 1
    assert coordinator.reader_count == 0
    spec.loader.exec_module(created[0])
    assert created[0].VALUE == 1
    assert coordinator.reader_count == 0
    _remove(name)


def test_interrupted_loader_execution_poisoning_releases_reader(tmp_path, monkeypatch):
    from dryml import session

    name = _module(tmp_path, "raise KeyboardInterrupt('loader interrupted')\n")
    adapter = _register(name, [])
    service = PublicationService(environ={}, affinity_getter=lambda: {0, 1}, affinity_setter=lambda _cpus: None)
    service.initialize(runtime.RuntimeState(enforcement=runtime.RuntimeEnforcement.OFF))
    _install_facade_plan(monkeypatch, service, name, adapter)
    monkeypatch.syspath_prepend(str(tmp_path))
    session.set_mode("orchestrator")
    spec = importlib.util.find_spec(name)
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(KeyboardInterrupt, match="loader interrupted"):
        spec.loader.exec_module(module)

    assert service.current().health == "failed"
    assert coordinator.reader_count == 0
    _remove(name)


def test_grouped_pure_validation_overlaps_and_publishes_one_pre_outcome(monkeypatch):
    """Pure group validation has no reader-to-reader ownership wait."""

    import dryml.runtime.imports as lifecycle_imports

    service = PublicationService()
    service.initialize(object())
    barrier = threading.Barrier(2)
    calls = []

    class Adapter:
        def validate_before_import(self, result):
            calls.append(threading.get_ident())
            barrier.wait(timeout=2)

    monkeypatch.setattr(lifecycle_imports, "publication", service)
    adapter = Adapter()
    first = lifecycle_imports._Lifecycle(None, FrameworkAdmission(0, 0, 1, "jax", "jax", "plan"), True, object(), adapter)
    second = lifecycle_imports._Lifecycle(None, FrameworkAdmission(0, 0, 1, "jax", "jaxlib", "plan"), True, object(), adapter)
    failures = []

    def validate(lifecycle):
        try:
            lifecycle_imports._validate_before_creation(lifecycle)
        except BaseException as exc:
            failures.append(exc)

    left = threading.Thread(target=validate, args=(first,))
    right = threading.Thread(target=validate, args=(second,))
    left.start()
    right.start()
    left.join(timeout=2)
    right.join(timeout=2)

    assert not left.is_alive() and not right.is_alive()
    assert failures == []
    assert len(calls) == 2
    assert dict(service.current().metadata["framework_pre_stages"]) == {"jax": "plan"}


def test_non_idempotent_grouped_pre_stage_rejects_joiner_without_waiting(monkeypatch):
    """An effectful pre-stage owns one root; peers receive retry guidance."""

    import dryml.runtime.imports as lifecycle_imports

    service = PublicationService()
    service.initialize(object())
    entered = threading.Event()
    release = threading.Event()

    class Adapter:
        pre_import_non_idempotent = True

        def validate_before_import(self, result):
            entered.set()
            assert release.wait(timeout=2)

    monkeypatch.setattr(lifecycle_imports, "publication", service)
    adapter = Adapter()
    first = lifecycle_imports._Lifecycle(None, FrameworkAdmission(0, 0, 1, "jax", "jax", "plan"), True, object(), adapter)
    second = lifecycle_imports._Lifecycle(None, FrameworkAdmission(0, 0, 1, "jax", "jaxlib", "plan"), True, object(), adapter)
    owner = threading.Thread(target=lifecycle_imports._validate_before_creation, args=(first,))
    owner.start()
    assert entered.wait(timeout=2)

    with pytest.raises(PublicationError, match="retry import"):
        lifecycle_imports._validate_before_creation(second)

    release.set()
    owner.join(timeout=2)
    assert not owner.is_alive()
    assert dict(service.current().metadata["framework_pre_stages"]) == {"jax": "plan"}


@pytest.mark.parametrize(
    ("order", "recursive", "expected_roots"),
    [
        ("jax", False, 1),
        ("jaxlib", False, 1),
        ("jaxlib,jax", False, 2),
        ("jax,jaxlib", False, 2),
        ("jax", True, 2),
    ],
)
def test_synthetic_jax_group_lifecycle_covers_direct_recursive_and_ordered_roots(tmp_path, order, recursive, expected_roots):
    """JAX roots share one pre outcome and finish each root before return."""

    for root in ("jax", "jaxlib"):
        package = tmp_path / root
        package.mkdir()
        body = "def devices(kind=None):\n    return []\n" if root == "jax" else ""
        if root == "jax":
            body += "import os\nif os.environ.get('DRYML_U3_RECURSIVE') == '1':\n    import jaxlib\n"
        (package / "__init__.py").write_text(body, encoding="utf-8")
    script = """
import importlib
import os
import sys
import dryml.runtime as runtime

sys.path.insert(0, sys.argv[1])
spec = runtime.RuntimeContextSpec.from_data({'mode': 'probe', 'frameworks': {'jax': {}}, 'device_visibility': {'policy': 'none'}})
plan = runtime.build_runtime_bootstrap_plan(spec, runtime.NoAllocation, policy=runtime.FrameworkBootstrapPolicy(('jax',)))
with runtime.enter_runtime(runtime.RuntimeMode.PROBE, runtime.NoAllocation, spec):
    with runtime.activate_runtime_bootstrap(plan):
        for root in sys.argv[2].split(','):
            importlib.import_module(root)
metadata = runtime.publication.current().metadata
assert len(metadata['framework_plan_fingerprints']) == int(sys.argv[3]), metadata
assert set(metadata['framework_pre_stages']) == {'jax'}, metadata
if sys.argv[2] == 'jaxlib':
    statuses = metadata['framework_statuses']
    assert statuses['jax:jaxlib:visibility'] == 'visibility-enforced', statuses
    assert statuses['jax:jaxlib:threads'] == 'pending-import', statuses
    assert statuses['jax:jaxlib:allocator'] == 'pending-import', statuses
"""
    environment = dict(os.environ, DRYML_U3_RECURSIVE="1" if recursive else "0")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), order, str(expected_roots)],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
