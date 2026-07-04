import importlib
import os
import subprocess
import sys
import types

import pytest

import dryml.runtime as runtime
import dryml.runtime.frameworks as runtime_frameworks
from dryml.runtime.errors import FrameworkImportSafetyError
from dryml.runtime.errors import RuntimeTransitionError


def test_importing_runtime_does_not_import_heavy_frameworks(monkeypatch):
    for name in ("torch", "tensorflow", "jax"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    importlib.reload(runtime)

    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules
    assert "jax" not in sys.modules


def test_bootstrap_plan_builds_without_heavy_imports_and_applies_env(monkeypatch):
    for name in ("torch", "tensorflow", "jax"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "frameworks": {"plain": {"num_threads": 2}}, "device_visibility": {"policy": "assigned"}})
    plan = runtime.build_runtime_bootstrap_plan(spec, runtime.RuntimeAllocationView(cpus=(0, 1), accelerators={"gpu": (0,)}))
    env = {}
    runtime.apply_runtime_bootstrap_plan(plan, environ=env)

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["OMP_NUM_THREADS"] == "2"
    assert "torch" not in sys.modules


def test_fake_already_imported_framework_conflict_is_clear(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "frameworks": {"torch": {"num_threads": 2}}, "device_visibility": {"policy": "assigned"}})
    plan = runtime.build_runtime_bootstrap_plan(
        spec,
        runtime.RuntimeAllocationView(accelerators={"gpu": (0,)}),
        policy=runtime.FrameworkBootstrapPolicy(("plain", "torch"), strict_preimport=True),
    )

    with pytest.raises(FrameworkImportSafetyError) as excinfo:
        runtime.apply_runtime_bootstrap_plan(plan, environ={})
    assert excinfo.value.context["framework"] == "torch"


def test_framework_bootstrap_policy_can_validate_major_frameworks(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "frameworks": {"plain": {}}, "device_visibility": {"policy": "assigned"}})
    plan = runtime.build_runtime_bootstrap_plan(
        spec,
        runtime.RuntimeAllocationView(cpus=(0,)),
        policy=runtime.FrameworkBootstrapPolicy(("plain", "torch", "tensorflow", "jax"), strict_preimport=True),
    )

    with pytest.raises(FrameworkImportSafetyError) as excinfo:
        runtime.apply_runtime_bootstrap_plan(plan, environ={})
    assert excinfo.value.context["framework"] == "torch"


def test_strict_policy_defaults_to_declared_frameworks(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", types.ModuleType("tensorflow"))
    spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "frameworks": {"tensorflow": {}}, "device_visibility": {"policy": "assigned"}})
    plan = runtime.build_runtime_bootstrap_plan(
        spec,
        runtime.RuntimeAllocationView(cpus=(0,)),
        policy=runtime.FrameworkBootstrapPolicy(strict_preimport=True),
    )

    assert set(plan.framework_results) == {"plain", "tensorflow"}
    with pytest.raises(FrameworkImportSafetyError) as excinfo:
        runtime.apply_runtime_bootstrap_plan(plan, environ={})
    assert excinfo.value.context["framework"] == "tensorflow"


def test_default_bootstrap_policy_validates_declared_frameworks_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", types.ModuleType("tensorflow"))
    spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "frameworks": {"plain": {}}, "device_visibility": {"policy": "assigned"}})
    plan = runtime.build_runtime_bootstrap_plan(spec, runtime.RuntimeAllocationView(cpus=(0,)))
    env = {}

    runtime.apply_runtime_bootstrap_plan(plan, environ=env)

    assert env[runtime.BOOTSTRAP_MARKER_ENV] == "1"


def test_activate_runtime_bootstrap_restores_environment(monkeypatch):
    monkeypatch.delenv(runtime.BOOTSTRAP_MARKER_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "before")
    spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "frameworks": {"tensorflow": {}}, "device_visibility": {"policy": "assigned"}})
    allocation = runtime.RuntimeAllocationView(accelerators={"gpu": (3,)})
    plan = runtime.build_runtime_bootstrap_plan(spec, allocation)

    with runtime.enter_runtime(runtime.RuntimeMode.WORKER, allocation, spec):
        with runtime.activate_runtime_bootstrap(plan) as state:
            assert state.frameworks == {"plain", "tensorflow"}
            assert runtime.active_runtime_bootstrap() is state
            assert runtime.BOOTSTRAP_MARKER_ENV in state.env_updates
            assert runtime.active_runtime_bootstrap().mode is runtime.RuntimeMode.WORKER
            assert os.environ[runtime.BOOTSTRAP_MARKER_ENV] == "1"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"

    assert runtime.active_runtime_bootstrap() is None
    assert runtime.BOOTSTRAP_MARKER_ENV not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "before"


def test_activate_runtime_bootstrap_requires_matching_active_runtime():
    allocation = runtime.RuntimeAllocationView(accelerators={"gpu": (0,)})
    spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "frameworks": {"tensorflow": {}}, "device_visibility": {"policy": "assigned"}})
    plan = runtime.build_runtime_bootstrap_plan(spec, allocation)

    with pytest.raises(RuntimeTransitionError) as excinfo:
        with runtime.activate_runtime_bootstrap(plan):
            pass
    assert excinfo.value.context["active_mode"] == "orchestrator"


def test_activate_runtime_bootstrap_rejects_process_controls_by_default(monkeypatch):
    allocation = runtime.RuntimeAllocationView(cpus=(0,))
    spec = runtime.RuntimeContextSpec.from_data(
        {
            "mode": "worker",
            "frameworks": {"plain": {"set_cpu_affinity": True}},
            "device_visibility": {"policy": "assigned"},
        }
    )
    plan = runtime.build_runtime_bootstrap_plan(spec, allocation)

    with runtime.enter_runtime(runtime.RuntimeMode.WORKER, allocation, spec):
        with pytest.raises(RuntimeTransitionError) as excinfo:
            with runtime.activate_runtime_bootstrap(plan):
                pass
    assert "plain" in excinfo.value.context["controls"]


def test_plain_bootstrap_can_apply_cpu_affinity_and_memory_limit(monkeypatch):
    calls = {}

    def fake_setaffinity(pid, cpus):
        calls["affinity"] = (pid, set(cpus))

    def fake_setrlimit(kind, limits):
        calls["rlimit"] = (kind, limits)

    monkeypatch.setattr(runtime_frameworks.os, "sched_setaffinity", fake_setaffinity, raising=False)
    monkeypatch.setattr(runtime_frameworks.resource, "setrlimit", fake_setrlimit)
    spec = runtime.RuntimeContextSpec.from_data(
        {
            "mode": "worker",
            "device_visibility": {"policy": "assigned"},
            "frameworks": {"plain": {"set_cpu_affinity": True}},
            "limits": {"memory": "128MiB"},
        }
    )
    allocation = runtime.RuntimeAllocationView(cpus=(2, 3), accelerators={"gpu": (0,)})
    plan = runtime.build_runtime_bootstrap_plan(spec, allocation, policy=runtime.FrameworkBootstrapPolicy(("plain",)))

    runtime.apply_runtime_bootstrap_plan(plan)

    assert calls["affinity"] == (0, {2, 3})
    assert calls["rlimit"] == (runtime_frameworks.resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))


def test_worker_bootstrap_imports_fake_framework_in_fresh_subprocess(tmp_path):
    module_path = tmp_path / "fakeframework.py"
    module_path.write_text("VALUE = 'loaded'\n", encoding="utf-8")
    code = f"""
import os
import sys
import dryml.runtime as runtime

def runtime_framework_result():
    from dryml.runtime.frameworks import FrameworkBootstrapResult
    return FrameworkBootstrapResult()

def runtime_framework_adapter():
    class Adapter:
        name = "fakeframework"
        def build_plan(self, runtime_spec, allocation_view, visibility_plan):
            return runtime_framework_result()
        def validate_before_import(self, result):
            return None
        def apply_pre_import(self, result, *, environ=None):
            return None
        def apply_post_import(self, result):
            return None
    return Adapter()

sys.path.insert(0, {str(tmp_path)!r})
allocation = runtime.RuntimeAllocationView(cpus=(0,), accelerators={{"gpu": (5,)}})
spec = runtime.RuntimeContextSpec.from_data({{
    "mode": "worker",
    "frameworks": {{"fakeframework": {{}}}},
    "device_visibility": {{"policy": "assigned"}},
}})

with runtime.enter_runtime(runtime.RuntimeMode.WORKER, allocation, spec):
    plan = runtime.build_runtime_bootstrap_plan(spec, allocation, adapters={{"fakeframework": runtime_framework_adapter()}})
    with runtime.activate_runtime_bootstrap(plan, adapters={{"fakeframework": runtime_framework_adapter()}}):
        module = runtime.import_configured_framework("fakeframework")
        print(os.environ["CUDA_VISIBLE_DEVICES"] + ":" + module.VALUE)
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert result.stdout.strip() == "5:loaded"
