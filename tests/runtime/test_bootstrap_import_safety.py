import importlib
import sys
import types

import pytest

import dryml.runtime as runtime
import dryml.runtime.frameworks as runtime_frameworks
from dryml.runtime.errors import FrameworkImportSafetyError


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
    plan = runtime.build_runtime_bootstrap_plan(spec, runtime.RuntimeAllocationView(accelerators={"gpu": (0,)}))

    with pytest.raises(FrameworkImportSafetyError) as excinfo:
        runtime.apply_runtime_bootstrap_plan(plan, environ={})
    assert excinfo.value.context["framework"] == "torch"


def test_default_bootstrap_policy_validates_major_frameworks(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "frameworks": {"plain": {}}, "device_visibility": {"policy": "assigned"}})
    plan = runtime.build_runtime_bootstrap_plan(spec, runtime.RuntimeAllocationView(cpus=(0,)))

    with pytest.raises(FrameworkImportSafetyError) as excinfo:
        runtime.apply_runtime_bootstrap_plan(plan, environ={})
    assert excinfo.value.context["framework"] == "torch"


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
