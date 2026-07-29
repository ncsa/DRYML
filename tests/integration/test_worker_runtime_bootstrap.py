import sys

import pytest

import dryml.runtime as runtime
from dryml.runtime.errors import NoAllocationError


def test_worker_setup_before_fake_framework_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    allocation = runtime.RuntimeAllocationView(role="trainer", replica=0, rank=0, local_rank=0, cpus=(0,), accelerators={"gpu": (3,)})
    spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "device_visibility": {"policy": "assigned"}})

    with runtime.enter_runtime(runtime.RuntimeMode.WORKER, allocation, spec):
        plan = runtime.build_runtime_bootstrap_plan(spec, allocation)
        env = {}
        runtime.apply_runtime_bootstrap_plan(plan, environ=env)
        monkeypatch.setitem(sys.modules, "torch", object())
        assert env["CUDA_VISIBLE_DEVICES"] == "3"
        assert runtime.require_worker_allocation().accelerators["gpu"] == (3,)


def test_worker_activation_can_select_strict_enforcement_over_process_baseline():
    allocation = runtime.RuntimeAllocationView(cpus=(0,))
    spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "device_visibility": {"policy": "assigned"}})

    with runtime.activate(mode=runtime.RuntimeMode.WORKER, allocation=allocation, spec=spec, restore_environ=True, enforcement="strict") as state:
        assert state.mode is runtime.RuntimeMode.WORKER
        assert runtime.enforcement() is runtime.RuntimeEnforcement.STRICT


def test_materialization_guard_requires_runtime_allocation():
    with runtime.enter_runtime(runtime.RuntimeMode.ORCHESTRATOR, enforcement="strict"):
        with pytest.raises(NoAllocationError):
            runtime.require_allocation("materialize object")
