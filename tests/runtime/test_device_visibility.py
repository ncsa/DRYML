import pytest

import dryml.runtime as runtime
from dryml.runtime.errors import DeviceVisibilityError


def test_orchestrator_and_probe_default_to_hidden_devices():
    assert runtime.build_device_visibility_plan(mode=runtime.RuntimeMode.ORCHESTRATOR).env_updates["CUDA_VISIBLE_DEVICES"] == ""
    assert runtime.build_device_visibility_plan(mode=runtime.RuntimeMode.PROBE).env_updates["CUDA_VISIBLE_DEVICES"] == ""


def test_worker_assigned_gpu_visibility_and_apply(monkeypatch):
    allocation = runtime.RuntimeAllocationView(accelerators={"gpu": (2, 4)})
    plan = runtime.build_device_visibility_plan(mode=runtime.RuntimeMode.WORKER, allocation_view=allocation)
    env = {}
    runtime.apply_device_visibility_plan(plan, environ=env)

    assert plan.env_updates == {"CUDA_VISIBLE_DEVICES": "2,4", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "XLA_VISIBLE_DEVICES": ""}
    assert env["CUDA_VISIBLE_DEVICES"] == "2,4"
    assert env["HIP_VISIBLE_DEVICES"] == ""


def test_assigned_cpu_only_worker_hides_all_accelerator_families():
    allocation = runtime.RuntimeAllocationView(cpus=(0, 1))
    plan = runtime.build_device_visibility_plan(mode=runtime.RuntimeMode.WORKER, allocation_view=allocation)

    assert plan.env_updates == {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "XLA_VISIBLE_DEVICES": ""}


def test_inherit_requires_explicit_opt_in_and_inline_requires_explicit_choice():
    with pytest.raises(DeviceVisibilityError):
        runtime.build_device_visibility_plan(mode=runtime.RuntimeMode.INLINE, policy="inherit")
    plan = runtime.build_device_visibility_plan(mode=runtime.RuntimeMode.INLINE, policy="explicit", explicit_devices={"gpu": [1]})
    assert plan.env_updates["CUDA_VISIBLE_DEVICES"] == "1"
    assert plan.env_updates["HIP_VISIBLE_DEVICES"] == ""


def test_visibility_plans_build_without_framework_imports():
    plan = runtime.build_device_visibility_plan(mode="orchestrator")
    assert plan.policy is runtime.DeviceVisibilityPolicy.NONE
