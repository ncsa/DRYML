"""Synthetic capability contracts for optional framework runtime leaves."""

from __future__ import annotations

import sys
import types

import pytest

from dryml.jax.runtime import adapter as jax_adapter
from dryml.runtime.allocation import RuntimeAllocationView
from dryml.runtime.devices import DeviceVisibilityPlan, DeviceVisibilityPolicy
from dryml.runtime.errors import FrameworkImportSafetyError
from dryml.runtime.frameworks import FrameworkBootstrapResult
from dryml.runtime.specs import RuntimeContextSpec
from dryml.tf.runtime import adapter as tensorflow_adapter
from dryml.torch.runtime import adapter as torch_adapter


def _result(*, devices=("0", "1"), limits=None, capacity=None, threads=2, allocator=None):
    return FrameworkBootstrapResult(
        env_updates={"DRYML_U3_PRE": "visible"},
        post_import_threads={"tensorflow": threads, "torch": threads, "jax": threads},
        post_import_interop_threads={"tensorflow": 1, "torch": 1},
        visible_devices={"gpu": devices},
        accelerator_memory={"gpu": limits if limits is not None else {device: 512 * 1024**2 for device in devices}},
        accelerator_capacity={"gpu": capacity if capacity is not None else {device: 1024 * 1024**2 for device in devices}},
        allocator_policy=allocator,
        process_memory=4 * 1024**3,
    )


def test_adapter_results_are_immutable_and_pre_import_only_observes_planned_environment():
    result = _result()
    target = {}

    torch_adapter().validate_before_import(result)
    torch_adapter().apply_pre_import(result, environ=target)

    assert target == {"DRYML_U3_PRE": "visible"}
    with pytest.raises(TypeError):
        result.env_updates["unexpected"] = "mutation"
    with pytest.raises(TypeError):
        result.accelerator_memory["gpu"]["0"] = 1


def test_new_memory_fields_do_not_shift_bootstrap_result_positional_arguments():
    result = FrameworkBootstrapResult({"A": "1"}, {"torch": 2}, (0,), 8)

    assert result.cpu_affinity == (0,)
    assert result.memory_limit == 8
    assert result.post_import_interop_threads == {}
    assert result.visible_devices == {}
    assert result.allocator_policy is None
    assert result.process_memory is None


def test_numeric_device_ids_are_canonicalized_for_framework_controls(monkeypatch):
    calls = []

    class CUDA:
        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def set_per_process_memory_fraction(fraction, device):
            calls.append((fraction, device))

    module = types.ModuleType("torch")
    module.cuda = CUDA
    module.set_num_threads = lambda _value: None
    module.set_num_interop_threads = lambda _value: None
    monkeypatch.setitem(sys.modules, "torch", module)
    result = _result(devices=(0,), limits={0: 512 * 1024**2}, capacity={0: 1024 * 1024**2})

    status = torch_adapter().post_import(result, "torch").statuses

    assert result.accelerator_memory == {"gpu": {"0": 512 * 1024**2}}
    assert calls == [(0.5, 0)]
    assert status["accelerator_memory"] == "framework-configured"


def test_torch_allocator_environment_is_planned_without_importing_torch():
    result = torch_adapter().build_plan(
        RuntimeContextSpec.from_data({"frameworks": {"torch": {"allocator": "backend:cudaMallocAsync"}}}),
        RuntimeAllocationView(cpus=(0,)),
        DeviceVisibilityPlan(DeviceVisibilityPolicy.NONE),
    )

    assert result.env_updates["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync"


def test_tensorflow_applies_visible_logical_devices_threads_and_allocator_locally(monkeypatch):
    calls = []

    class Config:
        physical = ("gpu-0", "gpu-1")
        visible = ()

        class threading:
            @staticmethod
            def set_intra_op_parallelism_threads(value):
                calls.append(("intra", value))

            @staticmethod
            def set_inter_op_parallelism_threads(value):
                calls.append(("inter", value))

        class experimental:
            @staticmethod
            def set_memory_growth(device, value):
                calls.append(("growth", device, value))

        class LogicalDeviceConfiguration:
            def __init__(self, *, memory_limit):
                self.memory_limit = memory_limit

        @classmethod
        def get_physical_devices(cls, kind):
            return cls.physical

        @classmethod
        def set_visible_devices(cls, devices, kind):
            cls.visible = tuple(devices)
            calls.append(("visible", cls.visible, kind))

        @classmethod
        def get_visible_devices(cls, kind):
            return cls.visible

        @staticmethod
        def set_logical_device_configuration(device, configs):
            calls.append(("logical", device, configs[0].memory_limit))

    module = types.ModuleType("tensorflow")
    module.config = Config
    monkeypatch.setitem(sys.modules, "tensorflow", module)

    post = tensorflow_adapter().post_import(_result(allocator="memory_growth"), "tensorflow")

    assert post.statuses["visibility"] == "visibility-enforced"
    assert post.statuses["threads"] == "framework-configured"
    assert post.statuses["process_memory"] == "declarative"
    assert post.statuses["accelerator_memory"] == "framework-configured"
    assert post.statuses["allocator"] == "framework-configured"
    assert ("logical", "gpu-0", 512) in calls
    assert ("logical", "gpu-1", 512) in calls


def test_mandatory_visibility_failure_is_not_downgraded(monkeypatch):
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", module)

    with pytest.raises(FrameworkImportSafetyError, match="visible devices"):
        torch_adapter().post_import(_result(), "torch")


def test_torch_requires_a_visibility_count_api(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    with pytest.raises(FrameworkImportSafetyError, match="cannot prove mandatory CUDA visibility"):
        torch_adapter().post_import(_result(devices=(), limits={}, capacity={}), "torch")


def test_torch_requires_capacity_for_fraction_and_proves_recoverable_allocator_rejection(monkeypatch):
    calls = []

    class CUDA:
        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def set_per_process_memory_fraction(fraction, device):
            calls.append((fraction, device))
            if device == 1:
                raise RuntimeError("fake allocator rejected the fraction")

    module = types.ModuleType("torch")
    module.cuda = CUDA
    module.set_num_threads = lambda value: calls.append(("threads", value))
    module.set_num_interop_threads = lambda value: calls.append(("interop", value))
    monkeypatch.setitem(sys.modules, "torch", module)
    result = torch_adapter().post_import(_result(), "torch")

    assert result.statuses["accelerator_memory:gpu:0"] == "framework-configured"
    assert result.statuses["accelerator_memory:gpu:1"] == "failed"
    assert result.statuses["accelerator_memory"] == "failed"
    assert result.statuses["process_memory"] == "declarative"
    unknown_capacity = _result(capacity={})
    result = torch_adapter().post_import(unknown_capacity, "torch")
    assert result.statuses["accelerator_memory"] == "unsupported"


def test_jaxlib_proves_visibility_from_the_pre_import_environment(monkeypatch):
    module = types.ModuleType("jaxlib")
    monkeypatch.setitem(sys.modules, "jaxlib", module)
    visibility = DeviceVisibilityPlan(
        DeviceVisibilityPolicy.EXPLICIT,
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "XLA_VISIBLE_DEVICES": "",
        },
        {"gpu": ("0", "1"), "rocm": (), "xla": ()},
        remap_assigned=False,
    )
    result = jax_adapter().build_plan(
        RuntimeContextSpec.from_data({"frameworks": {"jax": {}}}),
        RuntimeAllocationView(cpus=(0,)),
        visibility,
    )
    for key, value in visibility.env_updates.items():
        monkeypatch.setenv(key, value)

    pending = jax_adapter().post_import(result, "jaxlib")

    assert pending.statuses["visibility"] == "visibility-enforced"
    assert pending.statuses["threads"] == "pending-import"
    assert pending.statuses["allocator"] == "pending-import"
    assert pending.statuses["accelerator_memory"] == "pending-import"


@pytest.mark.parametrize("failure", ["absent", "plan-mismatch", "readback-mismatch"])
def test_jaxlib_fails_closed_without_consistent_visibility_proof(monkeypatch, failure):
    monkeypatch.setitem(sys.modules, "jaxlib", types.ModuleType("jaxlib"))
    env_updates = {
        "CUDA_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "XLA_VISIBLE_DEVICES": "",
    }
    visible_devices = {"gpu": ("0",), "rocm": (), "xla": ()}
    if failure == "absent":
        env_updates = {}
    elif failure == "plan-mismatch":
        visible_devices["gpu"] = ("1",)
    for key, value in env_updates.items():
        monkeypatch.setenv(key, "different" if failure == "readback-mismatch" and key == "CUDA_VISIBLE_DEVICES" else value)
    result = FrameworkBootstrapResult(env_updates=env_updates, visible_devices=visible_devices)

    with pytest.raises(FrameworkImportSafetyError, match="pre-import visibility"):
        jax_adapter().post_import(result, "jaxlib")


def test_jax_reports_heterogeneous_memory_unsupported(monkeypatch):
    module = types.ModuleType("jax")
    module.devices = lambda kind=None: ("gpu-0", "gpu-1") if kind == "gpu" else ()
    monkeypatch.setitem(sys.modules, "jax", module)
    heterogenous = _result(limits={"0": 256 * 1024**2, "1": 512 * 1024**2})
    post = jax_adapter().post_import(heterogenous, "jax")
    assert post.statuses["visibility"] == "visibility-enforced"
    assert post.statuses["accelerator_memory"] == "unsupported"


def test_adapter_memory_statuses_are_not_an_aggregate_process_cap(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(device_count=lambda: 1, set_per_process_memory_fraction=lambda fraction, device: None)
    torch.set_num_threads = lambda value: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    jax = types.ModuleType("jax")
    jax.devices = lambda kind=None: ("gpu-0",) if kind == "gpu" else ()
    monkeypatch.setitem(sys.modules, "jax", jax)
    result = _result(devices=("0",), limits={"0": 512 * 1024**2}, capacity={"0": 1024 * 1024**2})

    torch_status = torch_adapter().post_import(result, "torch").statuses
    jax_status = jax_adapter().post_import(result, "jax").statuses

    assert torch_status["accelerator_memory"] == "framework-configured"
    assert jax_status["accelerator_memory"] == "framework-configured"
    assert "process_memory" in torch_status and "process_memory" in jax_status
    assert "aggregate_memory" not in torch_status and "aggregate_memory" not in jax_status
