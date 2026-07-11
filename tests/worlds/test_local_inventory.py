from __future__ import annotations

import math
import json
import subprocess
import sys
from itertools import count

import pytest

from dryml.worlds import LocalResourceInventory, local_inventory
from dryml.worlds.errors import ResourceValidationError


def test_inventory_round_trip_is_deterministic():
    inventory = LocalResourceInventory((3, 1), {"gpu": ("1", 0)}, memory=1024, metadata={"policy": "test"})

    assert inventory.cpus == (1, 3)
    assert LocalResourceInventory.from_data(inventory.to_data()) == inventory
    assert inventory.summary()["accelerator_counts"] == {"gpu": 2}


@pytest.mark.parametrize("identifier", ("", "0,1", "GPU\x00unsafe", -1, "-1", "-7", "all", "none", "VOID"))
def test_inventory_rejects_unsafe_accelerator_visibility_identifiers(identifier):
    with pytest.raises(ResourceValidationError, match="accelerator identifier"):
        LocalResourceInventory((0,), {"gpu": (identifier,)})
    with pytest.raises(ResourceValidationError, match="accelerator identifier"):
        LocalResourceInventory.from_data({"cpus": [0], "accelerators": {"gpu": [identifier]}})


@pytest.mark.parametrize("identifier", ("-1", "all", "none", "VOID"))
def test_explicit_inventory_rejects_disabled_gpu_visibility_identifiers(identifier):
    with pytest.raises(ResourceValidationError, match="accelerator identifier"):
        local_inventory(environ={"DRYML_LOCAL_ACCELERATORS": f"gpu={identifier}"})


def test_inventory_accelerator_string_identifier_reaches_visibility_allocation():
    from dryml.dispatch.local_world import allocate_local_world
    from dryml.runtime import RuntimeAllocationView, RuntimeMode, build_device_visibility_plan

    plan = allocate_local_world(
        {"roles": {"main": {"process": {"resources": {"accelerators": {"gpu": 1}}}}}},
        inventory=LocalResourceInventory((0,), {"gpu": ("GPU-01234567-89ab-cdef-0123-456789abcdef",)}),
    )
    allocation = plan.world_allocation.roles["main"][0]
    visibility = build_device_visibility_plan(
        mode=RuntimeMode.WORKER,
        allocation_view=RuntimeAllocationView(accelerators=allocation.accelerators),
    )

    assert visibility.env_updates["CUDA_VISIBLE_DEVICES"] == "GPU-01234567-89ab-cdef-0123-456789abcdef"


def test_lightweight_inventory_uses_explicit_accelerator_override_without_mutation():
    environment = {"DRYML_LOCAL_ACCELERATORS": "gpu=2,0;fpga=a"}

    inventory = local_inventory(environ=environment)

    assert inventory.accelerators == {"fpga": ("a",), "gpu": (0, 2)}
    assert environment == {"DRYML_LOCAL_ACCELERATORS": "gpu=2,0;fpga=a"}


def test_lightweight_inventory_uses_default_device_root(monkeypatch):
    import dryml.worlds.inventory as inventory_module

    roots = []
    monkeypatch.setattr(
        inventory_module,
        "_accelerators_from_device_root",
        lambda _environ, root, _diagnostics: roots.append(root) or {},
    )

    local_inventory(environ={})

    assert roots == ["/dev"]


def test_explicit_accelerator_groups_may_not_repeat():
    with pytest.raises(ResourceValidationError, match="repeats an accelerator group"):
        local_inventory(environ={"DRYML_LOCAL_ACCELERATORS": "gpu=0;gpu=1"})


@pytest.mark.parametrize("timeout", (math.inf, math.nan))
def test_inventory_rejects_nonfinite_external_timeout(timeout):
    with pytest.raises(ResourceValidationError, match="timeout must be positive"):
        local_inventory(policy="external", timeout=timeout, command_runner=lambda *_args, **_kwargs: "")


def test_inventory_rejects_unbounded_or_non_json_metadata():
    with pytest.raises(ResourceValidationError, match="floats must be finite"):
        LocalResourceInventory((0,), metadata={"value": math.nan})
    with pytest.raises(ResourceValidationError, match="string exceeds"):
        LocalResourceInventory((0,), metadata={"value": "x" * 4097})


def test_inventory_rejects_aggregate_metadata_expansion():
    leaf = {f"leaf-{index}": "value" for index in range(11)}
    branch = {f"branch-{index}": leaf for index in range(11)}
    metadata = {f"outer-{index}": branch for index in range(11)}

    with pytest.raises(ResourceValidationError, match="aggregate bounded limit"):
        LocalResourceInventory((0,), metadata=metadata)


def test_external_inventory_bounds_runner_output(tmp_path):
    output = "\n".join(str(index) for index in range(200))
    inventory = local_inventory(
        policy="external",
        environ={},
        device_root=tmp_path,
        command_runner=lambda *_args, **_kwargs: type("Result", (), {"returncode": 0, "stdout": output})(),
    )

    assert len(inventory.accelerators["gpu"]) == 128
    assert "external accelerator identifiers were truncated" in inventory.metadata["diagnostics"]


def test_explicit_accelerator_override_is_not_broadened_by_external_discovery():
    inventory = local_inventory(
        policy="external",
        environ={"DRYML_LOCAL_ACCELERATORS": "gpu=0"},
        command_runner=lambda *_args, **_kwargs: "0\n1\n",
    )

    assert inventory.accelerators == {"gpu": (0,)}


@pytest.mark.parametrize(
    "visibility,expected",
    (("1", {"gpu": (1,)}), ("none", {}), ("GPU-opaque", {})),
)
def test_external_inventory_never_broadens_inherited_gpu_visibility(visibility, expected, tmp_path):
    inventory = local_inventory(
        policy="external",
        environ={"CUDA_VISIBLE_DEVICES": visibility},
        device_root=tmp_path,
        command_runner=lambda *_args, **_kwargs: "0\n1\n",
    )

    assert inventory.accelerators == expected


@pytest.mark.parametrize(
    "environ,expected",
    (
        ({"CUDA_VISIBLE_DEVICES": "all", "NVIDIA_VISIBLE_DEVICES": "none"}, {}),
        ({"CUDA_VISIBLE_DEVICES": "0,1", "NVIDIA_VISIBLE_DEVICES": "1,2"}, {"gpu": (1,)}),
    ),
)
def test_external_inventory_intersects_cuda_and_nvidia_visibility(environ, expected, tmp_path):
    inventory = local_inventory(
        policy="external",
        environ=environ,
        device_root=tmp_path,
        command_runner=lambda *_args, **_kwargs: "0\n1\n2\n",
    )

    assert inventory.accelerators == expected


@pytest.mark.parametrize(
    "output",
    (
        type("Result", (), {"returncode": 1, "stdout": ""})(),
        type("Result", (), {"returncode": 0, "stdout": "not-a-device"})(),
    ),
)
def test_external_inventory_failures_are_diagnostic_only(output, tmp_path):
    inventory = local_inventory(policy="external", environ={}, device_root=tmp_path, command_runner=lambda *_args, **_kwargs: output)

    assert "gpu" not in inventory.accelerators
    assert any(item.startswith("external accelerator discovery unavailable") for item in inventory.metadata["diagnostics"])


@pytest.mark.parametrize("failure", (FileNotFoundError(), TimeoutError()))
def test_external_inventory_missing_or_timed_out_runner_is_diagnostic_only(failure, tmp_path):
    inventory = local_inventory(
        policy="external",
        environ={},
        device_root=tmp_path,
        command_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    assert inventory.accelerators == {}
    assert any(item.startswith("external accelerator discovery unavailable") for item in inventory.metadata["diagnostics"])


def test_external_inventory_discards_partial_output_and_negative_identifiers(tmp_path):
    partial = " " * ((64 * 1024) - 1) + "12\n"
    inventory = local_inventory(policy="external", device_root=tmp_path, command_runner=lambda *_args, **_kwargs: partial)
    negative = local_inventory(policy="external", device_root=tmp_path, command_runner=lambda *_args, **_kwargs: "-1\n")

    assert "gpu" not in inventory.accelerators
    assert "gpu" not in negative.accelerators


@pytest.mark.parametrize("field,value", (("accelerators", []), ("accelerators", ""), ("metadata", []), ("metadata", "")))
def test_inventory_rejects_falsey_non_mapping_serialized_fields(field, value):
    with pytest.raises(ResourceValidationError):
        LocalResourceInventory.from_data({"cpus": [0], field: value})


def test_inventory_import_path_does_not_load_framework_modules():
    command = (
        "import json, sys; "
        "from dryml.worlds import local_inventory; "
        "local_inventory(environ={}); "
        "print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {'torch', 'tensorflow', 'jax', 'keras', 'cupy'})))"
    )
    output = subprocess.check_output([sys.executable, "-c", command], text=True)

    assert json.loads(output) == []


def test_empty_cpu_affinity_is_not_reported_as_cpu_zero(monkeypatch):
    import dryml.worlds.inventory as inventory_module

    monkeypatch.setattr(inventory_module.os, "sched_getaffinity", lambda _: set(), raising=False)

    with pytest.raises(ResourceValidationError, match="no executable CPUs"):
        local_inventory(environ={})


def test_large_discovered_cpu_affinity_is_conservatively_bounded(monkeypatch):
    import dryml.worlds.inventory as inventory_module

    monkeypatch.setattr(inventory_module.os, "sched_getaffinity", lambda _: set(range(4097)), raising=False)

    inventory = local_inventory(environ={}, device_root=None)

    assert len(inventory.cpus) == 4096
    assert "CPU discovery exceeded the bounded identifier limit" in inventory.metadata["diagnostics"]


def test_large_cpu_count_fallback_is_conservatively_bounded(monkeypatch):
    import dryml.worlds.inventory as inventory_module

    monkeypatch.setattr(inventory_module.os, "sched_getaffinity", lambda _: (_ for _ in ()).throw(OSError()), raising=False)
    monkeypatch.setattr(inventory_module.os, "cpu_count", lambda: 4097)

    inventory = local_inventory(environ={}, device_root=None)

    assert len(inventory.cpus) == 4096
    assert "CPU discovery exceeded the bounded identifier limit" in inventory.metadata["diagnostics"]


def test_lightweight_inventory_discovers_affinity_and_memory(monkeypatch):
    import dryml.worlds.inventory as inventory_module

    monkeypatch.setattr(inventory_module.os, "sched_getaffinity", lambda _: {5, 2}, raising=False)
    monkeypatch.setattr(inventory_module.Path, "exists", lambda path: False)
    monkeypatch.setattr(
        inventory_module.Path,
        "read_text",
        lambda path, **_kwargs: "MemAvailable:       2048 kB\n" if path.as_posix() == "/proc/meminfo" else "",
    )

    inventory = local_inventory(environ={}, device_root=None)

    assert inventory.cpus == (2, 5)
    assert inventory.memory == 2048 * 1024
    assert inventory.metadata["memory_source"] == "proc_meminfo"


def test_lightweight_inventory_uses_cpu_count_when_affinity_is_unavailable(monkeypatch):
    import dryml.worlds.inventory as inventory_module

    monkeypatch.setattr(inventory_module.os, "sched_getaffinity", lambda _: (_ for _ in ()).throw(OSError()), raising=False)
    monkeypatch.setattr(inventory_module.os, "cpu_count", lambda: 3)

    inventory = local_inventory(environ={}, device_root=None)

    assert inventory.cpus == (0, 1, 2)
    assert inventory.metadata["cpu_source"] == "cpu_count"


def test_device_root_accelerators_respect_numeric_visibility(tmp_path, monkeypatch):
    import dryml.worlds.inventory as inventory_module

    (tmp_path / "nvidia0").touch()
    (tmp_path / "nvidia1").touch()
    monkeypatch.setattr(inventory_module.Path, "is_char_device", lambda path: path.name.startswith("nvidia"))

    inventory = local_inventory(environ={"CUDA_VISIBLE_DEVICES": "1"}, device_root=tmp_path)

    assert inventory.accelerators == {"gpu": (1,)}


def test_device_root_accelerators_honor_all_visibility(tmp_path, monkeypatch):
    import dryml.worlds.inventory as inventory_module

    (tmp_path / "nvidia0").touch()
    (tmp_path / "nvidia1").touch()
    monkeypatch.setattr(inventory_module.Path, "is_char_device", lambda path: path.name.startswith("nvidia"))

    inventory = local_inventory(environ={"NVIDIA_VISIBLE_DEVICES": "all"}, device_root=tmp_path)

    assert inventory.accelerators == {"gpu": (0, 1)}


def test_device_root_ignores_regular_nvidia_named_files(tmp_path):
    (tmp_path / "nvidia0").touch()

    inventory = local_inventory(environ={}, device_root=tmp_path)

    assert inventory.accelerators == {}


def test_device_root_enumeration_is_bounded(tmp_path):
    for index in range(257):
        (tmp_path / f"nvidia{index}").touch()

    inventory = local_inventory(environ={}, device_root=tmp_path)

    assert inventory.accelerators == {}
    assert "device-file accelerator discovery exceeded the bounded entry limit" in inventory.metadata["diagnostics"]


def test_explicit_accelerator_override_is_bounded():
    values = ",".join(str(value) for value in range(129))

    with pytest.raises(ResourceValidationError, match="too many accelerator identifiers"):
        local_inventory(environ={"DRYML_LOCAL_ACCELERATORS": f"gpu={values}"})


def test_injected_inventory_identifier_iterables_are_bounded():
    with pytest.raises(ResourceValidationError, match="CPUs exceed"):
        LocalResourceInventory(count())
    with pytest.raises(ResourceValidationError, match="accelerator identifiers exceed"):
        LocalResourceInventory((0,), {"gpu": count()})


@pytest.mark.parametrize("cpus", (None, 1))
def test_inventory_serialization_rejects_malformed_cpu_values(cpus):
    with pytest.raises(ResourceValidationError, match="CPUs must be a sequence"):
        LocalResourceInventory.from_data({"cpus": cpus})


def test_inventory_serialization_rejects_mixed_unknown_keys_structurally():
    with pytest.raises(ResourceValidationError, match="unknown fields"):
        LocalResourceInventory.from_data({"cpus": (0,), "unexpected": True, 1: True})
