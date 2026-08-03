"""Public session facade contracts."""

from __future__ import annotations

import inspect
from types import MappingProxyType

import pytest

from dryml import session
from dryml.environments import CurrentEnvironmentSpec, ContainerEnvironmentSpec, PythonExecutableSpec
from dryml.runtime import NoAllocation, RequirementAxes, RuntimeAllocationView, RuntimeContextSpec, RuntimeEnforcement, RuntimeMode, RuntimeState
from dryml.runtime.publication import PublicationService
from dryml.worlds import (
    LocalResourceInventory,
    WorldSpec,
    WorldAllocation,
    attach_world_allocation_id,
    make_world_allocation_spec,
)


@pytest.fixture(autouse=True)
def isolated_session(monkeypatch):
    """Use one deterministic publication authority for each facade test."""

    import dryml.session.state as state

    affinity = {0, 1, 2}
    service = PublicationService(
        environ={},
        affinity_getter=lambda: affinity,
        affinity_setter=lambda cpus: (affinity.clear(), affinity.update(cpus)),
    )
    service.initialize(RuntimeState(enforcement=RuntimeEnforcement.OFF))
    monkeypatch.setattr(state, "publication", service)
    monkeypatch.setattr(
        state,
        "local_inventory",
        lambda: LocalResourceInventory((0, 1, 2), {"gpu": (0,)}, memory=8 * 1024**3),
    )


def test_flat_api_has_the_closed_public_signatures():
    assert tuple(inspect.signature(session.mode).parameters) == ()
    assert tuple(inspect.signature(session.current).parameters) == ()
    set_mode = inspect.signature(session.set_mode).parameters
    assert tuple(set_mode) == ("mode",)
    assert set_mode["mode"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for operation in (session.manage, session.worker_world_request):
        parameters = inspect.signature(operation).parameters
        assert tuple(parameters) == ("cpus", "memory", "gpus", "accelerator_memory")
        assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters.values())
    worker_environment = inspect.signature(session.worker_env_request).parameters
    assert tuple(worker_environment) == ("value",)
    assert worker_environment["value"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert not hasattr(session, "request_world")
    allocation = inspect.signature(session.allocate_world).parameters
    assert allocation["value"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert allocation["role"].kind is inspect.Parameter.KEYWORD_ONLY
    assert allocation["replica"].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(session.require_env).parameters["requirements"].kind is inspect.Parameter.VAR_POSITIONAL
    axes = inspect.signature(session.enforce_requirements).parameters
    assert tuple(axes) == ("environment", "world", "runtime")
    assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in axes.values())


def test_worker_session_publication_preserves_accelerator_memory():
    import dryml.session.state as state

    allocation = RuntimeAllocationView(
        world_allocation_id="worldalloc-v1-test",
        role="worker",
        replica=0,
        rank=0,
        local_rank=0,
        cpus=(0,),
        accelerators={"gpu": ("gpu-a",)},
        accelerator_memory={"gpu": {"gpu-a": 1024}},
    )

    snapshot = state.publish_worker_session(
        environment=CurrentEnvironmentSpec(),
        world=WorldSpec.from_data(
            {"roles": {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1, "accelerators": {"gpu": 1}}}}}}
        ),
        runtime_spec=RuntimeContextSpec(
            mode=RuntimeMode.WORKER,
            world_allocation_id="worldalloc-v1-test",
        ),
        allocation=allocation,
        requirement_policy="strict",
        requirement_axes=RequirementAxes.all(),
    )

    assert snapshot.allocation.process.accelerator_memory == {
        "gpu": {"gpu-a": 1024}
    }
    assert snapshot.runtime.allocation.accelerator_memory == {
        "gpu": {"gpu-a": 1024}
    }
    assert all(record.kind != "cpu_affinity" for record in state.publication.effect_journal())


def test_fresh_inspection_is_python_and_does_not_probe_inventory(monkeypatch):
    import dryml.session.state as state

    monkeypatch.setattr(state, "local_inventory", lambda: (_ for _ in ()).throw(AssertionError("must not probe")))

    snapshot = session.current()

    assert session.mode() == "python"
    assert snapshot.mode == "python"
    assert snapshot.allocation is None
    assert snapshot.runtime.mode is RuntimeMode.NONE
    assert snapshot.runtime.allocation is NoAllocation
    assert snapshot.requirement_axes.to_data() == []


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [
        ((False, False, False), []),
        ((True, False, False), ["environment"]),
        ((False, True, False), ["world"]),
        ((False, False, True), ["runtime"]),
        ((True, True, False), ["environment", "world"]),
        ((True, False, True), ["environment", "runtime"]),
        ((False, True, True), ["world", "runtime"]),
        ((True, True, True), ["environment", "world", "runtime"]),
    ],
)
def test_requirement_axes_are_canonical_and_atomically_replaced(enabled, expected):
    environment, world, runtime_axis = enabled
    snapshot = session.enforce_requirements(environment=environment, world=world, runtime=runtime_axis)

    assert snapshot.requirement_axes.to_data() == expected
    assert snapshot.to_data()["requirement_axes"] == expected
    configuration = session.normalize_configuration(
        mode="python",
        requirement_axes={"runtime": runtime_axis, "environment": environment, "world": world},
    )
    assert configuration.requirement_axes.to_data() == expected
    assert configuration.to_data()["requirement_axes"] == expected


def test_requirement_axes_reject_malformed_replacement_without_mutating_generation():
    before = session.enforce_requirements(environment=True, world=False, runtime=True)

    with pytest.raises(ValueError):
        session.configure(mode="managed", requirement_axes={"environment": True, "world": False})
    with pytest.raises(ValueError):
        session.configure(
            mode="managed",
            requirement_axes={"environment": True, "world": False, "runtime": 1},
        )

    assert session.current() == before


def test_configure_uses_mode_axis_defaults_when_omitted_and_replaces_explicit_axes():
    session.enforce_requirements(environment=True, world=False, runtime=False)

    managed = session.configure(mode="managed")
    python = session.configure(mode="python")
    orchestrator = session.configure(
        mode="orchestrator",
        requirement_axes={"runtime": False, "environment": True, "world": False},
    )

    assert managed.requirement_axes.to_data() == ["environment", "world", "runtime"]
    assert python.requirement_axes.to_data() == []
    assert orchestrator.requirement_axes.to_data() == ["environment"]


def test_manage_keeps_the_session_allocation_projection_deeply_immutable():
    import dryml.session.state as state

    snapshot = session.manage()

    assert snapshot.mode == "managed"
    assert snapshot.resources.to_data() == {"cpus": 3, "memory": "8GiB"}
    assert snapshot.allocation.process.cpus == (0, 1, 2)
    assert snapshot.allocation.process.accelerators == MappingProxyType({})
    assert snapshot.allocation.process.memory == 8 * 1024**3
    assert snapshot.controls["memory"] == "declarative"
    assert all(record.kind != "cpu_affinity" for record in state.publication.effect_journal())
    with pytest.raises(TypeError):
        snapshot.controls["memory"] = "changed"
    with pytest.raises(TypeError):
        snapshot.statuses["visibility"] = "failed"
    snapshot.runtime.allocation.accelerators["gpu"] = (0,)
    assert snapshot.allocation.process.accelerators == MappingProxyType({})
    with pytest.raises(TypeError):
        snapshot.allocation.process.accelerators["gpu"] = (0,)


def test_operation_table_replaces_only_its_owned_category_and_configure_is_complete_replacement():
    session.manage(cpus=1)
    session.worker_world_request(cpus=1, gpus=1)
    session.require_env("dryml>=0", python=">=3")

    orchestrator = session.set_mode(mode="orchestrator")
    assert orchestrator.allocation is None
    assert orchestrator.requested_world is not None
    assert orchestrator.environment.requirements == ("dryml>=0",)

    managed = session.set_mode("managed")
    assert managed.allocation is not None
    assert managed.allocation.process.cpus == (0, 1, 2)

    replaced = session.configure(mode="orchestrator")
    assert replaced.requested_world is None
    assert replaced.environment.requirements == ()
    assert replaced.allocation is None


def test_worker_environment_request_is_a_separate_typed_or_mapping_candidate():
    typed = PythonExecutableSpec("/opt/worker/python")
    first = session.worker_env_request(typed)
    second = session.worker_env_request(typed.to_data())

    assert first.requested_environment == typed
    assert second.requested_environment == typed
    assert second.environment.requirements == ()
    assert session.worker_env_request(ContainerEnvironmentSpec("example/image")).requested_environment.kind == "container"

    before = session.current()
    with pytest.raises(ValueError):
        session.worker_env_request({"kind": "unknown"})
    assert session.current() == before


def test_worker_requests_survive_flat_updates_and_configure_replaces_them():
    session.worker_env_request(PythonExecutableSpec("/opt/worker/python"))
    session.worker_world_request(cpus=2)
    preserved = session.require_env("dryml>=0")
    assert preserved.requested_environment.kind == "python"
    assert preserved.requested_world is not None

    replaced = session.configure(mode="python")
    assert replaced.requested_environment is None
    assert replaced.requested_world is None


def test_invalid_flat_or_declarative_input_leaves_the_generation_unchanged():
    before = session.manage(cpus=1)
    with pytest.raises(ValueError):
        session.worker_world_request()
    with pytest.raises(ValueError):
        session.configure(mode="managed", resources={"cpus": 1}, allocation={"value": {}})
    with pytest.raises(TypeError):
        session.manage(1)
    with pytest.raises(ValueError):
        session.require_env(True)
    assert session.current() == before


def test_configure_exact_allocation_is_revalidated_against_the_observed_inventory():
    allocation = {
        "value": {
            "schema": "dryml.world_allocation.v1",
            "schema_version": 1,
            "kind": "local_allocation",
            "payload": {"roles": {"main": [{
                "replica": 0, "rank": 0, "local_rank": 0,
                "resources": {"cpus": [9], "accelerators": {}},
            }]}},
        }
    }
    before = session.current()
    with pytest.raises(ValueError, match="broadens inherited"):
        session.configure(mode="managed", allocation=allocation)
    assert session.current() == before


def test_allocate_world_accepts_canonical_output_with_id_and_metadata():
    allocation = attach_world_allocation_id(make_world_allocation_spec(
        {
            "main": [{
                "replica": 0,
                "rank": 0,
                "local_rank": 0,
                "resources": {"cpus": [2], "accelerators": {}},
            }]
        },
        metadata={"scheduler": "local"},
    ))

    snapshot = session.allocate_world(allocation)

    assert allocation["id"].startswith("worldalloc-v1-")
    assert allocation["metadata"] == {"scheduler": "local"}
    assert snapshot.allocation.process.cpus == (2,)


def test_configure_accepts_typed_world_allocation_value():
    allocation = WorldAllocation.from_data({
        "roles": {
            "main": [{
                "replica": 0,
                "rank": 0,
                "local_rank": 0,
                "resources": {"cpus": [1], "accelerators": {}},
            }]
        }
    })

    snapshot = session.configure(mode="managed", allocation={"value": allocation})

    assert snapshot.allocation.role == "main"
    assert snapshot.allocation.process.cpus == (1,)
    with pytest.raises(TypeError):
        snapshot.allocation.process.metadata["mutated"] = True


def test_requirements_merge_atomically_and_reset_clears_all_categories():
    session.manage(cpus=1)
    session.worker_world_request(cpus=1)
    session.require_env("dryml>=0", python=">=3")
    merged = session.require_env("dryml<99", capabilities=("dryml.environments.v1",))
    assert merged.environment.requirements == ("dryml<99,>=0",)
    before = session.current()
    with pytest.raises(ValueError):
        session.require_env("dryml<0")
    assert session.current() == before

    reset = session.reset()
    assert reset.mode == "python"
    assert reset.allocation is None
    assert reset.requested_world is None
    assert reset.environment.requirements == ()
