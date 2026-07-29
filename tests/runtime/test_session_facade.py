"""Public session facade contracts."""

from __future__ import annotations

import inspect
from types import MappingProxyType

import pytest

from dryml import session
from dryml.runtime import NoAllocation, RuntimeEnforcement, RuntimeMode, RuntimeState
from dryml.runtime.publication import PublicationService
from dryml.worlds import (
    LocalResourceInventory,
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
    for operation in (session.manage, session.request_world):
        parameters = inspect.signature(operation).parameters
        assert tuple(parameters) == ("cpus", "memory", "gpus", "accelerator_memory")
        assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters.values())
    allocation = inspect.signature(session.allocate_world).parameters
    assert allocation["value"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert allocation["role"].kind is inspect.Parameter.KEYWORD_ONLY
    assert allocation["replica"].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(session.require_env).parameters["requirements"].kind is inspect.Parameter.VAR_POSITIONAL


def test_fresh_inspection_is_python_and_does_not_probe_inventory(monkeypatch):
    import dryml.session.state as state

    monkeypatch.setattr(state, "local_inventory", lambda: (_ for _ in ()).throw(AssertionError("must not probe")))

    snapshot = session.current()

    assert session.mode() == "python"
    assert snapshot.mode == "python"
    assert snapshot.allocation is None
    assert snapshot.runtime.mode is RuntimeMode.ORCHESTRATOR
    assert snapshot.runtime.allocation is NoAllocation


def test_manage_keeps_the_session_allocation_projection_deeply_immutable():
    snapshot = session.manage()

    assert snapshot.mode == "managed"
    assert snapshot.resources.to_data() == {"cpus": 3, "memory": "8GiB"}
    assert snapshot.allocation.process.cpus == (0, 1, 2)
    assert snapshot.allocation.process.accelerators == MappingProxyType({})
    assert snapshot.allocation.process.memory == 8 * 1024**3
    assert snapshot.controls["memory"] == "declarative"
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
    session.request_world(cpus=1, gpus=1)
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


def test_invalid_flat_or_declarative_input_leaves_the_generation_unchanged():
    before = session.manage(cpus=1)
    with pytest.raises(ValueError):
        session.request_world()
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
    session.request_world(cpus=1)
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
