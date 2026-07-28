"""Facade transition safety contracts."""

from __future__ import annotations

import pytest

from dryml import session
from dryml._framework_imports import finder
from dryml.runtime import RuntimeAllocationView, RuntimeEnforcement, RuntimeMode, RuntimeState, active_runtime, enter_runtime
from dryml.runtime.errors import RuntimeTransitionError
from dryml.runtime.frameworks import FrameworkRegistration, framework_registry
from dryml.runtime.publication import PublicationService
from dryml.worlds import LocalResourceInventory


@pytest.fixture(autouse=True)
def isolated_session(monkeypatch):
    import dryml.session.state as state

    affinity = {0, 1}
    service = PublicationService(
        environ={},
        affinity_getter=lambda: affinity,
        affinity_setter=lambda cpus: (affinity.clear(), affinity.update(cpus)),
    )
    service.initialize(RuntimeState(enforcement=RuntimeEnforcement.OFF))
    monkeypatch.setattr(state, "publication", service)
    import dryml.runtime.context as context

    monkeypatch.setattr(context, "publication", service)
    monkeypatch.setattr(state, "local_inventory", lambda: LocalResourceInventory((0, 1), {"gpu": (0, 1)}, memory=4 * 1024**3))


def test_exact_allocation_requires_an_unambiguous_process_and_respects_inventory():
    allocation = {
        "schema": "dryml.world_allocation.v1",
        "payload": {"roles": {"worker": [
            {"replica": 0, "rank": 0, "local_rank": 0, "resources": {"cpus": [0], "accelerators": {}}},
            {"replica": 1, "rank": 1, "local_rank": 1, "resources": {"cpus": [1], "accelerators": {}}},
        ]}},
    }
    with pytest.raises(ValueError, match="selectors"):
        session.allocate_world(allocation)
    snapshot = session.allocate_world(allocation, role="worker", replica=1)
    assert snapshot.allocation.process.cpus == (1,)


def test_inventory_drift_restages_without_process_effects(monkeypatch):
    import dryml.session.state as state

    first = LocalResourceInventory((0, 1), {}, memory=4 * 1024**3)
    second = LocalResourceInventory((0,), {}, memory=4 * 1024**3)
    values = iter((first, second, second, second))
    monkeypatch.setattr(state, "local_inventory", lambda: next(values))

    snapshot = session.manage()

    assert snapshot.allocation.process.cpus == (0,)


def test_late_framework_import_rejects_visibility_change_without_mutating(monkeypatch):
    import dryml.session.state as state

    session.manage(cpus=1)
    before = session.current()
    monkeypatch.setattr(state, "_loaded_framework_roots", lambda: ("torch",))

    with pytest.raises(RuntimeError, match="restart"):
        session.manage(cpus=2)
    assert session.current() == before


def test_active_sessions_publish_all_builtin_adapter_plans_and_pending_statuses():
    import sys
    import dryml.session.state as state

    before = set(sys.modules)
    snapshot = session.set_mode(mode="orchestrator")
    results = state.publication.current().metadata["framework_results"]

    assert {"tensorflow", "torch", "jax"}.issubset(results)
    assert all(name not in sys.modules or name in before for name in ("tensorflow", "torch", "jax", "jaxlib"))
    assert snapshot.statuses["tensorflow:tensorflow:visibility"] == "pending-import"
    assert snapshot.statuses["torch:torch:threads"] == "pending-import"
    assert snapshot.statuses["jax:jaxlib:allocator"] == "pending-import"
    with pytest.raises(RuntimeError, match="frozen"):
        framework_registry.register(FrameworkRegistration("late", ("dryml_late_framework",), "dryml.tf.runtime:adapter"))


def test_adapter_plan_change_after_a_framework_import_fails_without_publication(monkeypatch):
    import dryml.session.state as state

    session.set_mode("orchestrator")
    before = session.current()
    monkeypatch.setattr(state, "_loaded_framework_roots", lambda: ("torch",))

    with pytest.raises(RuntimeTransitionError, match="restart"):
        session.manage(cpus=1)

    assert session.current() == before


def test_reset_deactivates_builtin_plans_without_removing_the_passive_finder():
    import dryml.session.state as state

    session.set_mode("orchestrator")
    reset = session.reset()
    metadata = state.publication.current().metadata

    assert reset.mode == "python"
    assert not metadata["session_active"]
    assert not {"tensorflow", "torch", "jax"} & set(metadata["framework_results"])
    assert finder in __import__("sys").meta_path


def test_context_override_restores_the_session_baseline_and_cannot_broaden_visibility():
    session.manage(cpus=1)
    baseline = active_runtime()

    with enter_runtime(RuntimeMode.WORKER, RuntimeAllocationView(cpus=(0,)), enforcement="warn"):
        assert active_runtime().enforcement is RuntimeEnforcement.WARN
    assert active_runtime() == baseline

    with pytest.raises(RuntimeTransitionError, match="broadens session CPU visibility"):
        with enter_runtime(RuntimeMode.WORKER, RuntimeAllocationView(cpus=(2,))):
            pass
