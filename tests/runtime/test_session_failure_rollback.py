"""Facade rollback and fail-closed contracts."""

from __future__ import annotations

import pytest

from dryml import session
from dryml.runtime import NoAllocation, RuntimeEnforcement, RuntimeMode, RuntimeState
from dryml.runtime.publication import FrameworkAdmission
from dryml.runtime.publication import PublicationService
from dryml.runtime.frameworks import framework_registry
from dryml.worlds import LocalResourceInventory


def test_interrupted_facade_transition_keeps_the_prior_snapshot(monkeypatch):
    import dryml.session.state as state

    affinity = {0, 1}
    environ = {}
    service = PublicationService(
        environ=environ,
        affinity_getter=lambda: affinity,
        affinity_setter=lambda cpus: (affinity.clear(), affinity.update(cpus)),
    )
    service.initialize(RuntimeState(enforcement=RuntimeEnforcement.OFF))
    monkeypatch.setattr(state, "publication", service)
    monkeypatch.setattr(state, "local_inventory", lambda: LocalResourceInventory((0, 1), {}, memory=None))
    before = session.current()
    monkeypatch.setattr(service, "_publish", lambda _generation: (_ for _ in ()).throw(KeyboardInterrupt("stop")))

    with pytest.raises(KeyboardInterrupt, match="stop"):
        session.manage(cpus=1)

    assert session.current() == before
    assert environ == {}


def test_reset_restores_session_owned_visibility_only(monkeypatch):
    import dryml.session.state as state

    affinity = {0}
    environ = {"CUDA_VISIBLE_DEVICES": "7"}
    service = PublicationService(
        environ=environ,
        affinity_getter=lambda: affinity,
        affinity_setter=lambda cpus: (affinity.clear(), affinity.update(cpus)),
    )
    service.initialize(RuntimeState(enforcement=RuntimeEnforcement.OFF))
    monkeypatch.setattr(state, "publication", service)
    monkeypatch.setattr(state, "local_inventory", lambda: LocalResourceInventory((0,), {}, memory=None))
    monkeypatch.setattr(state, "_loaded_framework_roots", lambda: ())

    session.manage()
    assert environ["CUDA_VISIBLE_DEVICES"] == ""
    session.reset()

    assert environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_failed_facade_transition_restores_the_prior_registry_freeze(monkeypatch):
    import dryml.session.state as state

    service = PublicationService(environ={}, affinity_getter=lambda: {0}, affinity_setter=lambda _cpus: None)
    service.initialize(RuntimeState(enforcement=RuntimeEnforcement.OFF))
    monkeypatch.setattr(state, "publication", service)
    monkeypatch.setattr(state, "local_inventory", lambda: LocalResourceInventory((0,), {}, memory=None))
    monkeypatch.setattr(service, "_publish", lambda _generation: (_ for _ in ()).throw(RuntimeError("publish failed")))
    with framework_registry._lock:
        framework_registry._frozen = False

    with pytest.raises(RuntimeError, match="publish failed"):
        session.set_mode("orchestrator")

    assert not framework_registry._frozen


def test_interrupted_validator_cannot_leak_a_provisional_registry_freeze(monkeypatch):
    import dryml.session.state as state

    service = PublicationService(environ={}, affinity_getter=lambda: {0}, affinity_setter=lambda _cpus: None)
    service.initialize(RuntimeState(enforcement=RuntimeEnforcement.OFF))
    monkeypatch.setattr(state, "publication", service)
    monkeypatch.setattr(state, "local_inventory", lambda: LocalResourceInventory((0,), {}, memory=None))
    original_freeze = framework_registry.freeze

    def freeze_then_interrupt():
        original_freeze()
        raise KeyboardInterrupt("validator interrupted")

    monkeypatch.setattr(framework_registry, "freeze", freeze_then_interrupt)
    with framework_registry._lock:
        framework_registry._frozen = False

    with pytest.raises(KeyboardInterrupt, match="validator interrupted"):
        session.set_mode("orchestrator")

    assert not framework_registry._frozen
    assert session.mode() == "python"


def test_managed_to_orchestrator_restores_nonvisibility_effects(monkeypatch):
    import dryml.session.state as state

    affinity = {0, 1}
    environ = {"OMP_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": "7"}
    service = PublicationService(
        environ=environ,
        affinity_getter=lambda: affinity,
        affinity_setter=lambda cpus: (affinity.clear(), affinity.update(cpus)),
    )
    service.initialize(RuntimeState(enforcement=RuntimeEnforcement.OFF))
    monkeypatch.setattr(state, "publication", service)
    monkeypatch.setattr(state, "local_inventory", lambda: LocalResourceInventory((0, 1), {}, memory=None))
    monkeypatch.setattr(state, "_loaded_framework_roots", lambda: ())

    session.manage(cpus=1)
    assert affinity == {0, 1}
    assert environ["OMP_NUM_THREADS"] == "1"

    snapshot = session.set_mode("orchestrator")

    assert snapshot.mode == "orchestrator"
    assert affinity == {0, 1}
    assert environ["OMP_NUM_THREADS"] == "8"
    assert environ["CUDA_VISIBLE_DEVICES"] == ""


def test_terminal_framework_failure_projects_a_sanitized_orchestrator(monkeypatch):
    import dryml.session.state as state

    affinity = {0}
    service = PublicationService(
        environ={},
        affinity_getter=lambda: affinity,
        affinity_setter=lambda cpus: (affinity.clear(), affinity.update(cpus)),
    )
    service.initialize(RuntimeState(enforcement=RuntimeEnforcement.OFF))
    monkeypatch.setattr(state, "publication", service)
    monkeypatch.setattr(state, "local_inventory", lambda: LocalResourceInventory((0,), {}, memory=None))
    session.manage(cpus=1)
    session.enforce_requirements(environment=True, world=False, runtime=False)
    current = service.current()
    admission = FrameworkAdmission(
        current.number,
        current.metadata["control_epoch"],
        current.metadata["framework_registry_revision"],
        "torch",
        "torch",
        "plan",
    )

    failed = service.fail_framework(admission, RuntimeError("post import failed"))
    snapshot = session.current()

    assert failed.runtime.mode is RuntimeMode.ORCHESTRATOR
    assert failed.runtime.allocation is NoAllocation
    assert failed.runtime.enforcement is RuntimeEnforcement.STRICT
    assert failed.runtime.requirement_axes.to_data() == ["environment", "world", "runtime"]
    assert snapshot.mode == "orchestrator"
    assert snapshot.allocation is None
    assert snapshot.inventory is None
    assert not any(key.startswith("session_") for key in failed.metadata)
    assert "framework_results" not in failed.metadata
