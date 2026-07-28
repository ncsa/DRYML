"""Facade rollback and fail-closed contracts."""

from __future__ import annotations

import pytest

from dryml import session
from dryml.runtime import RuntimeEnforcement, RuntimeState
from dryml.runtime.publication import PublicationService
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

    session.manage()
    assert environ["CUDA_VISIBLE_DEVICES"] == ""
    session.reset()

    assert environ["CUDA_VISIBLE_DEVICES"] == "7"
