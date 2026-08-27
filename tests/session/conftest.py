"""Deterministic isolated runtime publication fixture for session tests."""

import pytest

from dryml.runtime import PublicationService, RuntimeState
from dryml.session import state
from dryml.worlds import LocalResourceInventory


@pytest.fixture
def session_runtime(monkeypatch):
    """Provide a process-effect-free runtime publication and fixed inventory."""

    affinity = {"value": (0, 1)}
    service = PublicationService(
        environ={},
        affinity_getter=lambda: affinity["value"],
        affinity_setter=lambda value: affinity.__setitem__("value", value),
    )
    service.initialize(RuntimeState())
    inventory = LocalResourceInventory((0, 1), {"gpu": (0, 1)}, memory=8 * 1024**3, accelerator_memory={"gpu": {0: 4 * 1024**3, 1: 4 * 1024**3}})
    monkeypatch.setattr(state, "publication", service)
    monkeypatch.setattr(state, "local_inventory", lambda: inventory)
    return service
