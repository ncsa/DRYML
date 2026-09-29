"""Contracts for dependency-light watched-framework registration."""

from __future__ import annotations

import sys
import uuid

import pytest

from dryml._framework_imports import ImportEpochCoordinator, PassiveFrameworkFinder
from dryml.runtime.frameworks import FrameworkRegistration, FrameworkRegistry


def _root() -> str:
    """Return one unique valid synthetic module root."""
    return "dryml_fake_" + uuid.uuid4().hex


def test_registry_rejects_callable_duplicate_overlap_observed_and_loaded(monkeypatch):
    """Registration closes all mutable root-contract races before import."""
    finder = PassiveFrameworkFinder(ImportEpochCoordinator())
    registry = FrameworkRegistry(finder=finder)
    root = _root()
    registry.register(FrameworkRegistration(root, (root,), object()))
    with pytest.raises(ValueError, match="callable"):
        registry.register(FrameworkRegistration(_root(), (_root(),), lambda: None))
    with pytest.raises(ValueError, match="registered"):
        registry.register(FrameworkRegistration(root, (root,), object()))
    with pytest.raises(ValueError, match="overlap"):
        registry.register(FrameworkRegistration(_root(), (root + ".child",), object()))
    observed = _root()
    finder.find_spec(observed)
    with pytest.raises(RuntimeError, match="observed"):
        registry.register(FrameworkRegistration(observed, (observed,), object()))
    loaded = _root()
    monkeypatch.setitem(sys.modules, loaded, object())
    with pytest.raises(RuntimeError, match="loaded"):
        registry.register(FrameworkRegistration(loaded, (loaded,), object()))


def test_registry_freezes_when_observation_ledger_reaches_its_bound():
    """The fixed first-observation ledger cannot grow without bound."""
    finder = PassiveFrameworkFinder(ImportEpochCoordinator())
    for index in range(finder.observation_limit):
        finder.find_spec(f"dryml_observed_{index}")
    assert finder.registration_frozen
    with pytest.raises(RuntimeError, match="frozen"):
        finder.can_register((_root(),))


def test_observation_callback_replays_a_root_seen_before_registry_setup():
    """Late registry setup still freezes for a previously watched root."""
    finder = PassiveFrameworkFinder(ImportEpochCoordinator())
    root = _root()
    finder.install_builtin_roots((root,))
    finder.find_spec(root)
    observed: list[str] = []

    finder.set_observation_callback(observed.append)

    assert observed == [root]
