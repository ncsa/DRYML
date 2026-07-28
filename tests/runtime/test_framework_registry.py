"""Deterministic contracts for the lightweight framework registry."""

from __future__ import annotations

import sys
import subprocess
import threading
import uuid
import importlib.util

import pytest

from dryml._framework_imports import ImportEpochCoordinator, PassiveFrameworkFinder, coordinator, finder
from dryml.runtime.frameworks import FrameworkRegistration, FrameworkRegistry, framework_registry


@pytest.fixture(autouse=True)
def _isolate_registry_freeze():
    """Synthetic registrations must not inherit another test's process epoch."""

    with framework_registry._lock:
        framework_registry._frozen = False
    try:
        yield
    finally:
        with framework_registry._lock:
            framework_registry._frozen = False


def _root() -> str:
    return "dryml_fake_" + uuid.uuid4().hex


def test_builtin_groups_are_registered_without_importing_frameworks():
    groups = framework_registry.registrations()

    assert {"tensorflow", "torch", "jax"}.issubset(groups)
    assert groups["jax"].roots == ("jax", "jaxlib")
    completed = subprocess.run(
        [sys.executable, "-c", "import sys, dryml.runtime; raise SystemExit(bool({'tensorflow', 'torch', 'jax', 'jaxlib'} & set(sys.modules)))"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_base_import_installs_builtin_metadata_before_the_finder_without_runtime():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, dryml; "
            "from dryml._framework_imports import finder; "
            "assert {'tensorflow', 'torch', 'jax', 'jaxlib'} <= finder.roots(); "
            "assert 'dryml.runtime' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_base_import_rejects_a_preloaded_builtin_root():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, types; sys.modules['torch'] = types.ModuleType('torch'); import dryml",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "already loaded" in completed.stderr


def test_registration_rejects_overlap_loaded_and_observed_roots(monkeypatch):
    root = _root()
    registration = FrameworkRegistration(root, (root,), object())
    framework_registry.register(registration)

    with pytest.raises(ValueError, match="already registered"):
        framework_registry.register(registration)
    with pytest.raises(ValueError, match="overlap"):
        framework_registry.register(FrameworkRegistration(root + "other", (root,), object()))

    observed = _root()
    finder.watch(observed)
    finder.find_spec(observed)
    with pytest.raises(RuntimeError, match="observed"):
        framework_registry.register(FrameworkRegistration(observed, (observed,), object()))

    loaded = _root()
    monkeypatch.setitem(sys.modules, loaded, object())
    with pytest.raises(RuntimeError, match="loaded"):
        framework_registry.register(FrameworkRegistration(loaded, (loaded,), object()))


def test_unregistered_first_observations_are_bounded_and_freeze_registration():
    """The finder closes the unregistered-import registration race at N/N+1."""

    local_finder = PassiveFrameworkFinder(ImportEpochCoordinator())
    for index in range(4095):
        local_finder.find_spec(f"dryml_observation_{index}_{uuid.uuid4().hex}")

    assert local_finder.observation_count == 4095
    assert not local_finder.registration_frozen
    local_finder.find_spec(f"dryml_observation_4095_{uuid.uuid4().hex}")
    assert local_finder.observation_count == 4096
    assert local_finder.registration_frozen
    with pytest.raises(RuntimeError, match="frozen"):
        local_finder.can_register((_root(),))


def test_unregistered_observation_blocks_later_registration():
    root = _root()
    local_finder = PassiveFrameworkFinder(ImportEpochCoordinator())

    local_finder.find_spec(root)

    with pytest.raises(RuntimeError, match="observed"):
        local_finder.can_register((root,))


def test_exact_observation_ledger_retains_descendants_and_blocks_ancestor_registration():
    root = _root()
    local_finder = PassiveFrameworkFinder(ImportEpochCoordinator())

    local_finder.find_spec(root + ".first")
    local_finder.find_spec(root + ".second")

    assert local_finder.observation_count == 1
    assert local_finder.first_observation(root) == root + ".first"
    assert local_finder.observed(root)
    with pytest.raises(RuntimeError, match="observed"):
        local_finder.can_register((root,))


def test_paused_finder_to_module_cache_gap_linearizes_before_registration(monkeypatch):
    """An observation wins even while its delegated finder remains paused."""

    import dryml.runtime.frameworks as frameworks

    root = _root()
    coordinator = ImportEpochCoordinator()
    local_finder = PassiveFrameworkFinder(coordinator)
    registry = FrameworkRegistry()
    delegated = threading.Event()
    release = threading.Event()

    class BlockingDelegate:
        def find_spec(self, fullname, path=None, target=None):
            delegated.set()
            assert release.wait(timeout=2)
            return None

    monkeypatch.setattr(frameworks, "coordinator", coordinator)
    monkeypatch.setattr(frameworks, "finder", local_finder)
    monkeypatch.setattr(sys, "meta_path", [local_finder, BlockingDelegate()])
    # Use the import machinery so the passive finder returns ``None`` and the
    # next finder owns the paused finder-to-module-cache interval.
    thread = threading.Thread(target=importlib.util.find_spec, args=(root,))
    thread.start()
    assert delegated.wait(timeout=2)

    with pytest.raises(RuntimeError, match="observed"):
        registry.register(FrameworkRegistration(root, (root,), object()))

    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_registration_wins_before_a_later_observation(monkeypatch):
    """A finder waits for a registered writer and observes the new root."""

    import dryml.runtime.frameworks as frameworks

    root = _root()
    coordinator = ImportEpochCoordinator()
    local_finder = PassiveFrameworkFinder(coordinator)
    registry = FrameworkRegistry()
    monkeypatch.setattr(frameworks, "coordinator", coordinator)
    monkeypatch.setattr(frameworks, "finder", local_finder)

    registry.register(FrameworkRegistration(root, (root,), object()))
    local_finder.find_spec(root + ".child")

    assert registry.registration_for(root + ".child").name == root
    assert local_finder.observed(root)


def test_registration_rejects_ancestor_factories_and_freeze():
    parent = _root()
    registry = FrameworkRegistry()
    registry.register(FrameworkRegistration(parent, (parent,), object()))

    with pytest.raises(ValueError, match="overlap"):
        registry.register(FrameworkRegistration(parent + "_child", (parent + ".child",), object()))
    with pytest.raises(ValueError, match="factories"):
        registry.register(FrameworkRegistration(_root(), (_root(),), lambda: object()))

    registry.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(FrameworkRegistration(_root(), (_root(),), object()))


def test_registry_freeze_is_busy_while_a_loader_reader_is_active():
    registry = FrameworkRegistry()

    with coordinator.reader():
        with pytest.raises(RuntimeError, match="import-busy"):
            registry.freeze()
