"""Focused contracts for inert Repo save-routing configuration."""

import sys
import threading
import subprocess

import pytest

from dryml.core import Object, Repo, SaveRouting, Selector
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore


class Routed(Object):
    """Object type used to exercise Selector-based routing decisions."""


class Unrouted(Object):
    """Object type that does not match the routing rules in these tests."""


def test_save_routing_normalizes_rules_and_placement_shorthands(tmp_path):
    """Placement shorthands retain the installed ordered all-match policy."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    routing = SaveRouting(
        routes=[(Selector(Routed), second)],
        match_mode="all",
        graph_mode="per-object",
    )
    repo = Repo([first, second], save_routing=routing)

    assert routing.routes == ((Selector(Routed), second),)
    assert repo.save_routing == routing
    repo.set_save_routing("closure")
    assert repo.save_routing == SaveRouting(routing.routes, "all", "closure")
    repo.set_save_routing("per-object")
    assert repo.save_routing == routing

    fresh = Repo(first, save_routing="per-object")
    assert fresh.save_routing == SaveRouting()


def test_internal_selection_uses_first_all_fallback_and_handle_deduplication(tmp_path):
    """Selection retains rule order and never treats one handle as two replicas."""

    fallback = DirStore(tmp_path / "fallback")
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(
        [fallback, first, second],
        save_routing=SaveRouting(
            ((Selector(Routed), first), (Selector(Routed), second)),
        ),
    )

    with repo._retain_save_context() as context:
        assert repo._select_save_destinations(context, Routed()) == (first,)
        assert repo._select_save_destinations(context, Unrouted()) == (fallback,)

    repo.set_save_routing(
        SaveRouting(
            ((Selector(Routed), first), (Selector(Routed), first), (Selector(Routed), second)),
            "all",
        )
    )
    with repo._retain_save_context() as context:
        assert repo._select_save_destinations(context, Routed()) == (first, second)


def test_readding_existing_store_without_default_keeps_connected_store_order(tmp_path):
    """Re-registering a handle is idempotent unless explicitly made default."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo([first, second])

    repo.add_store(first)

    assert repo.stores == [first, second]
    assert repo.default_store is first


def test_invalid_routing_and_physical_handles_fail_without_mutating_configuration(tmp_path):
    """Invalid routes and ambiguous built-in handle aliases leave Repo state intact."""

    connected = DirStore(tmp_path / "connected")
    disconnected = DirStore(tmp_path / "disconnected")
    repo = Repo(connected, save_routing="per-object")
    original = repo.save_routing
    stores = tuple(repo.stores)

    with pytest.raises(ValueError, match="connected"):
        repo.set_save_routing(SaveRouting(((Selector(Routed), disconnected),)))
    with pytest.raises(ValueError, match="graph mode"):
        repo.set_save_routing("unknown")
    with pytest.raises(ValueError, match="match_mode"):
        SaveRouting(match_mode="replicate")

    assert repo.save_routing == original
    assert tuple(repo.stores) == stores

    duplicate = DirStore(connected.base_dir)
    unconfigured = Repo(connected)
    unconfigured.add_store(duplicate)
    with pytest.raises(ValueError, match="physical"):
        unconfigured.set_save_routing(SaveRouting())
    assert unconfigured.save_routing is None
    assert tuple(unconfigured.stores) == (connected, duplicate)


def test_retained_context_isolated_from_configuration_and_source_changes(tmp_path, monkeypatch):
    """A retained context keeps its policy, source order, and default Store."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(first, save_routing="per-object")
    monkeypatch.setattr(first, "validate_local_state", lambda *_: "first-source")
    monkeypatch.setattr(second, "validate_local_state", lambda *_: "second-source")

    ready = threading.Event()
    release = threading.Event()
    observed = []

    def retain_and_observe():
        with repo._retain_save_context() as context:
            ready.set()
            assert release.wait(timeout=5)
            observed.extend((
                context.default_store,
                context.find_local_state(Routed().definition, "pkl-" + "a" * 64),
                repo._select_save_destinations(context, Routed()),
            ))

    worker = threading.Thread(target=retain_and_observe)
    worker.start()
    assert ready.wait(timeout=5)
    repo.add_store(second, make_default=True)
    repo.set_save_routing(SaveRouting(((Selector(Routed), second),)))
    release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert observed == [first, first, (first,)]


def test_save_context_lease_blocks_repo_close(tmp_path):
    """A context lease protects its Repo resources until the holder exits."""

    repo = Repo(DirStore(tmp_path / "store"))
    retained = repo._retain_save_context()
    retained.__enter__()
    try:
        with pytest.raises(RuntimeError, match="active save context"):
            repo.close(flush=False)
    finally:
        retained.__exit__(None, None, None)
    repo.close(flush=False)


def test_core_import_avoids_managed_policy_in_a_fresh_process():
    """Core imports do not load managed policy in an isolated interpreter."""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dryml.core; assert 'dryml.managed' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_empty_routing_does_not_create_a_store_during_selection():
    """Configured fallback without a default Store fails without implicit storage."""

    repo = Repo(save_routing=SaveRouting())

    assert repo.stores == []
    with repo._retain_save_context() as context:
        with pytest.raises(RepoSaveError, match="No Store"):
            repo._select_save_destinations(context, Unrouted())
    assert repo.stores == []
