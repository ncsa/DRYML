import pytest

from dryml.core import Repo, Serializable
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore


class ContendedValue(Serializable):
    constructions = 0

    def __init__(self):
        type(self).constructions += 1

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def test_active_reservation_rejects_second_save_before_construction(tmp_path):
    store = DirStore(tmp_path / "store")
    first = Repo(store)
    live = ContendedValue(repo=first)
    ContendedValue.constructions = 0
    second = Repo(DirStore(store.base_dir))

    with first.reserve_state_graph(live):
        with pytest.raises(RepoSaveError, match="reserved"):
            second.save_object(live)
    assert ContendedValue.constructions == 0


def test_reservation_release_permits_a_fresh_owner(tmp_path):
    """Native ownership lasts until release, after which a new handle may acquire it."""
    store = DirStore(tmp_path / "store")
    first = Repo(store)
    live = ContendedValue(repo=first)
    second = Repo(DirStore(store.base_dir))

    with first.reserve_state_graph(live) as old:
        with pytest.raises(RepoSaveError, match="reserved"):
            second.reserve_state_graph(live)
    with second.reserve_state_graph(live) as fresh:
        assert fresh is not old and fresh.active and not old.active


def test_foreign_thread_cannot_release_reservation(tmp_path):
    """Foreign-thread unwind cannot release the current native owner."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    live = ContendedValue(repo=repo)
    from threading import Thread
    releases = []
    with repo.reserve_state_graph(live) as lease:
        def foreign():
            releases.append(lease.release())
        thread = Thread(target=foreign)
        thread.start()
        thread.join(5)
        assert not thread.is_alive() and releases == [False]
        assert lease.active


def test_inactive_reservation_cannot_save_or_alias(tmp_path):
    """An inactive native reservation cannot authorize capture or name publication."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    live = ContendedValue(repo=repo)
    with repo.reserve_state_graph(live) as lease:
        pass

    before = store.authority_cut()
    with pytest.raises(RepoSaveError, match="inactive"):
        repo.save_object(live, reservation=lease, alias="stale")
    assert store.read_node_name("stale") is None
    assert store.authority_cut() == before


def test_idempotent_publication_preserves_active_reservation(tmp_path):
    """Idempotent metadata publication cannot consume a live ownership reservation."""
    store = DirStore(tmp_path / "store")
    first = Repo(store)
    live = ContendedValue(repo=first)
    reference = first.describe(live)
    record = store.publish_node(reference)
    second = Repo(DirStore(store.base_dir))

    with first.reserve_state_graph(live) as lease:
        before = store.authority_cut()
        assert second.default_store.publish_node(reference) == record
        assert store.authority_cut() == before and lease.active
