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


def test_active_claim_rejects_second_builder_before_construction(tmp_path):
    now = [0.0]
    store = DirStore(tmp_path / "store")
    first = Repo(store, clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "first")
    live = ContendedValue(repo=first)
    ContendedValue.constructions = 0
    second = Repo(DirStore(store.base_dir), clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "second")

    with first.reserve_state_graph(live):
        with pytest.raises(RepoSaveError, match="reserved"):
            second.save_object(live)
    assert ContendedValue.constructions == 0


def test_expired_claim_is_taken_over_with_a_new_generation(tmp_path):
    """Elapsed wall time cannot steal native ownership; release permits a fresh owner."""
    now = [0.0]
    store = DirStore(tmp_path / "store")
    first = Repo(store, clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "first")
    live = ContendedValue(repo=first)
    second = Repo(DirStore(store.base_dir), clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "second")

    with first.reserve_state_graph(live) as old:
        now[0] = 11.0
        with pytest.raises(RepoSaveError, match="reserved"):
            second.reserve_state_graph(live)
    with second.reserve_state_graph(live) as fresh:
        assert fresh is not old and fresh.active and not old.active


def test_expired_owner_cannot_abandon_its_claim_before_takeover(tmp_path):
    """Foreign-thread unwind cannot release the current native owner."""
    now = [0.0]
    store = DirStore(tmp_path / "store")
    repo = Repo(store, clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "first")
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


def test_expired_live_graph_cannot_save_or_alias(tmp_path):
    """An inactive native reservation cannot authorize capture or name publication."""
    now = [0.0]
    store = DirStore(tmp_path / "store")
    repo = Repo(store, clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "first")
    live = ContendedValue(repo=repo)
    with repo.reserve_state_graph(live) as lease:
        pass
    now[0] = 11.0

    before = store.authority_cut()
    with pytest.raises(RepoSaveError, match="inactive"):
        repo.save_object(live, reservation=lease, alias="stale")
    assert store.read_node_name("stale") is None
    assert store.authority_cut() == before


def test_backward_clock_movement_delays_takeover_until_the_stored_deadline(tmp_path):
    now = [10.0]
    store = DirStore(tmp_path / "store")
    first = Repo(store, clock=lambda: now[0], lease_duration=5, owner_token_factory=lambda: "first")
    live = ContendedValue(repo=first)
    second = Repo(
        DirStore(store.base_dir),
        clock=lambda: now[0],
        lease_duration=5,
        owner_token_factory=lambda: "second",
    )

    with first.reserve_state_graph(live):
        now[0] = 5.0
        with pytest.raises(RepoSaveError, match="reserved"):
            second.reserve_state_graph(live)


def test_get_or_declare_does_not_replace_an_active_initialization_claim(tmp_path):
    """Idempotent metadata publication cannot consume a live ownership reservation."""
    store = DirStore(tmp_path / "store")
    first = Repo(store, owner_token_factory=lambda: "first")
    live = ContendedValue(repo=first)
    reference = first.describe(live)
    record = store.publish_node(reference)
    second = Repo(DirStore(store.base_dir), owner_token_factory=lambda: "second")

    with first.reserve_state_graph(live) as lease:
        before = store.authority_cut()
        assert second.default_store.publish_node(reference) == record
        assert store.authority_cut() == before and lease.active
