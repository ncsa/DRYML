import pytest

from dryml.core import Repo, Serializable
from dryml.core.repo import RepoLoadError
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
    reference = first.declare_object(ContendedValue().definition)
    ContendedValue.constructions = 0
    first.build_object_ref(reference)
    second = Repo(DirStore(store.base_dir), clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "second")

    with pytest.raises(RepoLoadError, match="active first-construction claim"):
        second.build_object_ref(reference)

    assert ContendedValue.constructions == 1


def test_expired_claim_is_taken_over_with_a_new_generation(tmp_path):
    now = [0.0]
    store = DirStore(tmp_path / "store")
    first = Repo(store, clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "first")
    reference = first.declare_object(ContendedValue().definition)
    first._acquire_claim(reference, store)
    now[0] = 11.0
    second = Repo(DirStore(store.base_dir), clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "second")

    lease = second._acquire_claim(reference, second.default_store)

    assert lease.generation == 2
    assert lease.owner == "second"


def test_expired_owner_cannot_abandon_its_claim_before_takeover(tmp_path):
    now = [0.0]
    store = DirStore(tmp_path / "store")
    repo = Repo(store, clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "first")
    reference = repo.declare_object(ContendedValue().definition)

    lease = repo._acquire_claim(reference, store)
    now[0] = 11.0

    assert not repo._abandon_claim(lease)
    claim = store.read_claim_record(reference.digest())
    assert claim.status == "claimed"
    assert claim.owner == "first"


def test_expired_live_graph_cannot_save_or_alias(tmp_path):
    now = [0.0]
    store = DirStore(tmp_path / "store")
    repo = Repo(store, clock=lambda: now[0], lease_duration=10, owner_token_factory=lambda: "first")
    reference = repo.declare_object(ContendedValue().definition)
    live = repo.build_object_ref(reference)
    now[0] = 11.0

    with pytest.raises(RepoLoadError, match="stale first-construction claim"):
        repo.save_object(live)
    with pytest.raises(RepoLoadError, match="stale first-construction claim"):
        repo.set_alias("stale", live)
    assert store.read_object_alias("stale") is None


def test_backward_clock_movement_delays_takeover_until_the_stored_deadline(tmp_path):
    now = [10.0]
    store = DirStore(tmp_path / "store")
    first = Repo(store, clock=lambda: now[0], lease_duration=5, owner_token_factory=lambda: "first")
    reference = first.declare_object(ContendedValue().definition)
    first._acquire_claim(reference, store)
    now[0] = 5.0
    second = Repo(
        DirStore(store.base_dir),
        clock=lambda: now[0],
        lease_duration=5,
        owner_token_factory=lambda: "second",
    )

    with pytest.raises(RepoLoadError, match="active first-construction claim"):
        second._acquire_claim(reference, second.default_store)
