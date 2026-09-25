from pathlib import Path
from abc import abstractmethod

import pytest

from dryml.core import Definition, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.store.records import ClaimRecord

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class ClaimedValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")


class AbstractClaimedValue(Serializable):
    @abstractmethod
    def required(self):
        """Return the required implementation result."""


def test_aggregate_abstract_admission_precedes_all_claims(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    concrete = repo.declare_object(Definition(ClaimedValue, 1).concretize(repo=repo))
    abstract = repo.declare_object(Definition(AbstractClaimedValue).concretize(repo=repo))
    before = {
        reference.digest(): store.read_claim_record(reference.digest())
        for reference in (concrete, abstract)
    }

    with pytest.raises(TypeError, match="required"):
        repo.materialize_boundary((concrete, abstract))

    after = {
        reference.digest(): store.read_claim_record(reference.digest())
        for reference in (concrete, abstract)
    }
    assert after == before


def test_object_ref_abstract_admission_precedes_its_claim(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = repo.declare_object(
        Definition(AbstractClaimedValue).concretize(repo=repo)
    )
    before = store.read_claim_record(reference.digest())

    with pytest.raises(TypeError, match="required"):
        repo.build_object_ref(reference)

    assert store.read_claim_record(reference.digest()) == before


def test_claim_can_be_renewed_abandoned_and_completed_with_one_fence(tmp_path):
    now = [10.0]
    store = DirStore(tmp_path / "store")
    repo = Repo(store, clock=lambda: now[0], lease_duration=5, owner_token_factory=lambda: "owner")
    reference = repo.declare_object(ClaimedValue(1).definition)

    lease = repo._acquire_claim(reference, store)
    assert store.read_claim_record(reference.digest()).lease_until == 15
    now[0] = 12
    repo._renew_claim(lease)
    assert store.read_claim_record(reference.digest()).lease_until == 17
    assert repo._abandon_claim(lease)

    obj = repo.build_object_ref(reference)
    state = repo.save_object(obj)
    claim = store.read_claim_record(reference.digest())
    assert claim.status == "completed"
    assert claim.state_ref_digest == state.digest()


def test_matching_state_ref_repairs_claim_but_other_state_ref_cannot_complete(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store, clock=lambda: 10.0, owner_token_factory=lambda: "owner")
    reference = repo.declare_object(ClaimedValue(1).definition)
    other = repo.save_object(ClaimedValue(2, repo=repo))

    lease = repo._acquire_claim(reference, store)
    assert lease is not None
    assert other.object != reference
    assert store.read_claim_record(reference.digest()).status == "claimed"
    assert repo._abandon_claim(lease)

    state = repo.save_object(repo.build_object_ref(reference))
    store.write_claim_record(ClaimRecord(reference.digest(), 99, "claimed", "recovered", 20.0))

    assert repo._acquire_claim(reference, store) is None
    repaired = store.read_claim_record(reference.digest())
    assert repaired.status == "completed"
    assert repaired.state_ref_digest == state.digest()
