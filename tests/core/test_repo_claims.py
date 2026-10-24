from pathlib import Path
from abc import abstractmethod

import pytest

from dryml.core import Definition, ObjectId, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.repo import RepoSaveError

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
    concrete = Definition(ClaimedValue, 1).with_identity(ObjectId()).concretize()
    abstract = Definition(AbstractClaimedValue).with_identity(ObjectId()).concretize()
    before = store.authority_cut()
    constructions = repo._num_constructions

    with pytest.raises(TypeError, match="required"):
        repo.materialize_boundary((concrete, abstract))

    assert store.authority_cut() == before
    assert repo._num_constructions == constructions


def test_object_ref_abstract_admission_precedes_its_claim(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = Definition(AbstractClaimedValue).with_identity(ObjectId()).concretize()
    before = store.authority_cut()

    with pytest.raises(TypeError, match="required"):
        repo.build(reference)

    assert store.authority_cut() == before
    assert repo._num_constructions == 0


def test_claim_can_be_renewed_abandoned_and_completed_with_one_fence(tmp_path):
    """Native live ownership replaces renewable wall-clock first-build claims."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = ClaimedValue(1, repo=repo)
    with repo.reserve_state_graph(obj) as lease:
        assert lease.active
        with pytest.raises(RepoSaveError, match="reserved"):
            repo.reserve_state_graph(obj)
        state = repo.save_object(obj, reservation=lease)
    assert not lease.active
    assert repo.query().saved().one().graph_equal(state)
    with repo.reserve_state_graph(obj) as fresh:
        assert fresh.active and fresh is not lease


def test_matching_state_ref_repairs_claim_but_other_state_ref_cannot_complete(tmp_path):
    """A reservation authorizes only its exact live graph, never another receipt."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj, other = ClaimedValue(1, repo=repo), ClaimedValue(2, repo=repo)
    with repo.reserve_state_graph(obj) as lease:
        with pytest.raises(RepoSaveError, match="does not cover"):
            repo.save_object(other, reservation=lease)
        assert other.last_state_ref is None
        state = repo.save_object(obj, reservation=lease)
    assert obj.last_state_ref is state
    assert repo.query().saved().one().graph_equal(state)
