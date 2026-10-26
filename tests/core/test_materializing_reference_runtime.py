import pytest
from pathlib import Path

from dryml.core import Definition, Object, ObjectId, Repo, Serializable
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from tests.core.test_repo_aliases import _saved_digest


class ReferenceLeaf(Object):
    def __init__(self, value):
        super().__init__()
        self.value = value


class ReferenceParent(Object):
    def __init__(self, target):
        super().__init__()
        self.target = target


class FailingReferenceParent(Object):
    def __init__(self, target):
        raise RuntimeError("parent failed")


class IdentifiedReferenceLeaf(Serializable):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value.txt").write_text(self.value, encoding="utf-8")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = Path(src_dir, "value.txt").read_text(encoding="utf-8")


class IdentifiedReferenceParent(Serializable):
    def __init__(self, target):
        self.target = target


def test_ref_exact_value_is_retained_without_materialization():
    repo = Repo()
    target = Definition(ReferenceLeaf, "exact").concretize(repo=repo)
    parent = ReferenceParent(DefLink.finalized(EdgeKind.REF, target), repo=repo)

    assert parent.graph_at('$[@param("target")]') is target


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_registered_materializing_object_ref_uses_its_claim(tmp_path):
    """A bound Node initializes once and closure publication supplies saved authority."""
    repo = Repo(DirStore(tmp_path / "store"))
    reference = Definition(IdentifiedReferenceLeaf, "exact").with_identity(ObjectId()).concretize()
    parent = Definition(ReferenceParent, reference).build(repo=repo)

    assert isinstance(parent.target, IdentifiedReferenceLeaf)
    assert parent.target.object_id == reference.object_id

    parent.target.value = "saved"
    state, report = repo.save_object(parent, deep_capture=True, report_stores=True)
    publication = next(item for item in report.publications if item.phase == "snapshot")
    assert publication.status == "completed"
    assert publication.state_ref.at('$[@param("target")]').object_id == reference.object_id
    assert len(publication.protected_state_keys) == 1
    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(
        state, reuse_live="never"
    )
    assert loaded.target.object_id == reference.object_id
    assert loaded.target.value == "saved"


@pytest.mark.usefixtures("fixed_snapshot_environment")
@pytest.mark.parametrize("after", [False, True])
def test_closure_descendant_claim_interruption_reports_exact_boundary(tmp_path, monkeypatch, after):
    """Interrupted closure batches report whether the complete saved root is visible."""
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    reference = Definition(IdentifiedReferenceLeaf, "exact").with_identity(ObjectId()).concretize()
    parent = Definition(ReferenceParent, reference).build(repo=repo)
    original = store.publish_save_batch

    def interrupt(batch):
        if after:
            original(batch)
        raise KeyboardInterrupt("nested claim interruption")

    monkeypatch.setattr(store, "publish_save_batch", interrupt)
    with pytest.raises(KeyboardInterrupt) as raised:
        repo.save_object(parent)
    report = raised.value.report
    publication = next(item for item in report.publications if item.phase == "snapshot")
    assert publication.state_ref.at('$[@param("target")]').object_id == reference.object_id
    assert publication.status == "failed"
    assert (store.read_node(_saved_digest(publication.state_ref), kind="saved") is not None) is after
    if after:
        loaded = Repo(store).load_state_ref(publication.state_ref, reuse_live="never")
        assert loaded.target.value == "exact"
    else:
        assert not repo.query().saved().exists()


def test_failed_parent_construction_abandons_materialized_reference_claim(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    reference = Definition(IdentifiedReferenceLeaf, "exact").with_identity(ObjectId()).concretize()

    with pytest.raises(RepoLoadError, match="parent failed"):
        Definition(FailingReferenceParent, reference).build(repo=repo)

    assert not repo.query().saved().exists()
    rebuilt = reference.build(repo=repo, cache="none")
    with repo.reserve_state_graph(rebuilt) as reservation:
        assert reservation.active


def test_failed_parent_construction_abandons_nested_reference_claims(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = Definition(IdentifiedReferenceLeaf, "child").with_identity(ObjectId()).concretize()
    parent = Definition(IdentifiedReferenceParent, child).with_identity(ObjectId()).concretize()

    with pytest.raises(RepoLoadError, match="parent failed"):
        Definition(FailingReferenceParent, parent).build(repo=repo)

    assert not repo.query().saved().exists()
    rebuilt = parent.build(repo=repo, cache="none")
    assert rebuilt.object_id == parent.object_id
    assert rebuilt.target.object_id == child.object_id
    with repo.reserve_state_graph(rebuilt) as reservation:
        assert len(reservation.object_ids) == 2
