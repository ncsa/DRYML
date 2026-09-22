import pytest
from pathlib import Path

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore


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


def test_unregistered_materializing_object_ref_is_rejected():
    repo = Repo()
    original = IdentifiedReferenceLeaf("exact", repo=repo)

    with pytest.raises(RepoLoadError, match="declaration Store"):
        ReferenceParent(original.object_ref, repo=repo)


def test_registered_materializing_object_ref_uses_its_claim(tmp_path):
    """Closure placement completes and reports a materialized descendant claim."""
    repo = Repo(DirStore(tmp_path / "store"))
    reference = repo.declare_object(
        Definition(IdentifiedReferenceLeaf, "exact").concretize(repo=repo)
    )
    parent = ReferenceParent(reference, repo=repo)

    assert isinstance(parent.target, IdentifiedReferenceLeaf)
    assert parent.target.object_id == reference.object_id

    parent.target.value = "saved"
    state, report = repo.save_object(parent, deep_capture=True, report_stores=True)
    claim = repo.default_store.read_claim_record(reference.digest())

    assert claim.status == "completed"
    publication = next(item for item in report.publications if item.phase == "claim")
    assert publication.status == "completed"
    assert publication.state_ref.object == reference
    assert claim.state_ref_digest == publication.state_ref.digest()
    assert parent.target._claim_lease is None
    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(
        state, reuse_live="never"
    )
    assert loaded.target.object_id == reference.object_id
    assert loaded.target.value == "saved"


@pytest.mark.parametrize("after", [False, True])
def test_closure_descendant_claim_interruption_reports_exact_boundary(tmp_path, monkeypatch, after):
    """Interrupted nested claim completion preserves published closure authority."""
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    reference = repo.declare_object(Definition(IdentifiedReferenceLeaf, "exact").concretize(repo=repo))
    parent = ReferenceParent(reference, repo=repo)
    original = repo._mark_initial_state_ref_complete

    def interrupt(state, destination, lease):
        if after:
            original(state, destination, lease)
        raise KeyboardInterrupt("nested claim interruption")

    monkeypatch.setattr(repo, "_mark_initial_state_ref_complete", interrupt)
    with pytest.raises(KeyboardInterrupt) as raised:
        repo.save_object(parent)
    report = raised.value.report
    publication = next(item for item in report.publications if item.phase == "claim")
    assert publication.state_ref.object == reference
    assert publication.status == ("completed" if after else "failed")
    assert store.read_claim_record(reference.digest()).status == ("completed" if after else "available")
    snapshot = next(item for item in report.publications if item.phase == "snapshot")
    assert snapshot.status == "completed"
    loaded = Repo(store).load_state_ref(snapshot.state_ref, reuse_live="never")
    assert loaded.target.value == "exact"


def test_failed_parent_construction_abandons_materialized_reference_claim(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    reference = repo.declare_object(
        Definition(IdentifiedReferenceLeaf, "exact").concretize(repo=repo)
    )

    with pytest.raises(RuntimeError, match="parent failed"):
        FailingReferenceParent(reference, repo=repo)

    assert repo.default_store.read_claim_record(reference.digest()).status == "available"


def test_failed_parent_construction_abandons_nested_reference_claims(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = repo.declare_object(
        Definition(IdentifiedReferenceLeaf, "child").concretize(repo=repo)
    )
    parent = repo.declare_object(
        Definition(IdentifiedReferenceParent, child).concretize(repo=repo)
    )

    with pytest.raises(RuntimeError, match="parent failed"):
        FailingReferenceParent(parent, repo=repo)

    assert repo.default_store.read_claim_record(child.digest()).status == "available"
    assert repo.default_store.read_claim_record(parent.digest()).status == "available"
