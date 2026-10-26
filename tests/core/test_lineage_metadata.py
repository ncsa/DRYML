from datetime import datetime, timezone

import pytest

from dryml.core import Definition, Object, ObjectId, Repo, Serializable
from dryml.core.repo_plan import apply_node_identities, build_save_plan
from dryml.core.snapshot_capture import capture_lineages
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Index, Parameter


class LineageLeaf(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class LineageRoot(Object):
    def __init__(self, child):
        self.child = child


class BrokenLineageLeaf(Serializable):
    def __init__(self):
        raise RuntimeError("construction failed")


def test_runtime_allocation_records_lineages_at_primary_object_id_paths(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = LineageLeaf("child", repo=repo)
    root = LineageRoot([child, child], repo=repo)

    lineages = capture_lineages(build_save_plan(repo, root))
    child_path = GraphPath((Parameter("child"), Index(0)))

    assert root.object_id is None
    assert set(lineages) == {child_path}
    assert lineages[child_path].object_ref.object_id == child.object_id
    assert lineages[child_path].object_ref.creation_view().graph_equal(child.definition)
    assert lineages[child_path].created_at == repo._lineage_candidates[child.object_id]


def test_exact_identity_replaces_provisional_lineage_facts_before_capture(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    source = LineageLeaf("value", repo=repo)
    expected = repo.describe(source)
    restored = LineageLeaf("value", repo=repo)
    known = datetime(2024, 1, 2, tzinfo=timezone.utc)

    apply_node_identities(
        restored,
        expected,
        lineage_facts={GraphPath(): known},
    )

    lineages = capture_lineages(build_save_plan(repo, restored))
    assert restored.object_id == expected.object_id
    assert lineages[GraphPath()].created_at == known

    apply_node_identities(restored, expected)
    assert capture_lineages(build_save_plan(repo, restored))[GraphPath()].creation_status == "unknown"


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_declaration_and_forks_create_in_memory_candidates_without_store_hooks(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    source = repo.save_object(LineageLeaf("value", repo=repo))
    declared = Definition(LineageLeaf, "declared").with_identity(ObjectId())
    object_fork = source.without_state().replace_bindings(identities={"$": ObjectId()})
    state_fork = source.replace_bindings(identities={"$": ObjectId()})
    candidates = dict(repo._lineage_candidates)
    assert declared.object_id not in candidates
    assert object_fork.object_id not in candidates
    assert state_fork.object_id not in candidates
    assert state_fork.state == source.state
    assert object_fork.object_id != source.object_id
    assert state_fork.object_id != source.object_id


def test_declared_identity_propagates_its_candidate_into_live_realization(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    declared = Definition(LineageLeaf, "declared").with_identity(ObjectId())
    built = repo.build(declared.concretize())

    assert repo.get_cached(built.definition) is built
    assert built.object_id == declared.object_id
    assert capture_lineages(build_save_plan(repo, built))[GraphPath()].creation_status == "unknown"
    assert declared.object_id not in repo._lineage_candidates


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_declaration_lineage_fact_survives_reopening_before_construction(tmp_path, monkeypatch):
    created_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    monkeypatch.setattr("dryml.core.snapshot_capture.current_utc_time", lambda: created_at)
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    built = LineageLeaf("declared", repo=repo)
    declared = repo.save_object(built)
    reopened = Repo(DirStore(store.base_dir))

    assert reopened.get_lineage_metadata(declared).created_at == created_at


def test_declaration_lineage_failure_withholds_discoverable_authority(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    monkeypatch.setattr(
        store, "publish_save_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("lineage publication failed")),
    )

    from dryml.core.repo import RepoSaveError
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(LineageLeaf("declared", repo=repo))
    assert "lineage publication" in str(raised.value.__cause__)

    assert store.query().saved().count() == 0
    assert store.query().cdefs().count() == 0


def test_failed_initialization_does_not_create_a_lineage_candidate(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))

    with pytest.raises(RuntimeError, match="construction failed"):
        BrokenLineageLeaf(repo=repo)

    assert repo._lineage_candidates == {}
