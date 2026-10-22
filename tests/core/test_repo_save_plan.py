from dryml.core import Object, Repo, Serializable
from dryml.core.repo_plan import build_save_plan
from dryml.core.store.dir import DirStore
from pathlib import Path
from contextlib import closing
import pytest
from dryml.core.utils.graph.path import GraphPath, Parameter, Index

def child_path(index):
    return GraphPath((Parameter("child"), Index(index)))

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class SavePlanLeaf(Serializable):
    captures = 0
    def __init__(self, name):
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        from dryml.core.store.retention import _writers
        assert not _writers(), "capture ran under a Store writer fence"
        type(self).captures += 1
        Path(dest_dir, "value").write_text(str(self.name))


class SavePlanNode(Object):
    def __init__(self, child):
        self.child = child


def test_save_plan_uses_retained_primary_bindings_once(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = SavePlanLeaf("child", repo=repo)
    parent = SavePlanNode([child, child], repo=repo)

    plan = build_save_plan(repo, parent)

    assert len(plan.actions) == 1
    assert plan.actions[0].obj is child
    assert plan.actions[0].path in parent.object_ref.objects


def test_save_plan_keeps_independent_equal_nodes_distinct(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = SavePlanLeaf("same", repo=repo)
    second = SavePlanLeaf("same", repo=repo)
    parent = SavePlanNode([first, second], repo=repo)

    plan = build_save_plan(repo, parent)

    assert {action.obj for action in plan.actions} == {first, second}


def test_partial_reuse_captures_only_remaining_distinct_nodes(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    first = SavePlanLeaf("same", repo=repo)
    donor = repo.save(first)
    second = SavePlanLeaf("same", repo=repo)
    root = SavePlanNode([first, first, second], repo=repo)
    SavePlanLeaf.captures = 0
    saved = repo.save(root, states={child_path(0): donor.state})
    assert SavePlanLeaf.captures == 1
    assert saved.at(child_path(0)).object_id != saved.at(child_path(2)).object_id
    assert saved.at(child_path(0)).state.key == saved.at(child_path(2)).state.key
    assert saved.at(child_path(0)) is saved.at(child_path(1))
    assert len(saved.state_spec()) == 2


def test_plain_save_recaptures_but_reuses_installed_private_data(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    obj = SavePlanLeaf("a", repo=repo)
    SavePlanLeaf.captures = 0
    first = repo.save(obj)
    data = Path(store.state_data_path(first.state.key))
    inode = (data / "value").stat().st_ino
    second = repo.save(obj)
    assert SavePlanLeaf.captures == 2
    assert first.state.key == second.state.key
    assert (data / "value").stat().st_ino == inode
    obj.name = "b"
    third = repo.save(obj)
    assert SavePlanLeaf.captures == 3
    assert third.state.key != first.state.key
    assert third.creation_digest() == first.creation_digest()
    assert first.construction_view.state is None
    assert first.construction_view.object_id is None


def test_reuse_validation_and_unsupported_backend_precede_capture(tmp_path):
    from dryml.core.store.zip import ZipStore
    from dryml.core.store.store import StoreCapabilityError
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    donor = repo.save(SavePlanLeaf("a", repo=repo))
    obj = SavePlanLeaf("different", repo=repo)
    SavePlanLeaf.captures = 0
    with pytest.raises(ValueError, match="incompatible"):
        repo.save(obj, states={"$": donor.state})
    with closing(ZipStore(tmp_path / "store.zip")) as archive:
        with pytest.raises(StoreCapabilityError):
            repo.save(obj, store=archive)
    assert SavePlanLeaf.captures == 0


def test_user_save_retains_full_keys_before_return(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    saved, report = repo.save(SavePlanLeaf("a", repo=repo), report_stores=True)
    assert store.delete_state(saved.state.key).status == "retained"
    assert report.publications
    assert all(item.status == "completed" for item in report.publications)
    publication = next(p for p in report.publications if p.phase == "snapshot")
    assert publication.store_uuid == store.store_uuid
    assert publication.retention_token


def test_indexed_multinode_save_has_one_sql_commit_and_shared_export(tmp_path):
    from dryml.core.repo_plan import build_routed_save_plan, execute_routed_save_plan
    from dryml.core.node_prepare import NodePreparation
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    index = store.open_query_index()
    index.maintain_store()
    statements = []
    index._connections.connection().set_trace_callback(statements.append)
    children = [SavePlanLeaf(str(i), repo=repo) for i in range(4)]
    root = SavePlanNode(children, repo=repo)
    plan = build_save_plan(repo, root)
    with repo._retain_save_context() as context:
        saved = execute_routed_save_plan(repo, build_routed_save_plan(repo, plan, context), main=True, alias="latest")
    assert sum(sql.strip().upper() == "COMMIT" for sql in statements) == 1
    assert len(index.projection_rows("saved_root", refresh=False)) == 1
    assert len(index.projection_rows("manifest", refresh=False)) == 4
    assert len(index.projection_rows("name", refresh=False)) == 2
    assert not store.authority_cut().pending
    assert plan.preparation.counts["export_rows"] == 5
    assert plan.preparation.counts["reused_rows"] > 0
    assert plan.capture_counts["discovery"] == 4
    assert plan.preparation.counts["export_bytes"] > 0
    assert len(tuple(Path(store.base_dir, "definitions", "blocks").rglob("*.record"))) == 1
    assert len(saved.state_spec()) == 4


def test_identical_plain_save_captures_but_is_an_authoritative_noop(tmp_path, monkeypatch):
    import shutil
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    obj = SavePlanLeaf("a", repo=repo)
    SavePlanLeaf.captures = 0
    saved = repo.save_object(obj)
    before = store.authority_cut()
    index = store.open_query_index()
    statements = []
    index._connections.connection().set_trace_callback(statements.append)
    monkeypatch.setattr(shutil, "copytree", lambda *a, **kw: pytest.fail("installed state copied"))
    again = repo.save_object(obj)
    assert SavePlanLeaf.captures == 2
    assert again.graph_equal(saved)
    assert store.authority_cut() == before
    assert not any(sql.strip().upper() == "COMMIT" for sql in statements)
    reused = repo.save_object(obj, states=saved.state_spec())
    assert reused.graph_equal(saved)
    assert SavePlanLeaf.captures == 2


def test_routed_index_failure_preserves_independent_targets_and_repairs_without_capture(tmp_path, monkeypatch):
    from dryml.core import SaveRouting, Selector
    from dryml.core.repo import RepoSaveError
    stores = [DirStore(tmp_path / name, query_index="sqlite") for name in ("a", "b", "c")]
    repo = Repo(stores, save_routing=SaveRouting(tuple((Selector(SavePlanLeaf), s) for s in stores), "all"))
    indexes = [s.open_query_index() for s in stores]
    for index in indexes:
        index.maintain_store()
    original = indexes[1].maintain_store
    def fail_after_authority(*args, **kwargs):
        if stores[1].authority_cut().pending:
            raise OSError("projection unavailable")
        return original(*args, **kwargs)
    monkeypatch.setattr(indexes[1], "maintain_store", fail_after_authority)
    obj = SavePlanLeaf("a", repo=repo)
    SavePlanLeaf.captures = 0
    with pytest.raises(RepoSaveError) as caught:
        repo.save_object(obj, report_stores=True)
    report = caught.value.report
    assert SavePlanLeaf.captures == 1
    assert [p.status for p in report.publications] == ["completed", "uncertain", "completed"]
    assert len(report.snapshots) == 3
    assert all(p.retention_token and p.store_uuid and p.mutation_id for p in report.publications)
    assert all(s.delete_state(obj.last_state_ref.state.key).status == "retained" for s in (stores[0], stores[2]))
    monkeypatch.setattr(indexes[1], "maintain_store", original)
    original()
    assert not stores[1].authority_cut().pending
    assert SavePlanLeaf.captures == 1


@pytest.mark.parametrize("index_policy", ["none", "sqlite"])
def test_partial_authority_repair_is_owner_fenced_and_does_not_capture(tmp_path, monkeypatch, index_policy):
    from dryml.core.repo import RepoSaveError
    store = DirStore(tmp_path / "store", query_index=index_policy)
    repo = Repo(store)
    obj = SavePlanLeaf("a", repo=repo)
    SavePlanLeaf.captures = 0
    original = store._atomic_write
    def fail_membership(path, payload):
        if "/membership/saved/" in str(path):
            raise OSError("membership unavailable")
        return original(path, payload)
    monkeypatch.setattr(store, "_atomic_write", fail_membership)
    with pytest.raises(RepoSaveError) as caught:
        repo.save_object(obj)
    publication = caught.value.report.publications[0]
    token, mutation_id = publication.retention_token, publication.mutation_id
    assert store.retention.read(token).disposition == "prepared"
    monkeypatch.setattr(store, "_atomic_write", original)
    store.repair_save_batch(mutation_id, token)
    assert store.retention.read(token).disposition == "adopted"
    assert not store.authority_cut().pending
    assert SavePlanLeaf.captures == 1
    assert store.read_node(store.retention.read(token).graph_digest, kind="saved") is not None


def test_withdrawn_key_republication_uses_separate_exclusive_phase(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    obj = SavePlanLeaf("a", repo=repo)
    saved, report = repo.save_object(obj, report_stores=True)
    store.retention.release(report.publications[0].retention_token, (store,))
    assert store.delete_state(saved.state.key).status == "removed"
    again = repo.save_object(obj)
    assert again.state.key == saved.state.key
    assert store.delete_state(again.state.key).status == "retained"


def test_selected_variants_share_one_header_interpretation_at_valid_cut(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    first = SavePlanLeaf("same", repo=repo)
    second = SavePlanLeaf("same", repo=repo)
    one = repo.save_object(first)
    second.name = "changed private data"
    two = repo.save_object(second)
    assert one.state.key.cdef_digest == two.state.key.cdef_digest
    root = SavePlanNode([first, second], repo=repo)
    calls = []
    original = store.read_creation
    def count(digest):
        calls.append(digest)
        return original(digest)
    monkeypatch.setattr(store, "read_creation", count)
    SavePlanLeaf.captures = 0
    saved = repo.save_object(root, states={child_path(0): one.state, child_path(1): two.state})
    assert SavePlanLeaf.captures == 0
    assert calls == [one.state.key.cdef_digest]
    assert len(saved.state_spec()) == 2


def test_missing_selected_bytes_and_malformed_metadata_fail_before_hooks(tmp_path):
    from dryml.core import SaveAnnotations
    from dryml.core.repo import RepoSaveError
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    obj = SavePlanLeaf("a", repo=repo)
    saved, report = repo.save_object(obj, report_stores=True)
    store.retention.release(report.publications[0].retention_token, (store,))
    store.delete_state(saved.state.key)
    SavePlanLeaf.captures = 0
    with pytest.raises(RepoSaveError, match="unavailable"):
        repo.save_object(obj, states=saved.state_spec())
    with pytest.raises(TypeError):
        repo.save_object(obj, annotations=SaveAnnotations(state={"unsupported": object()}))
    assert SavePlanLeaf.captures == 0
    current = repo.save_object(SavePlanLeaf("metadata", repo=repo), annotations=SaveAnnotations(state={}))
    assert SavePlanLeaf.captures == 1
    assert repo.get_snapshot_metadata(current).captured_state_annotations == {}
