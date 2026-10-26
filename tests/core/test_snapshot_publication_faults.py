"""Deterministic failures at Store-4 combined save and archive commit seams.

Historical test IDs preserve lineage, annotation, retry and source obligations;
the current ledger has one snapshot batch per target, not independent old phases.
"""

from contextlib import contextmanager
from pathlib import Path

import pytest

import dryml.filesystem as filesystem
import dryml.environments as envs
from dryml.core import (MetadataConflictError, Object, Repo, SaveAnnotations,
                        SaveRouting, Selector, Serializable, StoreReport)
from dryml.core.repo import RepoSaveError, _commit_save_report
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError
from dryml.core.store.zip import ZipStore
from dryml.core.utils.graph.path import GraphPath, Parameter
from tests.core.test_zipstore_identity import _saved, ZipPayloadObject
from tests.core.test_repo_aliases import _saved_digest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class FaultPayload(Serializable):
    """Small stateful value with deterministic payload capture and restore."""

    def __init__(self, value="value"):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        Path(directory, "value").write_text(self.value, encoding="ascii")

    def restore_state_from_dir_imp(self, directory, *, codec):
        self.value = Path(directory, "value").read_text(encoding="ascii")


class FaultContainer(Object):
    """Stateless root with multiple lineage-bearing children."""

    def __init__(self, left, right):
        self.left, self.right = left, right


def _publication(error):
    publications = error.report.publications
    assert {p.phase for p in publications} == {"snapshot"}
    return publications[0]


def _pending(store, publication):
    assert publication.status == "uncertain"
    assert publication.mutation_id and publication.retention_token
    assert publication.mutation_id in store.authority_cut().pending
    assert publication.index_disposition == "repair_pending"


def _interrupt_record(store, monkeypatch, fragment, error, *, after=True):
    original = store._atomic_write
    entered = []
    def operation(path, data):
        if fragment not in str(path).replace("\\", "/"):
            return original(path, data)
        entered.append(Path(path))
        if after:
            original(path, data)
        raise error
    monkeypatch.setattr(store, "_atomic_write", operation)
    return entered


def _observe_environment(observed, value):
    observed.append(value)
    return envs.EnvironmentRecord(
        python=envs.PythonRecord("3.12.0", "CPython"),
        platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
        distributions={}, dryml=envs.DrymlRuntimeRecord())


@pytest.mark.parametrize("after", [False, True], ids=["before", "after"])
def test_lineage_interruption_reports_only_entered_authority(tmp_path, monkeypatch, after):
    """Partial batch retains actual lineage visibility, grants and pending cut."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    entered = _interrupt_record(store, monkeypatch, "/metadata/lineage/",
                                KeyboardInterrupt("lineage interruption"), after=after)
    with pytest.raises(KeyboardInterrupt, match="lineage interruption") as raised:
        repo.save_object(FaultPayload(repo=repo), alias="latest")
    publication = _publication(raised.value)
    _pending(store, publication)
    assert len(entered) == 1 and entered[0].exists() is after
    state = publication.state_ref
    assert store.read_node(_saved_digest(state), kind="saved").graph_equal(state)
    assert store.read_node_name("latest").root_digest == _saved_digest(state)
    assert publication.state_keys == publication.protected_state_keys == (state.state.key,)
    assert store._read_node_metadata(state, "capture") is None


def test_post_snapshot_current_object_interruption_reports_completed_snapshot_and_mapping(tmp_path, monkeypatch):
    """Current-object visibility does not certify later state/capture/index work."""
    store = DirStore(tmp_path / "store")
    _interrupt_record(store, monkeypatch, "/metadata/current_object/", KeyboardInterrupt("object mapping interruption"))
    with pytest.raises(KeyboardInterrupt, match="object mapping") as raised:
        Repo(store).save_object(FaultPayload(), annotations=SaveAnnotations(object={"object": "installed"}, state={"state": "later"}))
    publication = _publication(raised.value)
    _pending(store, publication)
    state = publication.state_ref
    assert store.read_metadata(state.object_view()) == {"object": "installed"}
    assert store.read_metadata(state) is None and store._read_node_metadata(state, "capture") is None
    assert publication.protected_state_keys == (state.state.key,)


def test_descendant_lineage_failure_preserves_exact_completed_path_statuses(tmp_path, monkeypatch):
    """Path-specific lineage bytes survive a second-lineage batch failure."""
    store = DirStore(tmp_path / "store")
    original, entered = store._atomic_write, []
    def fail_second(path, data):
        if "/metadata/lineage/" in str(path).replace("\\", "/"):
            entered.append(Path(path))
            if len(entered) == 2:
                raise OSError("descendant lineage failed")
        return original(path, data)
    monkeypatch.setattr(store, "_atomic_write", fail_second)
    with pytest.raises(RepoSaveError) as raised:
        Repo(store).save_object(FaultContainer(FaultPayload("left"), FaultPayload("right")))
    publication = _publication(raised.value)
    _pending(store, publication)
    assert [p.exists() for p in entered] == [True, False]
    saved = publication.state_ref
    paths = (GraphPath(), GraphPath((Parameter("left"),)), GraphPath((Parameter("right"),)))
    records = [Path(store._node_metadata_path("lineage", saved.at(p).object_projection_digest())) for p in paths]
    assert [r.exists() for r in records] == [False, True, False]
    assert saved.object_id is None
    assert len({saved.at(p).object_id for p in paths[1:]}) == 2
    assert len(publication.state_keys) == len(publication.protected_state_keys) == 2


def test_native_annotations_preserve_first_capture(tmp_path):
    """Current annotation replacements preserve immutable first-capture evidence."""
    store = DirStore(tmp_path / "store")
    repo, value = Repo(store), FaultPayload()
    state = repo.save_object(value)
    captured = store._read_node_metadata(state, "capture").to_bytes()
    assert repo.save_object(value, annotations=SaveAnnotations(object={}, state={"state": True})).graph_equal(state)
    assert store.read_metadata(state.object_view()) == {} and store.read_metadata(state) == {"state": True}
    assert store._read_node_metadata(state, "capture").to_bytes() == captured
    assert not store.authority_cut().pending


def test_annotations_apply_to_each_routed_root_with_actual_writes(tmp_path):
    """Each published root gets requested annotations; unannotated copies stay absent."""
    root, child = DirStore(tmp_path / "root"), DirStore(tmp_path / "child")
    repo = Repo([root, child], save_routing=SaveRouting(
        ((Selector(FaultContainer), root), (Selector(FaultPayload), child)), graph_mode="per-object"))
    state = repo.save_object(FaultContainer(FaultPayload("left"), FaultPayload("right")),
                            annotations=SaveAnnotations(object={"root": True}))
    assert root.read_metadata(state.object_view()) == {"root": True}
    for name in ("left", "right"):
        descendant = state.at(GraphPath((Parameter(name),)))
        assert child.read_node(_saved_digest(descendant), kind="saved").graph_equal(descendant)
        assert child.read_metadata(descendant.object_view()) == {"root": True}
        assert child.read_metadata(descendant) is None
    plain = Repo(child).save_object(FaultPayload("unannotated"))
    assert child.read_metadata(plain.object_view()) is None
    assert child.read_metadata(plain) is None
    replica = DirStore(tmp_path / "replica")
    copied = Repo([child, replica]).save_object(FaultPayload("unannotated"), store=replica,
                                               states=plain.state_spec(), source_store=child)
    assert copied.object_id != plain.object_id and copied.state == plain.state
    assert replica.read_metadata(copied.object_view()) is None
    assert replica.read_metadata(copied) is None
    assert not root.authority_cut().pending and not child.authority_cut().pending


def test_post_snapshot_current_state_interruption_reports_both_completed_mappings(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    _interrupt_record(store, monkeypatch, "/metadata/current_state/", KeyboardInterrupt("state mapping interruption"))
    with pytest.raises(KeyboardInterrupt, match="state mapping") as raised:
        Repo(store).save_object(FaultPayload(), annotations=SaveAnnotations(object={"object": "installed"}, state={"state": "installed"}))
    publication = _publication(raised.value)
    _pending(store, publication)
    state = publication.state_ref
    assert store.read_metadata(state.object_view()) == {"object": "installed"}
    assert store.read_metadata(state) == {"state": "installed"}
    assert store._read_node_metadata(state, "capture") is None


@pytest.mark.parametrize("boundary", ["capture", "ack"])
def test_snapshot_post_publish_interruption_retains_completed_authority(tmp_path, monkeypatch, boundary):
    """Installed capture is readable while batch failure remains explicitly pending."""
    store = DirStore(tmp_path / "store")
    value = FaultPayload()
    original = store._atomic_write
    if boundary == "capture":
        _interrupt_record(store, monkeypatch, "/metadata/capture/", OSError("snapshot post-publication interruption"))
    else:
        from dryml.core.store import authority
        original_ack = authority.acknowledge
        monkeypatch.setattr(authority, "acknowledge", lambda *a: (_ for _ in ()).throw(OSError("index acknowledgement failed")))
    with pytest.raises(RepoSaveError, match="publication") as raised:
        Repo(store).save_object(value, alias="latest")
    publication = _publication(raised.value)
    _pending(store, publication)
    state = publication.state_ref
    assert Repo(store).get_snapshot_metadata(state).state_ref.graph_equal(state)
    assert store.read_node(_saved_digest(state), kind="saved").graph_equal(state)
    assert publication.protected_state_keys == (state.state.key,)
    assert store.open_query_index().status().state == "dirty"
    assert store.read_node_name("latest").root_digest == _saved_digest(state)
    error = raised.value.__cause__
    assert error.mutation_id == publication.mutation_id
    assert error.cut.pending and error.confirmed_refs
    assert error.index_disposition == "repair_pending"
    monkeypatch.setattr(FaultPayload, "save_state_to_dir_imp", lambda *a, **k: pytest.fail("repair capture"))
    if boundary == "capture":
        monkeypatch.setattr(store, "_atomic_write", original)
        store.repair_save_batch(publication.mutation_id, publication.retention_token)
    else:
        monkeypatch.setattr(authority, "acknowledge", original_ack)
        store.open_query_index().maintain_store()
    assert not store.authority_cut().pending and store.open_query_index().status().state == "ready"


def test_retry_before_snapshot_install_captures_new_evidence(tmp_path, monkeypatch):
    """No capture association exists before the batch; retry observes new evidence."""
    store = DirStore(tmp_path / "store")
    repo, value, observed = Repo(store), FaultPayload(), []
    original = store.publish_save_batch
    def fail(batch):
        raise OSError("staged snapshot replacement failed")
    monkeypatch.setattr(store, "publish_save_batch", fail)
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(value, _snapshot_observer=lambda: _observe_environment(observed, "first"))
    publication = _publication(raised.value)
    assert publication.status == "failed" and publication.mutation_id is None
    state = publication.state_ref
    assert store.read_node(_saved_digest(state), kind="saved") is None
    assert not Path(store._node_metadata_path("capture", _saved_digest(state))).exists()
    monkeypatch.setattr(store, "publish_save_batch", original)
    assert repo.save_object(value, _snapshot_observer=lambda: _observe_environment(observed, "retry")).graph_equal(state)
    assert observed == ["first", "retry"] and not store.authority_cut().pending


def test_retry_after_post_snapshot_failure_reuses_evidence_without_replaying_annotations(tmp_path, monkeypatch):
    """Repair preserves captured bytes; explicit retry annotations are fresh LWW."""
    store = DirStore(tmp_path / "store")
    repo, value = Repo(store), FaultPayload("same")
    original = store._atomic_write
    _interrupt_record(store, monkeypatch, "/metadata/capture/", OSError("first current mapping fails"))
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(value, annotations=SaveAnnotations(object={"first": 1}))
    publication = _publication(raised.value)
    _pending(store, publication)
    state = publication.state_ref
    captured = store._read_node_metadata(state, "capture").to_bytes()
    monkeypatch.setattr(store, "_atomic_write", original)
    monkeypatch.setattr(FaultPayload, "save_state_to_dir_imp", lambda *a, **k: pytest.fail("repair recaptured"))
    store.repair_save_batch(publication.mutation_id, publication.retention_token)
    reopened = Repo(DirStore(store.base_dir))
    reopened.set_metadata(state.object_view(), {"intervening": 2})
    assert reopened.save_object(value, states=state.state_spec()).graph_equal(state)
    assert reopened.get_metadata(state.object_view()) == {"intervening": 2}
    assert reopened.save_object(value, states=state.state_spec(), annotations=SaveAnnotations(object={"retry": 3})).graph_equal(state)
    assert reopened.get_metadata(state.object_view()) == {"retry": 3}
    assert store._read_node_metadata(state, "capture").to_bytes() == captured
    assert not store.authority_cut().pending


@pytest.mark.parametrize("explicit", [False, True], ids=["capture", "selected"])
def test_malformed_final_snapshot_is_preserved_and_not_recaptured(tmp_path, monkeypatch, explicit):
    store = DirStore(tmp_path / "store")
    repo, value = Repo(store), FaultPayload("same")
    state = repo.save_object(value)
    path = Path(store._node_metadata_path("capture", _saved_digest(state)))
    original = path.read_bytes()
    path.write_bytes(b"not metadata")
    observed = []
    capture = FaultPayload.save_state_to_dir_imp
    def observe(*args, **kwargs):
        observed.append("capture")
        return capture(*args, **kwargs)
    monkeypatch.setattr(FaultPayload, "save_state_to_dir_imp", observe)
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(value, states=state.state_spec() if explicit else None,
                         _snapshot_observer=lambda: _observe_environment(observed, "probe"))
    assert isinstance(raised.value.__cause__, StoreAuthorityError)
    assert observed == [] and path.read_bytes() == b"not metadata" != original


def test_zip_commit_interruption_after_replace_reports_completed_and_reopens(tmp_path, monkeypatch):
    """Commit wrapper interruption recognizes a completed native archive commit."""
    path = tmp_path / "store.zip"
    store = ZipStore(path)
    try:
        node, record = _saved(store)
        original = store.commit
        def interrupt():
            original()
            raise KeyboardInterrupt("archive bookkeeping interruption")
        monkeypatch.setattr(store, "commit", interrupt)
        with pytest.raises(KeyboardInterrupt, match="archive bookkeeping") as raised:
            _commit_save_report(Repo(store), None, StoreReport((store,), {}, ()), stores=(store,))
        assert [(p.phase, p.status) for p in raised.value.report.publications] == [("commit", "completed")]
        reopened = ZipStore.open_existing(path)
        try:
            assert reopened.read_node(record.digest, kind="saved").graph_equal(node)
            with reopened.read_state(node.state.key) as source:
                assert Path(source.data_directory, "payload.txt").read_text() == "state"
        finally:
            reopened.close()
    finally:
        store.close()


def test_current_metadata_post_publish_failure_retains_mapping_and_dirty_fallback(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = repo.save_object(FaultPayload())
    target = Path(store._node_metadata_path("current_object", state.object_projection_digest()))
    original = filesystem.publish_file
    def fail(source, destination, *, replace=False):
        result = original(source, destination, replace=replace)
        if Path(destination) == target:
            raise OSError("metadata post-publication failure")
        return result
    monkeypatch.setattr(filesystem, "publish_file", fail)
    with pytest.raises(OSError, match="metadata post-publication"):
        repo.set_metadata(state.object_view(), {"complete": True})
    assert store.open_query_index().status().state == "dirty" and store.authority_cut().pending
    assert DirStore(store.base_dir, query_index="none").read_metadata(state.object_view()) == {"complete": True}


def test_conflicting_copy_and_corrupt_or_expired_sources_do_not_claim_destination(tmp_path, monkeypatch):
    """Selected-source conflicts/corruption/lifetime never create target authority."""
    first, conflicting = DirStore(tmp_path / "first"), DirStore(tmp_path / "conflicting")
    value = FaultPayload()
    state = Repo(first).save_object(value, annotations=SaveAnnotations(object={"source": 1}))
    Repo(conflicting).save_object(value, annotations=SaveAnnotations(object={"source": 2}))
    before = [s._read_node_metadata(state, "capture").to_bytes() for s in (first, conflicting)]
    assert before[0] != before[1]
    with pytest.raises(MetadataConflictError):
        Repo([first, conflicting]).load(state)
    assert [s._read_node_metadata(state, "capture").to_bytes() for s in (first, conflicting)] == before
    target = DirStore(tmp_path / "corrupt-target")
    manifest = Path(first.state_data_path(state.state.key)).parent / "manifest.record"
    manifest.write_bytes(b"corrupt source manifest")
    with pytest.raises(StoreAuthorityError):
        Repo([first, conflicting, target]).save_object(value, store=target, states=state.state_spec(), source_store=first)
    assert not target.query().saved().collect() and target.authority_cut().epoch == 0
    assert manifest.read_bytes() == b"corrupt source manifest"
    archive = ZipStore(tmp_path / "source.zip")
    try:
        selected, _ = _saved(archive)
        archive.commit()
        expired = DirStore(tmp_path / "expired-target")
        original = archive.read_states
        entered = []
        archive_bytes = Path(archive.archive_path).read_bytes()
        @contextmanager
        def close_after_read(keys):
            with original(keys) as sources:
                entered.extend(sources)
                archive.close()
                yield sources
        monkeypatch.setattr(archive, "read_states", close_after_read)
        with pytest.raises(RuntimeError, match="closed"):
            Repo([archive, expired]).save_object(
                # Match the native archive fixture's creation, with a fresh identity.
                ZipPayloadObject("creation"),
                store=expired, states=selected.state_spec(), source_store=archive)
        assert entered == [selected.state.key]
        assert Path(archive.archive_path).read_bytes() == archive_bytes
        assert not expired.query().saved().collect() and expired.authority_cut().epoch == 0
    finally:
        archive.close()
