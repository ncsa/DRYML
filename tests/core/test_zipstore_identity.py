"""Store-4 archive buffering, byte-CAS, commit, reopening and lease contracts.

Low-level Node/state/name publication is transaction-local until archive commit.
User-retained Repo saves require durable grants and refuse buffered archives.
Historical test IDs retain the original archive obligations at these seams.
"""

from io import BytesIO
from contextlib import contextmanager
import multiprocessing
from pathlib import Path
import threading
import zipfile

import pytest

import dryml.filesystem as filesystem
import dryml.core.store as store_exports
import dryml.core.store.zip as zip_module
from dryml.core import Definition, Object, ObjectId, Repo, SaveRouting, Serializable, StoreReport
from dryml.core.repo import RepoSaveError, _commit_save_report
from dryml.core.store.records import NodeNameRecord
from dryml.core.store.store import StoreAuthorityError, StoreCapabilityError
from dryml.core.store.zip import ZipStore, ZipStoreConflictError, ZipStoreCommitError

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


@contextmanager
def _open(path, *, existing=False):
    store = ZipStore.open_existing(path) if existing else ZipStore(path)
    try:
        yield store
    finally:
        store.close()


class ZipRecordObject(Object):
    """Stateless archive recipe fixture."""

    def __init__(self, value=""):
        self.value = value


class ZipPayloadObject(Serializable):
    """State capture fixture used to prove retained-save preflight ordering."""

    def __init__(self, value="payload"):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "payload.txt").write_text(self.value)


def _record(value="record"):
    return Definition(ZipRecordObject, value=value).concretize()


def _saved(store, value="state"):
    """Publish a real full-key saved root without claiming durable retention."""
    origin = Definition(ZipPayloadObject, value="creation").concretize()
    stage = store.create_local_state_staging()
    Path(stage, "data", "payload.txt").write_text(value)
    installed = store.install_state(origin, stage, codec="pkl", codec_version=1, interpretation={})
    assert installed.durability == "buffered"
    node = origin.with_identity(ObjectId()).with_state(installed.state)
    record = store.publish_node(node, kind="saved")
    return node, record


def _assert_root(store, record, node, kind="recipe"):
    assert store.read_node(record.digest, kind=kind).graph_equal(node)


def _roots(store, kind="recipe"):
    from dryml.core.node_prepare import NodePreparation
    nodes = store.query().saved().collect() if kind == "saved" else store.query().cdefs().collect()
    domain = "dryml-saved-root/v1" if kind == "saved" else "dryml-recipe-root/v1"
    return {NodePreparation((node,)).digest(node, domain=domain) for node in nodes}


def _concurrent_direct_record_commit(path, value, ready, start, results):
    """Stage native Node authority and compete on committed archive bytes."""
    store = ZipStore(path)
    try:
        node = _record(value)
        ready.set()
        assert start.wait(30)
        record = store.publish_node(node)
        store.commit()
        results.put(("published", record.digest))
    except ZipStoreConflictError:
        results.put(("conflict", None))
    except BaseException as error:
        results.put(("error", repr(error)))
    finally:
        store.close()


def test_zipstore_round_trips_current_logical_definition_authority(tmp_path):
    path, node = tmp_path / "store.zip", _record()
    store = ZipStore(path)
    record = store.publish_node(node)
    store.commit()
    store.close()
    with _open(path) as reopened:
        _assert_root(reopened, record, node)
        assert _roots(reopened) == {record.digest}


def test_path_backed_zip_buffers_direct_record_mutation_until_commit(tmp_path):
    path, node = tmp_path / "store.zip", _record()
    with _open(path) as store:
        record = store.publish_node(node)
        assert not path.exists() and store.committed_archive_cut is None
        _assert_root(store, record, node)
        store.commit()
        assert path.is_file() and store.committed_archive_cut == store.authority_cut()
    with _open(path) as reopened:
        _assert_root(reopened, record, node)


@pytest.mark.parametrize("kind", ["missing", "empty", "malformed"], ids=str)
def test_existing_only_zip_open_rejects_uncommitted_or_invalid_archive(tmp_path, kind):
    path = tmp_path / f"{kind}.zip"
    if kind == "empty":
        path.touch()
    elif kind == "malformed":
        path.write_bytes(b"not a zip archive")
    with pytest.raises(StoreAuthorityError):
        ZipStore.open_existing(path)
    assert not path.exists() if kind == "missing" else path.is_file()


def test_open_existing_validates_archive_integrity_once(tmp_path, monkeypatch):
    path = tmp_path / "store.zip"
    with _open(path) as store:
        store.publish_node(_record())
        store.commit()
    validated, original = [], ZipStore._validate_existing_archive
    def observe(cls, archive_path):
        validated.append(archive_path)
        return original(archive_path)
    monkeypatch.setattr(ZipStore, "_validate_existing_archive", classmethod(observe))
    with _open(path, existing=True):
        assert len(validated) == 1


def test_repo_save_reports_and_commits_a_buffered_zip_publication(tmp_path, monkeypatch):
    """Refuse retained save before capture; native buffered commits still report."""
    path = tmp_path / "repo-save.zip"
    with _open(path) as store:
        repo = Repo(store)
        value = ZipPayloadObject(repo=repo)
        monkeypatch.setattr(ZipPayloadObject, "save_state_to_dir_imp", lambda *a, **k: pytest.fail("capture"))
        with pytest.raises(StoreCapabilityError, match="durable"):
            repo.save(value, report_stores=True)
        assert value.last_state_ref is None and not path.exists()
        assert not store._archive_dirty and store.authority_cut().epoch == 0
        node, record = _saved(store)
        report = _commit_save_report(repo, None, StoreReport((store,), {}, ()), stores=(store,))
        assert [(p.phase, p.status) for p in report.publications] == [("commit", "completed")]
        assert not store._archive_dirty
        with _open(path, existing=True) as reopened:
            _assert_root(reopened, record, node, "saved")


def test_repo_save_reports_an_unrelated_configured_commit_failure(tmp_path, monkeypatch):
    with _open(tmp_path / "target.zip") as target, _open(tmp_path / "unrelated.zip") as unrelated:
        node = _record()
        record = target.publish_node(node)
        unrelated.publish_node(_record("unrelated"))
        monkeypatch.setattr(unrelated, "commit", lambda: (_ for _ in ()).throw(OSError("unrelated commit failed")))
        with pytest.raises(RepoSaveError) as raised:
            _commit_save_report(Repo([target, unrelated]), None, StoreReport((target, unrelated), {}, ()),
                                stores=(target, unrelated))
        assert [(p.store, p.status) for p in raised.value.report.publications] == [
            (target, "completed"), (unrelated, "failed")]
        with _open(target.archive_path, existing=True) as reopened:
            _assert_root(reopened, record, node)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("after", [False, True], ids=["before", "after"])
def test_routed_zip_commit_control_flow_retains_identity_and_report(tmp_path, monkeypatch, error_type, after):
    """The actual commit ledger preserves control flow and observable byte cut."""
    path = tmp_path / "interrupted.zip"
    with _open(path) as store:
        node, record = _saved(store, "changed")
        original = store.commit
        def interrupt():
            if after:
                original()
            raise error_type("commit interruption")
        monkeypatch.setattr(store, "commit", interrupt)
        with pytest.raises(error_type, match="commit interruption") as raised:
            _commit_save_report(Repo(store, save_routing=SaveRouting()), None,
                                StoreReport((store,), {}, ()), stores=(store,))
        assert [(p.phase, p.status) for p in raised.value.report.publications] == [
            ("commit", "completed" if after else "failed")]
        assert path.exists() is after
        assert (store.committed_archive_cut is not None) is after
        if after:
            with _open(path, existing=True) as reopened:
                _assert_root(reopened, record, node, "saved")


def test_relative_zip_path_retains_original_archive_destination_after_cwd_change(tmp_path, monkeypatch):
    original, later = tmp_path / "original", tmp_path / "later"
    original.mkdir()
    later.mkdir()
    monkeypatch.chdir(original)
    with _open("store.zip") as store:
        store.publish_node(_record("stable-path"))
        monkeypatch.chdir(later)
        store.commit()
    assert (original / "store.zip").is_file() and not (later / "store.zip").exists()


def test_no_op_zip_commit_hydrates_without_rewriting_archive_bytes(tmp_path):
    path = tmp_path / "store.zip"
    with _open(path) as store:
        record = store.publish_node(_record())
        store.commit()
    before = path.read_bytes()
    with _open(path) as reader:
        assert _roots(reader) == {record.digest}
        reader.commit()
    assert path.read_bytes() == before


def test_no_op_stale_zip_commit_preserves_newer_archive_bytes(tmp_path):
    path = tmp_path / "store.zip"
    with _open(path) as initial:
        initial.publish_node(_record("initial"))
        initial.commit()
    with _open(path) as reader, _open(path) as writer:
        writer.publish_node(_record("newer"))
        writer.commit()
        before = path.read_bytes()
        reader.commit()
        assert path.read_bytes() == before


def test_zip_commit_builds_valid_complete_sibling_before_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path / "store.zip"
    validated, replacements = [], []
    original_testzip, original_publish = zipfile.ZipFile.testzip, filesystem.publish_file
    def validate(archive):
        validated.append(Path(archive.filename))
        return original_testzip(archive)
    with _open(path) as store:
        node, record = _saved(store)
        name = NodeNameRecord("current", "saved", record.digest)
        store.compare_and_set_node_name(name, expected=None)
        store.write_metadata(node, {"null": None, "empty": {}})
        expected = {p.relative_to(store.base_dir).as_posix(): p.read_bytes()
                    for p in Path(store.base_dir).rglob("*") if p.is_file()
                    and not any(part in {".staging", ".state-leases", ".retention-locks"} for part in p.parts)
                    and str(p) != store._writer_lock_path}
        def inspect(source, destination, *, replace=False):
            if Path(destination) == path:
                staged = Path(source)
                assert staged.parent == path.parent and staged != path
                with zipfile.ZipFile(staged) as archive:
                    assert archive.testzip() is None
                    assert {n: archive.read(n) for n in archive.namelist()} == expected
                replacements.append(staged)
            return original_publish(source, destination, replace=replace)
        monkeypatch.setattr(zipfile.ZipFile, "testzip", validate)
        monkeypatch.setattr(filesystem, "publish_file", inspect)
        store.commit()
    assert len(validated) == 2 and len(replacements) == 1 and replacements[0].suffix == ".zip"


def test_failed_zip_commit_keeps_previous_complete_archive_bytes(tmp_path, monkeypatch):
    path = tmp_path / "store.zip"
    with _open(path) as store:
        first = store.publish_node(_record("first"))
        store.commit()
        before, cut = path.read_bytes(), store.committed_archive_cut
        store.publish_node(_record("second"))
        original = filesystem.publish_file
        def fail(source, destination, *, replace=False):
            if Path(destination) == path:
                raise OSError("injected archive replacement failure")
            return original(source, destination, replace=replace)
        monkeypatch.setattr(filesystem, "publish_file", fail)
        with pytest.raises(ZipStoreCommitError, match="archive replacement") as raised:
            store.commit()
        assert raised.value.cut == store.authority_cut() and raised.value.mutation_id
        assert raised.value.committed_archive_cut == cut and raised.value.durability == "buffered"
        assert path.read_bytes() == before and store._archive_dirty
        with _open(path) as reopened:
            assert _roots(reopened) == {first.digest}


def test_zip_post_publish_failure_preserves_complete_reopenable_replacement(tmp_path, monkeypatch):
    """Visible replacement remains uncertain until retry crosses its barrier."""
    path = tmp_path / "store.zip"
    with _open(path) as store:
        node = _record("replacement")
        record = store.publish_node(node)
        original = filesystem.publish_file
        def fail(source, destination, *, replace=False):
            result = original(source, destination, replace=replace)
            if Path(destination) == path:
                raise OSError("archive post-publication failure")
            return result
        monkeypatch.setattr(filesystem, "publish_file", fail)
        with pytest.raises(ZipStoreCommitError, match="archive post-publication") as raised:
            store.commit()
        assert store._archive_dirty and store.committed_archive_cut is None
        assert raised.value.mutation_id and raised.value.confirmed_refs
        with _open(path, existing=True) as reopened:
            _assert_root(reopened, record, node)
        monkeypatch.setattr(filesystem, "publish_file", original)
        store.commit()
        assert store.committed_archive_cut == store.authority_cut() and not store._archive_dirty


def test_stale_zip_writer_cannot_replace_newer_direct_definition_authority(tmp_path):
    path = tmp_path / "store.zip"
    with _open(path) as first, _open(path) as stale:
        one = first.publish_node(_record("first"))
        first.commit()
        before = path.read_bytes()
        two_node = _record("stale")
        two = stale.publish_node(two_node)
        with pytest.raises(ZipStoreConflictError, match="reopen"):
            stale.commit()
        assert path.read_bytes() == before
    with _open(path) as retry:
        retry.publish_node(two_node)
        retry.commit()
        assert _roots(retry) == {one.digest, two.digest}


def test_stale_zip_writer_cannot_replace_newer_direct_reference_authority(tmp_path):
    path = tmp_path / "store.zip"
    with _open(path) as first, _open(path) as stale:
        one = first.publish_node(_record("first"))
        first.compare_and_set_node_name(NodeNameRecord("first", "recipe", one.digest), expected=None)
        first.commit()
        before = path.read_bytes()
        two = stale.publish_node(_record("stale"))
        stale.compare_and_set_node_name(NodeNameRecord("stale", "recipe", two.digest), expected=None)
        with pytest.raises(ZipStoreConflictError, match="reopen"):
            stale.commit()
        assert path.read_bytes() == before
    with _open(path) as retry:
        assert retry.read_node_name("first").root_digest == one.digest
        assert retry.read_node_name("stale") is None


def test_path_backed_zip_serializes_direct_record_publication_across_processes(tmp_path):
    path = tmp_path / "store.zip"
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    ready = [context.Event(), context.Event()]
    processes = [context.Process(target=_concurrent_direct_record_commit,
                 args=(str(path), value, ready[i], start, results)) for i, value in enumerate(("first", "second"))]
    for process in processes:
        process.start()
    try:
        assert all(event.wait(30) for event in ready)
        start.set()
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        outcomes = [results.get(timeout=5) for _ in processes]
        assert sorted(item[0] for item in outcomes) == ["conflict", "published"], outcomes
        with _open(path) as reopened:
            assert _roots(reopened) == {next(d for status, d in outcomes if status == "published")}
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)


def test_malformed_or_escaping_zip_members_fail_closed_without_rewriting_bytes(tmp_path):
    malformed = tmp_path / "malformed.zip"
    malformed.write_bytes(b"not a zip archive")
    before = malformed.read_bytes()
    with pytest.raises(StoreAuthorityError, match="malformed"):
        ZipStore(malformed)
    assert malformed.read_bytes() == before
    escaping = tmp_path / "escaping.zip"
    with zipfile.ZipFile(escaping, "w") as archive:
        archive.writestr("../outside.record", b"outside")
        archive.writestr("..\\windows-outside.record", b"outside")
    before = escaping.read_bytes()
    with pytest.raises(StoreAuthorityError, match="escapes"):
        ZipStore(escaping)
    assert escaping.read_bytes() == before
    assert not (tmp_path / "outside.record").exists() and not (tmp_path / "windows-outside.record").exists()


def test_retired_object_root_archive_fails_at_the_direct_format_gate_without_rewrite(tmp_path):
    path = tmp_path / "retired.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("objects/aa/legacy/def.pkl", b"retired root authority")
    before = path.read_bytes()
    with pytest.raises(StoreAuthorityError, match="incompatible Store authority"):
        ZipStore(path)
    assert path.read_bytes() == before


def test_record_and_local_state_identity_do_not_depend_on_archive_staging_paths(tmp_path):
    keys, manifests = [], []
    for name in ("first", "second"):
        path = tmp_path / f"{name}.zip"
        with _open(path) as store:
            node, record = _saved(store, "same")
            with store.read_state(node.state.key) as source:
                keys.append(source.state.key)
                manifests.append(source.manifest)
            store.commit()
        with _open(path) as reopened:
            _assert_root(reopened, record, node, "saved")
            with reopened.read_state(node.state.key) as source:
                assert source.manifest == manifests[-1]
    assert keys[0] == keys[1] and manifests[0] == manifests[1]


def test_snapshot_and_mutable_references_round_trip_through_zip(tmp_path):
    path = tmp_path / "store.zip"
    with _open(path) as store:
        node, saved = _saved(store)
        recipe = store.publish_node(node.object_view())
        names = (NodeNameRecord("main", "recipe", recipe.digest), NodeNameRecord("current", "saved", saved.digest))
        for name in names:
            store.compare_and_set_node_name(name, expected=None)
        store.write_metadata(node.object_view(), {})
        store.write_metadata(node, {"state": None})
        store.commit()
    with _open(path) as reopened:
        _assert_root(reopened, recipe, node.object_view())
        _assert_root(reopened, saved, node, "saved")
        assert all(reopened.read_node_name(n.name) == n for n in names)
        assert reopened.read_declaration_anchor(node.object_id).sources
        assert reopened.read_metadata(node.object_view()) == {}
        assert reopened.read_metadata(node) == {"state": None}
        with reopened.read_state(node.state.key) as source:
            assert Path(source.data_directory, "payload.txt").read_text() == "state"


def test_zip_hydration_derives_the_memory_query_index_from_definition_records(tmp_path):
    path = tmp_path / "store.zip"
    with _open(path) as store:
        records = [store.publish_node(_record(value)) for value in ("first", "second")]
        store.commit()
    with _open(path) as reopened:
        assert _roots(reopened) == {r.digest for r in records}
        assert reopened.open_query_index() is None and reopened.query_index_status().backend == "memory"
        with zipfile.ZipFile(path) as archive:
            assert not any(n.startswith(".dryml/") for n in archive.namelist())


def test_zip_close_discards_uncommitted_buffer_and_cleans_extraction(tmp_path):
    path = tmp_path / "store.zip"
    store = ZipStore(path)
    extracted = Path(store.base_dir)
    store.publish_node(_record())
    store.close()
    store.close()
    assert not extracted.exists() and not path.exists()


def test_zip_close_invalidates_borrowed_snapshot_paths(tmp_path):
    store = ZipStore(tmp_path / "store.zip")
    node, _ = _saved(store)
    with store.read_state(node.state.key) as borrowed:
        path = Path(borrowed.data_directory)
    store.close()
    assert not path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        _ = borrowed.data_directory


def test_closed_dirty_zip_cannot_replace_committed_archive(tmp_path):
    path = tmp_path / "store.zip"
    with _open(path) as writer:
        committed = writer.publish_node(_record("committed"))
        writer.commit()
    before = path.read_bytes()
    stale = ZipStore(path)
    stale.publish_node(_record("discarded"))
    stale.close()
    for action in (stale.commit, lambda: stale.read_node(committed.digest), lambda: stale.publish_node(_record("late"))):
        with pytest.raises((RuntimeError, StoreAuthorityError), match="closed"):
            action()
    assert path.read_bytes() == before
    with _open(path) as reopened:
        assert _roots(reopened) == {committed.digest}


def test_zip_export_store_is_retired_from_the_public_archive_surface():
    assert "ZipExportStore" not in store_exports.__all__
    assert not hasattr(store_exports, "ZipExportStore") and not hasattr(zip_module, "ZipExportStore")


def test_file_like_zip_is_explicitly_read_only_for_current_authority():
    with _open(BytesIO()) as store:
        with pytest.raises(StoreCapabilityError, match="writable"):
            store.publish_node(_record())


def test_unconfigured_repo_accepts_file_like_zip_store():
    with _open(BytesIO()) as store:
        repo = Repo(store)
        assert repo.stores == [store] and repo.default_store is store
        repo.close(flush=False)


def test_file_like_zip_hydrates_current_authority_without_permitting_mutation(tmp_path):
    path, node = tmp_path / "store.zip", _record("read-only")
    with _open(path) as writer:
        record = writer.publish_node(node)
        writer.commit()
    before = path.read_bytes()
    stream = BytesIO(before)
    with _open(stream) as reader:
        _assert_root(reader, record, node)
        assert not reader.publication_capabilities.writable
        with pytest.raises(StoreCapabilityError, match="writable"):
            reader.publish_node(node)
    assert stream.getvalue() == before


def test_stale_zip_save_cannot_replace_a_newer_exact_state_ref(tmp_path):
    path = tmp_path / "state.zip"
    with _open(path) as first, _open(path) as stale:
        current, record = _saved(first, "first")
        old, old_record = _saved(stale, "stale")
        assert current.state.key != old.state.key
        first.commit()
        before = path.read_bytes()
        with pytest.raises(ZipStoreConflictError, match="reopen"):
            stale.commit()
        assert path.read_bytes() == before
        with _open(path) as reopened:
            _assert_root(reopened, record, current, "saved")
            assert reopened.read_node(old_record.digest, kind="saved") is None
            assert reopened.inspect_states(old.state.key.cdef_digest, available_only=True)[0].state.key == current.state.key


def test_same_handle_commit_does_not_clear_a_concurrent_mutation(tmp_path, monkeypatch):
    path = tmp_path / "fenced.zip"
    with _open(path) as store:
        one = store.publish_node(_record("first"))
        entered, release, errors, second = threading.Event(), threading.Event(), [], []
        original = store._archive_identity
        def pause(candidate=None):
            result = original(candidate)
            if candidate is not None and not entered.is_set():
                entered.set()
                assert release.wait(10)
            return result
        def run(action):
            try:
                action()
            except BaseException as error:
                errors.append(error)
        monkeypatch.setattr(store, "_archive_identity", pause)
        commit = threading.Thread(target=lambda: run(store.commit))
        commit.start()
        assert entered.wait(10)
        mutation = threading.Thread(target=lambda: run(lambda: second.append(store.publish_node(_record("second")))))
        mutation.start()
        release.set()
        commit.join(10)
        mutation.join(10)
        assert not commit.is_alive() and not mutation.is_alive() and not errors
        assert store._archive_dirty
        store.commit()
    with _open(path) as reopened:
        assert _roots(reopened) == {one.digest, second[0].digest}


def test_stale_zip_current_state_writer_must_reopen_and_reapply(tmp_path):
    path = tmp_path / "current.zip"
    with _open(path) as initial:
        _, record = _saved(initial, "initial")
        expected = NodeNameRecord("history", "saved", record.digest)
        initial.compare_and_set_node_name(expected, expected=None)
        initial.commit()
    with _open(path) as winner, _open(path) as stale:
        _, current = _saved(winner, "winner")
        name = NodeNameRecord("history", "saved", current.digest)
        winner.compare_and_set_node_name(name, expected=expected)
        winner.commit()
        before = path.read_bytes()
        _, rejected = _saved(stale, "stale")
        stale.compare_and_set_node_name(NodeNameRecord("history", "saved", rejected.digest), expected=expected)
        with pytest.raises(ZipStoreConflictError, match="reopen"):
            stale.commit()
        assert path.read_bytes() == before
    with _open(path) as retry:
        assert retry.read_node_name("history") == name
        _, reapplied = _saved(retry, "stale")
        latest = NodeNameRecord("history", "saved", reapplied.digest)
        retry.compare_and_set_node_name(latest, expected=name)
        retry.commit()
    with _open(path) as reopened:
        assert reopened.read_node_name("history") == latest
