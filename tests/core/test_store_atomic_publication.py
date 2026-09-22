import hashlib
import multiprocessing
import os
import pytest
import threading
from pathlib import Path

from dryml.core import Object, Repo, Serializable
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import (
    DefinitionRecord, LocalStateManifest, MainRefRecord, ObjectAliasRecord,
    StateAliasRecord, StoredRootRecord,
)
from dryml.core.store.store import StoreAuthorityError


class AtomicRecordObject(Object):
    def __init__(self, value=""):
        super().__init__()
        self.value = value


class AtomicPayloadObject(Serializable):
    def __init__(self, value="state"):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "payload.txt").write_text(self.value)


def _stage(store, record, payload=b"state"):
    stage = Path(store.create_local_state_staging())
    data = stage / "data"
    (data / "value.bin").write_bytes(payload)
    definition_bytes = record.to_bytes()
    manifest = LocalStateManifest(
        "Codec", record.graph_hash, record.digest,
        hashlib.sha256(definition_bytes).hexdigest(),
        (("value.bin", len(payload), hashlib.sha256(payload).hexdigest()),),
    )
    (stage / "def.pkl").write_bytes(definition_bytes)
    (stage / "manifest.record").write_bytes(manifest.to_bytes())
    return stage, manifest


def _write_main_ref_in_process(store_path, digest, attempted, completed, results):
    """Attempt one mutable reference replacement after the parent holds its lock."""
    try:
        store = DirStore(store_path, query_index="memory")
        attempted.set()
        store.write_main_ref(MainRefRecord(digest))
        results.put("ok")
    except BaseException as error:
        results.put(repr(error))
    finally:
        completed.set()


def test_immutable_definition_collision_is_idempotent_only_after_full_validation(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    store.write_definition_record(record)

    assert store.write_definition_record(record) == record
    path = Path(store.base_dir, "definitions", record.digest[:2], f"{record.digest}.record")
    path.write_bytes(b"not a record")

    with pytest.raises(Exception, match="Malformed Store record"):
        store.write_definition_record(record)


def test_direct_stored_root_read_validates_only_the_requested_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    store.write_definition_record(record)

    assert store.read_stored_root_record(record.digest) == StoredRootRecord(record.digest)
    assert store.read_stored_root_record("0" * 64) is None

    Path(store._definition_path(record.digest)).unlink()
    with pytest.raises(StoreAuthorityError, match="missing DefinitionRecord"):
        store.read_stored_root_record(record.digest)

    corrupt = DirStore(tmp_path / "corrupt")
    corrupt.write_definition_record(record)
    Path(corrupt._stored_root_path(record.digest)).write_bytes(b"not a record")
    with pytest.raises(StoreAuthorityError, match="Malformed Store record"):
        corrupt.read_stored_root_record(record.digest)


def test_short_definition_write_never_publishes_truncated_authority(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    target = Path(store.base_dir, "definitions", record.digest[:2], f"{record.digest}.record")
    original_fdopen = os.fdopen

    class ShortWriter:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def write(self, data):
            self.file.write(data[:-1])
            return len(data) - 1

        def __getattr__(self, name):
            return getattr(self.file, name)

    monkeypatch.setattr(os, "fdopen", lambda *args, **kwargs: ShortWriter(original_fdopen(*args, **kwargs)))
    with pytest.raises(OSError, match="incomplete"):
        store.write_definition_record(record)

    assert not target.exists()
    assert store.read_definition_record(record.digest) is None


def test_definition_replace_failure_leaves_no_new_immutable_authority(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    target = Path(store.base_dir, "definitions", record.digest[:2], f"{record.digest}.record")
    original_replace = store._replace_durable

    def fail_replace(source, destination, *, replace=True):
        if Path(destination) == target:
            raise OSError("injected immutable replacement failure")
        return original_replace(source, destination, replace=replace)

    monkeypatch.setattr(store, "_replace_durable", fail_replace)
    with pytest.raises(OSError, match="immutable replacement"):
        store.write_definition_record(record)

    assert not target.exists()
    assert store.read_definition_record(record.digest) is None


def test_mutable_reference_replace_leaves_previous_complete_record_on_failure(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    previous = MainRefRecord("1" * 64)
    replacement = MainRefRecord("2" * 64)
    store.write_main_ref(previous)
    target = Path(store.base_dir, "refs", "main.record")
    original_replace = store._replace_durable

    def fail_replace(source, destination, *, replace=True):
        if Path(destination) == target:
            raise OSError("injected reference replacement failure")
        return original_replace(source, destination, replace=replace)

    monkeypatch.setattr(store, "_replace_durable", fail_replace)
    with pytest.raises(OSError, match="replacement"):
        store.write_main_ref(replacement)

    assert store.read_main_ref() == previous


@pytest.mark.parametrize("reference", ["main", "object-alias", "state-alias"])
def test_mutable_reference_replacement_keeps_the_previous_complete_record_on_permission_failure(
        tmp_path, monkeypatch, reference):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first_object = AtomicPayloadObject("first", repo=repo)
    first = first_object.save(repo=repo)
    second = AtomicPayloadObject("second", repo=repo).save(repo=repo)
    first_object.value = "updated"
    updated = first_object.save(repo=repo, deep_capture=True)
    if reference == "main":
        previous = MainRefRecord("1" * 64)
        replacement = MainRefRecord("2" * 64)
        read = store.read_main_ref
        write = store.write_main_ref
        target = Path(store.base_dir, "refs", "main.record")
    elif reference == "object-alias":
        previous = ObjectAliasRecord("latest", first.object)
        replacement = ObjectAliasRecord("latest", second.object)
        read = lambda: store.read_object_alias("latest")
        write = store.write_object_alias
        target = Path(store.base_dir, "refs", "objects", "latest.record")
    else:
        previous = StateAliasRecord("latest", first.object, first.digest())
        replacement = StateAliasRecord("latest", first.object, updated.digest())
        read = lambda: store.read_state_alias(first.object.digest(), "latest")
        write = store.write_state_alias
        target = Path(store.base_dir, "refs", "states", first.object.digest()[:2], first.object.digest(), "latest.record")
    write(previous)
    before = target.read_bytes()
    original_replace = store._replace_durable

    def deny_replace(source, destination, *, replace=True):
        if Path(destination) == target:
            raise PermissionError("injected authority replacement denial")
        return original_replace(source, destination, replace=replace)

    monkeypatch.setattr(store, "_replace_durable", deny_replace)
    with pytest.raises(PermissionError, match="denial"):
        write(replacement)

    assert target.read_bytes() == before
    assert read() == previous


def test_preparation_interruption_leaves_staging_unpublished(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    stage, manifest = _stage(store, record)
    original_validate = store._validate_local_state_dir

    def interrupt_validation(*args, **kwargs):
        raise KeyboardInterrupt("injected during local-state preparation")

    monkeypatch.setattr(store, "_validate_local_state_dir", interrupt_validation)
    with pytest.raises(KeyboardInterrupt, match="preparation"):
        store.prepare_local_state(stage, manifest)

    assert stage.is_dir()
    monkeypatch.setattr(store, "_validate_local_state_dir", original_validate)


def test_prepared_local_state_remains_private_until_snapshot_publication(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    first_stage, manifest = _stage(store, record)
    first = store.prepare_local_state(first_stage, manifest)
    second_stage, duplicate = _stage(store, record)
    second = store.prepare_local_state(second_stage, duplicate)

    assert first.manifest == second.manifest == manifest
    assert Path(first.handle, "data", "value.bin").read_bytes() == b"state"
    assert Path(second.handle, "data", "value.bin").read_bytes() == b"state"


@pytest.mark.parametrize("mutation", ["malformed", "foreign-manifest", "missing", "extra", "symlink", "special"], ids=str)
def test_local_state_rejects_non_authoritative_manifest_payload_trees(tmp_path, mutation):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    stage, manifest = _stage(store, record)
    payload = stage / "data" / "value.bin"
    if mutation == "malformed":
        (stage / "manifest.record").write_bytes(b"not a manifest")
    elif mutation == "foreign-manifest":
        foreign = LocalStateManifest(
            manifest.codec, manifest.graph_hash, manifest.definition_digest,
            "0" * 64, manifest.files,
        )
        (stage / "manifest.record").write_bytes(foreign.to_bytes())
    elif mutation == "missing":
        payload.unlink()
    elif mutation == "extra":
        (stage / "data" / "extra.bin").write_bytes(b"extra")
    elif mutation == "symlink":
        (stage / "data" / "link.bin").symlink_to(payload)
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("platform has no FIFO support")
        os.mkfifo(stage / "data" / "pipe")

    with pytest.raises(StoreAuthorityError):
        store.prepare_local_state(stage, manifest)


def test_local_state_preparation_keeps_owned_staging_private(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    stage, manifest = _stage(store, record)
    source = store.prepare_local_state(stage, manifest)

    assert Path(source.handle) == stage
    assert not (Path(store.base_dir) / "snapshots").exists()


def test_prepared_local_state_remains_complete_while_another_staging_source_is_created(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    first_stage, first = _stage(store, record, b"first")
    stale_handle = Path(store.prepare_local_state(first_stage, first).handle)
    second_stage, second = _stage(store, record, b"second")
    second_source = store.prepare_local_state(second_stage, second)

    assert (stale_handle / "data" / "value.bin").read_bytes() == b"first"
    assert Path(second_source.handle, "data", "value.bin").read_bytes() == b"second"


def test_reference_readers_observe_only_complete_old_or_new_records(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    previous = MainRefRecord("1" * 64)
    replacement = MainRefRecord("2" * 64)
    store.write_main_ref(previous)
    target = Path(store.base_dir, "refs", "main.record")
    entered_replace = threading.Event()
    release_replace = threading.Event()
    errors = []
    original_replace = store._replace_durable

    def pause_before_replace(source, destination, *, replace=True):
        if Path(destination) == target:
            entered_replace.set()
            assert release_replace.wait(10)
        return original_replace(source, destination, replace=replace)

    def write():
        try:
            store.write_main_ref(replacement)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(store, "_replace_durable", pause_before_replace)
    writer = threading.Thread(target=write)
    writer.start()
    assert entered_replace.wait(10)
    assert {store.read_main_ref() for _ in range(20)} == {previous}
    release_replace.set()
    writer.join(10)

    assert not writer.is_alive()
    assert not errors
    assert store.read_main_ref() == replacement


def test_store_writer_lock_serializes_processes_at_the_reference_boundary(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    store.write_main_ref(MainRefRecord("1" * 64))
    context = multiprocessing.get_context("spawn")
    attempted = context.Event()
    completed = context.Event()
    results = context.Queue()
    writer = context.Process(
        target=_write_main_ref_in_process,
        args=(store.base_dir, "2" * 64, attempted, completed, results),
    )

    from dryml.locking import interprocess_lock
    with interprocess_lock(store._writer_lock_path):
        writer.start()
        assert attempted.wait(10)
        assert not completed.is_set()
    writer.join(10)

    assert not writer.is_alive()
    assert writer.exitcode == 0
    assert results.get(timeout=10) == "ok"
    assert store.read_main_ref() == MainRefRecord("2" * 64)


def test_closing_one_repo_does_not_close_a_shared_store_query_index(tmp_path):
    """Store-owned query indexes remain usable through another borrowing Repo."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    first = Repo(store)
    second = Repo(store)
    first_index = first._query_index.open_store_index(first._query_index.store_bindings[0])
    if first_index is None:
        pytest.skip("SQLite query indexes are unavailable")
    first.close(flush=False)

    second_index = second._query_index.open_store_index(second._query_index.store_bindings[0])
    assert second_index is first_index
    second.close(flush=False)
    store.close()


def test_definition_publication_interruption_keeps_authority_and_notifies_query_rebuild(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    record = DefinitionRecord(AtomicRecordObject().definition)
    original_replace = store._replace_durable

    def interrupt_after_dirty_marker_replace(
            source, destination, *, replace=True):
        result = original_replace(source, destination, replace=replace)
        if Path(destination).parent == Path(store.dryml_dir) and Path(destination).name.startswith("query-index.dirty."):
            raise KeyboardInterrupt("injected after dirty marker publication")
        return result

    monkeypatch.setattr(
        store, "_replace_durable", interrupt_after_dirty_marker_replace,
    )
    with pytest.raises(KeyboardInterrupt, match="dirty marker"):
        store.write_definition_record(record)

    assert store.read_definition_record(record.digest) == record
    assert store.query_index_is_dirty()


def test_state_ref_failure_leaves_only_verified_unreferenced_local_state(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = AtomicPayloadObject(repo=repo)

    monkeypatch.setattr(store, "publish_snapshot", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("snapshot failure")))
    with pytest.raises(RepoSaveError, match="publication") as raised:
        obj.save(repo=repo)
    assert isinstance(raised.value.__cause__, OSError)
    assert "snapshot failure" in str(raised.value.__cause__)
    assert raised.value.report is not None

    assert not (Path(store.base_dir) / "snapshots").exists()
    assert obj._last_state_hash is not None
