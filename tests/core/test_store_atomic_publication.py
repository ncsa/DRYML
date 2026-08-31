import hashlib
import multiprocessing
import os
import pytest
import threading
from pathlib import Path

from dryml.core import Object, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.store.records import (
    DefinitionRecord, LocalStateManifest, MainRefRecord, ObjectAliasRecord,
    StateAliasRecord,
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
    original_replace = os.replace

    def fail_replace(source, destination):
        if Path(destination) == target:
            raise OSError("injected immutable replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_replace)
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
    original_replace = os.replace

    def fail_replace(source, destination):
        if Path(destination) == target:
            raise OSError("injected reference replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_replace)
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
    original_replace = os.replace

    def deny_replace(source, destination):
        if Path(destination) == target:
            raise PermissionError("injected authority replacement denial")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", deny_replace)
    with pytest.raises(PermissionError, match="denial"):
        write(replacement)

    assert target.read_bytes() == before
    assert read() == previous


def test_install_interruption_after_atomic_directory_move_leaves_complete_new_state(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    stage, manifest = _stage(store, record)
    destination = Path(store.base_dir, "local-state", record.graph_hash[:2], record.graph_hash, manifest.state_hash)
    original_replace = os.replace

    def interrupt_after_move(source, target):
        result = original_replace(source, target)
        if Path(target) == destination:
            raise KeyboardInterrupt("injected after local-state install")
        return result

    monkeypatch.setattr(os, "replace", interrupt_after_move)
    with pytest.raises(KeyboardInterrupt, match="after local-state"):
        store.install_local_state(stage, manifest)

    assert Path(store.open_local_state(record.graph_hash, manifest.state_hash)).is_dir()


def test_interruption_before_direct_local_state_install_leaves_it_unpublished(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    stage, manifest = _stage(store, record)
    destination = Path(store.base_dir, "local-state", record.graph_hash[:2], record.graph_hash, manifest.state_hash)
    original_replace = os.replace

    def interrupt_before_move(source, target):
        if Path(target) == destination:
            raise KeyboardInterrupt("injected before local-state install")
        return original_replace(source, target)

    monkeypatch.setattr(os, "replace", interrupt_before_move)
    with pytest.raises(KeyboardInterrupt, match="before local-state"):
        store.install_local_state(stage, manifest)

    assert stage.is_dir()
    assert not destination.exists()
    with pytest.raises(StoreAuthorityError, match="missing"):
        store.open_local_state(record.graph_hash, manifest.state_hash)


def test_direct_local_state_is_immutable_and_idempotent_after_full_validation(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    first_stage, manifest = _stage(store, record)
    store.install_local_state(first_stage, manifest)
    destination = Path(store.open_local_state(record.graph_hash, manifest.state_hash))
    before = (destination / "data" / "value.bin").read_bytes()
    second_stage, duplicate = _stage(store, record)

    assert store.install_local_state(second_stage, duplicate) == manifest
    assert (destination / "data" / "value.bin").read_bytes() == before
    assert second_stage.is_dir()


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
        store.install_local_state(stage, manifest)

    destination = Path(store.base_dir, "local-state", record.graph_hash[:2], record.graph_hash, manifest.state_hash)
    assert not destination.exists()


def test_local_state_staging_must_remain_on_the_store_filesystem(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    stage, manifest = _stage(store, record)
    original_stat = os.stat

    class Device:
        def __init__(self, device):
            self.st_dev = device

    def different_devices(path, *args, **kwargs):
        if Path(path) == stage:
            return Device(1)
        if Path(path) == Path(store.base_dir):
            return Device(2)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", different_devices)
    with pytest.raises(StoreAuthorityError, match="filesystem"):
        store.install_local_state(stage, manifest)


def test_opened_immutable_local_state_remains_complete_after_a_new_state_install(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(AtomicRecordObject().definition)
    first_stage, first = _stage(store, record, b"first")
    store.install_local_state(first_stage, first)
    stale_handle = Path(store.open_local_state(record.graph_hash, first.state_hash))
    second_stage, second = _stage(store, record, b"second")
    store.install_local_state(second_stage, second)

    assert (stale_handle / "data" / "value.bin").read_bytes() == b"first"
    assert Path(store.open_local_state(record.graph_hash, second.state_hash), "data", "value.bin").read_bytes() == b"second"


def test_reference_readers_observe_only_complete_old_or_new_records(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    previous = MainRefRecord("1" * 64)
    replacement = MainRefRecord("2" * 64)
    store.write_main_ref(previous)
    target = Path(store.base_dir, "refs", "main.record")
    entered_replace = threading.Event()
    release_replace = threading.Event()
    errors = []
    original_replace = os.replace

    def pause_before_replace(source, destination):
        if Path(destination) == target:
            entered_replace.set()
            assert release_replace.wait(10)
        return original_replace(source, destination)

    def write():
        try:
            store.write_main_ref(replacement)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(os, "replace", pause_before_replace)
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

    from dryml.core.store.locking import interprocess_lock
    with interprocess_lock(store._writer_lock_path):
        writer.start()
        assert attempted.wait(10)
        assert not completed.is_set()
    writer.join(10)

    assert not writer.is_alive()
    assert writer.exitcode == 0
    assert results.get(timeout=10) == "ok"
    assert store.read_main_ref() == MainRefRecord("2" * 64)


def test_definition_publication_interruption_keeps_authority_and_notifies_query_rebuild(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    record = DefinitionRecord(AtomicRecordObject().definition)
    original_replace = os.replace

    def interrupt_after_dirty_marker_replace(source, destination):
        result = original_replace(source, destination)
        if Path(destination).parent == Path(store.dryml_dir) and Path(destination).name.startswith("query-index.dirty."):
            raise KeyboardInterrupt("injected after dirty marker publication")
        return result

    monkeypatch.setattr(os, "replace", interrupt_after_dirty_marker_replace)
    with pytest.raises(KeyboardInterrupt, match="dirty marker"):
        store.write_definition_record(record)

    assert store.read_definition_record(record.digest) == record
    assert store.query_index_is_dirty()


def test_state_ref_failure_leaves_only_verified_unreferenced_local_state(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = AtomicPayloadObject(repo=repo)

    monkeypatch.setattr(store, "write_state_ref_record", lambda record: (_ for _ in ()).throw(OSError("state ref failure")))
    with pytest.raises(OSError, match="state ref failure"):
        obj.save(repo=repo)

    assert not (Path(store.base_dir) / "state-refs").exists()
    assert obj._last_state_hash is not None
    assert Path(store.validate_local_state(obj.definition, obj._last_state_hash)).is_dir()
