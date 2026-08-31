"""ZipStore contracts migrated from retired object-root archive authority.

Retired object roots and generation/current pointers map to immutable
``DefinitionRecord``, ``LocalStateManifest``, and ``StateRefRecord`` authority.
Retired root aliases map to direct ``MainRefRecord``, ``ObjectAliasRecord``, and
``StateAliasRecord`` references. Retired query-root scans map to derivation from
``DefinitionRecord`` records. Retired archive-wide pickle globals map to the
``StoreFormatRecord`` format gate. ``ZipExportStore`` is retired: ``ZipStore``
is the only public archive Store and supplies its buffered path-backed
transaction.
"""

import hashlib
from io import BytesIO
import multiprocessing
import os
from pathlib import Path
import zipfile

import pytest

import dryml.core.store as store_exports
import dryml.core.store.zip as zip_module
from dryml.core import Object, Repo, Serializable, StateRef
from dryml.core.store.records import (
    DeclarationRecord, DefinitionRecord, LocalStateManifest, MainRefRecord,
    ObjectAliasRecord, StateAliasRecord, StateRefRecord,
)
from dryml.core.store.store import StoreAuthorityError, StoreCapabilityError
from dryml.core.store.zip import ZipStore, ZipStoreConflictError


class ZipRecordObject(Object):
    def __init__(self, value=""):
        super().__init__()
        self.value = value


class ZipPayloadObject(Serializable):
    def __init__(self, value="payload"):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "payload.txt").write_text(self.value)


def _archive_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _record(value="record") -> DefinitionRecord:
    return DefinitionRecord(ZipRecordObject(value).definition)


def _stage_local_state(store, record: DefinitionRecord, payload: bytes) -> LocalStateManifest:
    stage = Path(store.create_local_state_staging())
    data = stage / "data"
    (data / "payload.bin").write_bytes(payload)
    definition_bytes = record.to_bytes()
    manifest = LocalStateManifest(
        "pkl", record.graph_hash, record.digest,
        hashlib.sha256(definition_bytes).hexdigest(),
        (("payload.bin", len(payload), hashlib.sha256(payload).hexdigest()),),
    )
    (stage / "def.pkl").write_bytes(definition_bytes)
    (stage / "manifest.record").write_bytes(manifest.to_bytes())
    store.install_local_state(stage, manifest)
    return manifest


def _concurrent_direct_record_commit(path: str, value: str, ready, start, results) -> None:
    """Stage one direct DefinitionRecord and report atomic archive publication."""

    store = ZipStore(path)
    try:
        record = _record(value)
        ready.set()
        if not start.wait(timeout=30):
            results.put(("error", "start timeout"))
            return
        store.write_definition_record(record)
        store.commit()
        results.put(("published", record.digest))
    except ZipStoreConflictError:
        results.put(("conflict", None))
    except BaseException as error:
        results.put(("error", repr(error)))
    finally:
        store.close()


def test_zipstore_round_trips_current_logical_definition_authority(tmp_path):
    path = tmp_path / "store.zip"
    record = _record()
    store = ZipStore(path)
    store.write_definition_record(record)
    store.commit()
    store.close()

    reopened = ZipStore(path)
    try:
        assert reopened.read_definition_record(record.digest) == record
        assert tuple(reopened.iter_definition_records()) == (record,)
    finally:
        reopened.close()


def test_path_backed_zip_buffers_direct_record_mutation_until_commit(tmp_path):
    path = tmp_path / "store.zip"
    record = _record()
    store = ZipStore(path)
    try:
        store.write_definition_record(record)

        assert not path.exists()
        assert store.read_definition_record(record.digest) == record

        store.commit()
        assert path.is_file()
    finally:
        store.close()

    reopened = ZipStore(path)
    try:
        assert reopened.read_definition_record(record.digest) == record
    finally:
        reopened.close()


def test_no_op_zip_commit_hydrates_without_rewriting_archive_bytes(tmp_path):
    path = tmp_path / "store.zip"
    writer = ZipStore(path)
    writer.write_definition_record(_record())
    writer.commit()
    writer.close()
    before = _archive_bytes(path)

    reader = ZipStore(path)
    try:
        assert tuple(reader.hydrate_index())
        reader.commit()
    finally:
        reader.close()

    assert _archive_bytes(path) == before


def test_no_op_stale_zip_commit_preserves_newer_archive_bytes(tmp_path):
    path = tmp_path / "store.zip"
    initial = ZipStore(path)
    initial.write_definition_record(_record("initial"))
    initial.commit()
    initial.close()
    reader = ZipStore(path)
    writer = ZipStore(path)
    writer.write_definition_record(_record("newer"))
    writer.commit()
    after_writer = _archive_bytes(path)

    try:
        reader.commit()
        assert _archive_bytes(path) == after_writer
    finally:
        reader.close()
        writer.close()


def test_zip_commit_builds_valid_complete_sibling_before_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path / "store.zip"
    record = _record()
    store = ZipStore(path)
    validated = []
    replacements = []
    original_testzip = zipfile.ZipFile.testzip
    original_replace = os.replace

    def observe_validation(archive):
        validated.append(Path(archive.filename))
        return original_testzip(archive)

    def inspect_replace(source, destination):
        if Path(destination) == path:
            staged = Path(source)
            assert staged.parent == path.parent
            assert staged != path
            with zipfile.ZipFile(staged) as archive:
                assert archive.testzip() is None
                assert set(archive.namelist()) == {
                    "store-format.record",
                    f"definitions/{record.digest[:2]}/{record.digest}.record",
                }
            replacements.append(staged)
        return original_replace(source, destination)

    monkeypatch.setattr(zipfile.ZipFile, "testzip", observe_validation)
    monkeypatch.setattr(os, "replace", inspect_replace)
    try:
        store.write_definition_record(record)
        store.commit()
    finally:
        store.close()

    assert len(validated) == 2  # The explicit validation and replacement inspection.
    assert replacements and replacements[0].suffix == ".zip"


def test_failed_zip_commit_keeps_previous_complete_archive_bytes(tmp_path, monkeypatch):
    path = tmp_path / "store.zip"
    store = ZipStore(path)
    first = _record("first")
    store.write_definition_record(first)
    store.commit()
    before = _archive_bytes(path)
    original_replace = os.replace

    def fail_archive_replace(source, destination):
        if Path(destination) == path:
            raise OSError("injected archive replacement failure")
        return original_replace(source, destination)

    store.write_definition_record(_record("second"))
    monkeypatch.setattr(os, "replace", fail_archive_replace)
    try:
        with pytest.raises(OSError, match="archive replacement"):
            store.commit()
        assert _archive_bytes(path) == before
        assert store._archive_dirty
    finally:
        store.close()

    reopened = ZipStore(path)
    try:
        assert tuple(reopened.iter_definition_records()) == (first,)
    finally:
        reopened.close()


def test_stale_zip_writer_cannot_replace_newer_direct_definition_authority(tmp_path):
    path = tmp_path / "store.zip"
    first = ZipStore(path)
    stale = ZipStore(path)
    first_record = _record("first")
    stale_record = _record("stale")
    first.write_definition_record(first_record)
    first.commit()
    before = _archive_bytes(path)
    stale.write_definition_record(stale_record)

    with pytest.raises(ZipStoreConflictError, match="reopen"):
        stale.commit()
    assert _archive_bytes(path) == before

    first.close()
    stale.close()

    retry = ZipStore(path)
    try:
        retry.write_definition_record(stale_record)
        retry.commit()
        assert set(retry.iter_definition_records()) == {first_record, stale_record}
    finally:
        retry.close()


def test_stale_zip_writer_cannot_replace_newer_direct_reference_authority(tmp_path):
    path = tmp_path / "store.zip"
    target = ZipPayloadObject("target").object_ref
    first = ZipStore(path)
    stale = ZipStore(path)
    first_ref = ObjectAliasRecord("first", target)
    stale_ref = ObjectAliasRecord("stale", target)
    first.write_object_alias(first_ref)
    first.commit()
    before = _archive_bytes(path)
    stale.write_object_alias(stale_ref)

    with pytest.raises(ZipStoreConflictError, match="reopen"):
        stale.commit()
    assert _archive_bytes(path) == before

    first.close()
    stale.close()
    retry = ZipStore(path)
    try:
        assert retry.read_object_alias("first") == first_ref
        assert retry.read_object_alias("stale") is None
    finally:
        retry.close()


def test_path_backed_zip_serializes_direct_record_publication_across_processes(tmp_path):
    path = tmp_path / "store.zip"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    ready = [context.Event(), context.Event()]
    processes = [
        context.Process(
            target=_concurrent_direct_record_commit,
            args=(str(path), value, ready[index], start, results),
        )
        for index, value in enumerate(("first", "second"))
    ]
    for process in processes:
        process.start()
    try:
        for event in ready:
            assert event.wait(timeout=30)
        start.set()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        outcomes = [results.get(timeout=1) for _ in processes]
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert sorted(outcome[0] for outcome in outcomes) == ["conflict", "published"]
    published_digest = next(digest for status, digest in outcomes if status == "published")
    reopened = ZipStore(path)
    try:
        assert tuple(record.digest for record in reopened.iter_definition_records()) == (published_digest,)
    finally:
        reopened.close()


def test_malformed_or_escaping_zip_members_fail_closed_without_rewriting_bytes(tmp_path):
    malformed = tmp_path / "malformed.zip"
    malformed.write_bytes(b"not a zip archive")
    malformed_before = _archive_bytes(malformed)
    with pytest.raises(StoreAuthorityError, match="malformed"):
        ZipStore(malformed)
    assert _archive_bytes(malformed) == malformed_before

    escaping = tmp_path / "escaping.zip"
    with zipfile.ZipFile(escaping, "w") as archive:
        archive.writestr("../outside.record", b"outside")
        archive.writestr("..\\windows-outside.record", b"outside")
    escaping_before = _archive_bytes(escaping)
    with pytest.raises(StoreAuthorityError, match="escapes"):
        ZipStore(escaping)
    assert _archive_bytes(escaping) == escaping_before
    assert not (tmp_path / "outside.record").exists()
    assert not (tmp_path / "windows-outside.record").exists()


def test_retired_object_root_archive_fails_at_the_direct_format_gate_without_rewrite(tmp_path):
    path = tmp_path / "retired.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("objects/aa/legacy/def.pkl", b"retired root authority")
    before = _archive_bytes(path)

    with pytest.raises(StoreAuthorityError, match="lacks current store-format"):
        ZipStore(path)

    assert _archive_bytes(path) == before


def test_record_and_local_state_identity_do_not_depend_on_archive_staging_paths(tmp_path):
    obj = ZipPayloadObject("same")
    record = DefinitionRecord(obj.definition)
    state = StateRef(obj.object_ref, {next(iter(obj.object_ref.objects)): "pkl-" + "a" * 64})
    manifests = []
    archives = []
    for name in ("first", "second"):
        archive = tmp_path / f"{name}.zip"
        store = ZipStore(archive)
        manifest = _stage_local_state(store, record, b"same payload")
        store.write_definition_record(record)
        store.write_state_ref_record(StateRefRecord(state))
        store.commit()
        store.close()
        manifests.append(manifest)
        archives.append(archive)

    assert manifests[0] == manifests[1]
    assert manifests[0].state_hash == manifests[1].state_hash
    for archive in archives:
        reopened = ZipStore(archive)
        try:
            assert reopened.read_definition_record(record.digest) == record
            assert reopened.read_state_ref_record(state.digest()) == StateRefRecord(state)
            assert Path(reopened.open_local_state(record.graph_hash, manifests[0].state_hash), "data", "payload.bin").read_bytes() == b"same payload"
        finally:
            reopened.close()


def test_direct_records_local_state_and_references_round_trip_through_zip(tmp_path):
    path = tmp_path / "store.zip"
    store = ZipStore(path)
    obj = ZipPayloadObject("state")
    record = DefinitionRecord(obj.definition)
    manifest = _stage_local_state(store, record, b"state payload")
    state = StateRef(obj.object_ref, {next(iter(obj.object_ref.objects)): manifest.state_hash})
    declaration = DeclarationRecord(state.object)
    main = MainRefRecord(record.digest)
    object_alias = ObjectAliasRecord("current", state.object)
    state_alias = StateAliasRecord("current", state.object, state.digest())
    store.write_definition_record(record)
    store.write_state_ref_record(StateRefRecord(state))
    store.write_declaration_record(declaration)
    store.write_main_ref(main)
    store.write_object_alias(object_alias)
    store.write_state_alias(state_alias)
    store.commit()
    store.close()

    reopened = ZipStore(path)
    try:
        assert reopened.read_definition_record(record.digest) == record
        assert reopened.read_state_ref_record(state.digest()) == StateRefRecord(state)
        assert reopened.read_declaration_record(declaration.digest) == declaration
        assert reopened.read_main_ref() == main
        assert reopened.read_object_alias("current") == object_alias
        assert reopened.read_state_alias(state.object.digest(), "current") == state_alias
        assert Path(reopened.validate_local_state(record.definition, manifest.state_hash), "data", "payload.bin").read_bytes() == b"state payload"
    finally:
        reopened.close()


def test_zip_hydration_derives_the_memory_query_index_from_definition_records(tmp_path):
    path = tmp_path / "store.zip"
    records = (_record("first"), _record("second"))
    store = ZipStore(path)
    for record in records:
        store.write_definition_record(record)
    store.commit()
    store.close()

    reopened = ZipStore(path)
    try:
        assert tuple(reopened.hydrate_index()) == tuple(record.definition for record in sorted(records, key=lambda item: item.digest))
        assert reopened.open_query_index() is None
        assert reopened.query_index_status().backend == "memory"
        assert not any(name.startswith(".dryml/") for name in zipfile.ZipFile(path).namelist())
    finally:
        reopened.close()


def test_zip_close_discards_uncommitted_buffer_and_cleans_extraction(tmp_path):
    path = tmp_path / "store.zip"
    store = ZipStore(path)
    extracted = Path(store.base_dir)
    store.write_definition_record(_record())

    store.close()
    store.close()

    assert not extracted.exists()
    assert not path.exists()


def test_zip_export_store_is_retired_from_the_public_archive_surface():
    assert "ZipExportStore" not in store_exports.__all__
    assert not hasattr(store_exports, "ZipExportStore")
    assert not hasattr(zip_module, "ZipExportStore")


def test_file_like_zip_is_explicitly_read_only_for_current_authority():
    store = ZipStore(BytesIO())
    try:
        with pytest.raises(StoreCapabilityError, match="writable"):
            store.preflight_publication("write definition")
    finally:
        store.close()


def test_file_like_zip_hydrates_current_authority_without_permitting_mutation(tmp_path):
    path = tmp_path / "store.zip"
    record = DefinitionRecord(ZipRecordObject("read-only").definition)
    writer = ZipStore(path)
    writer.write_definition_record(record)
    writer.commit()
    writer.close()

    archive_bytes = path.read_bytes()
    file_like = BytesIO(archive_bytes)
    reader = ZipStore(file_like)
    try:
        assert reader.read_definition_record(record.digest) == record
        assert not reader.publication_capabilities.writable
        with pytest.raises(StoreCapabilityError, match="writable"):
            reader.write_definition_record(record)
    finally:
        reader.close()
    assert file_like.getvalue() == archive_bytes


def test_stale_zip_save_cannot_replace_a_newer_exact_state_ref(tmp_path):
    path = tmp_path / "state.zip"
    first = ZipStore(path)
    stale = ZipStore(path)
    first_state = ZipPayloadObject("first", repo=Repo(first)).save(repo=Repo(first))
    # Use separately realized state on the stale archive transaction so its
    # commit must compare against the archive identity recorded at open.
    stale_state = ZipPayloadObject("stale", repo=Repo(stale)).save(repo=Repo(stale))
    first.commit()

    with pytest.raises(ZipStoreConflictError, match="reopen"):
        stale.commit()

    reopened = ZipStore(path)
    try:
        assert reopened.read_state_ref_record(first_state.digest()).state_ref == first_state
        assert reopened.read_state_ref_record(stale_state.digest()) is None
    finally:
        first.close()
        stale.close()
        reopened.close()
