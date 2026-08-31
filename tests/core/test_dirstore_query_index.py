"""DefinitionRecord authority and derived DirStore query-index integration tests."""

import os
from pathlib import Path
import threading

import dill
import pytest

from dryml.core import Definition, Object, Repo, SKIP_ARGS
from dryml.core.query.sqlite import SQLiteQueryIndexConfig, require_sqlite, sqlite_available
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord, StoreFormatRecord
from dryml.core.store.store import StoreAuthorityError


class IndexedRecordObject(Object):
    def __init__(self, value="value"):
        self.value = value


def _record(value="value"):
    return DefinitionRecord(IndexedRecordObject(value).definition)


def _definition_path(store, record):
    return Path(store.base_dir, "definitions", record.digest[:2], f"{record.digest}.record")


def _framed_definition_data(data, *, version=1):
    return b"DRYML-STORE-RECORD/definition/" + str(version).encode("ascii") + b"\n" + dill.dumps(data, protocol=5)


def test_dirstore_query_index_default_is_auto_and_lazy(tmp_path):
    store = DirStore(tmp_path / "store")

    assert store.query_index_policy == "auto"
    index = store.open_query_index()
    if sqlite_available():
        assert index is not None
        assert store.query_index_status().state == "missing"
    else:
        assert index is None
        assert store.query_index_status().state == "ready"
    assert not Path(store.base_dir, ".dryml").exists()


@pytest.mark.parametrize("policy", ["auto", "memory", "none", "sqlite"])
def test_every_current_query_index_policy_leaves_direct_authority_unchanged(tmp_path, policy):
    store = DirStore(tmp_path / policy, query_index=policy)
    record = _record(policy)

    store.write_definition_record(record)

    if policy in {"auto", "sqlite"} and sqlite_available():
        assert store.open_query_index() is not None
        assert store.query_index_is_dirty()
    else:
        assert store.open_query_index() is None
    assert tuple(store.hydrate_index()) == (record.definition,)
    assert store.read_definition_record(record.digest) == record
    assert not Path(store.base_dir, "objects").exists()


def test_dirstore_rejects_unknown_or_factory_query_index_policy(tmp_path):
    with pytest.raises(ValueError, match="query_index"):
        DirStore(tmp_path / "unknown", query_index="bad-policy")
    with pytest.raises(ValueError, match="query_index"):
        DirStore(tmp_path / "factory", query_index=lambda store: store)


def test_dirstore_accepts_sqlite_config_without_creating_a_sidecar(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    index = store.open_query_index()

    assert index is not None
    assert index.path == Path(store.query_index_path)
    assert not index.path.exists()


def test_dirstore_rejects_non_policy_non_config_query_index(tmp_path):
    with pytest.raises(ValueError, match="query_index"):
        DirStore(tmp_path / "store", query_index=object())
    assert not Path(tmp_path / "store", ".dryml").exists()


def test_query_index_status_reflects_each_current_policy(tmp_path):
    statuses = [
        DirStore(tmp_path / policy, query_index=policy).query_index_status()
        for policy in ("auto", "memory", "none", "sqlite")
    ]

    assert [status.backend for status in statuses] == ["sqlite", "memory", "none", "sqlite"]
    assert [status.state for status in statuses] == ["missing", "ready", "disabled", "missing"]
    assert all(status.generation is None for status in statuses)


def test_rebuild_and_reconcile_build_a_persistent_index_from_direct_records(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")

    rebuilt = store.rebuild_query_index()
    reconciled = store.reconcile_query_index()

    assert rebuilt.action == "rebuild"
    assert rebuilt.definitions_scanned == 0
    assert reconciled.action == "validate"
    assert store.validate_query_index(thorough=True).ok
    assert store.query_index_status().state == "ready"


def test_definition_scan_rebuilds_only_from_direct_authoritative_records(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    record = _record()
    store.write_definition_record(record)
    legacy = Path(store.base_dir, "objects", "deadbeef", "def.pkl")
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"retired layout")

    assert tuple(store.hydrate_index()) == (record.definition,)


def test_definition_scan_is_deterministic_and_marks_persistent_index_dirty(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    records = [_record(value) for value in ("third", "first", "second")]
    for record in records:
        store.write_definition_record(record)

    scanned = tuple(store.iter_definition_records())

    assert [record.digest for record in scanned] == sorted(record.digest for record in records)
    assert store.query_index_status().state == "dirty"
    assert store.query_index_is_dirty()


def test_definition_scan_ignores_directories_and_nonmatching_record_suffixes(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record("valid")
    store.write_definition_record(record)
    root = Path(store.base_dir, "definitions")
    (root / "ignored").mkdir()
    (root / "ignored" / "not-a-record.txt").write_bytes(b"ignored")

    assert tuple(store.iter_definition_records()) == (record,)


def test_definition_scan_rejects_record_under_wrong_digest_path(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    wrong_path = Path(store.base_dir, "definitions", "00", "0" * 64 + ".record")
    wrong_path.parent.mkdir(parents=True)
    wrong_path.write_bytes(record.to_bytes())

    with pytest.raises(StoreAuthorityError, match="invalid digest path"):
        tuple(store.iter_definition_records())


def test_definition_scan_rejects_malformed_direct_definition_without_altering_other_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    good = _record("good")
    bad = _record("bad")
    store.write_definition_record(good)
    store.write_definition_record(bad)
    bad_path = _definition_path(store, bad)
    bad_path.write_bytes(b"not a definition record")

    with pytest.raises(StoreAuthorityError, match="Malformed Store record"):
        tuple(store.hydrate_index())

    assert store.read_definition_record(good.digest) == good
    assert bad_path.read_bytes() == b"not a definition record"


def test_definition_scan_rejects_schema_mismatch_without_sidecar_recovery(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    path = _definition_path(store, record)
    path.parent.mkdir(parents=True)
    path.write_bytes(StoreFormatRecord().to_bytes())

    with pytest.raises(StoreAuthorityError, match="Malformed Store record"):
        tuple(store.iter_definition_records())
    assert not Path(store.base_dir, ".dryml").exists()


def test_definition_scan_rejects_unsupported_definition_record_version(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    path = _definition_path(store, record)
    path.parent.mkdir(parents=True)
    path.write_bytes(_framed_definition_data(record.to_data(), version=2))

    with pytest.raises(StoreAuthorityError, match="Malformed Store record"):
        tuple(store.hydrate_index())


def test_definition_scan_rejects_definition_semantic_hash_mismatch(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    data = record.to_data()
    data["structural_hash"] = "0" * 64
    path = _definition_path(store, record)
    path.parent.mkdir(parents=True)
    path.write_bytes(_framed_definition_data(data))

    with pytest.raises(StoreAuthorityError, match="Malformed Store record"):
        tuple(store.hydrate_index())


def test_store_format_version_mismatch_rejects_open_without_scanning_authority(tmp_path):
    root = tmp_path / "store"
    store = DirStore(root)
    format_path = Path(store.store_format_path)
    format_path.write_bytes(_framed_definition_data(StoreFormatRecord().to_data()))

    with pytest.raises(StoreAuthorityError, match="Malformed Store record"):
        DirStore(root)


def test_reopened_repo_queries_direct_definition_records_after_explicit_hydration(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    first = IndexedRecordObject("first", repo=repo)
    second = IndexedRecordObject("second", repo=repo)
    repo.save_object(first)
    repo.save_object(second)

    reopened = DirStore(store.base_dir, query_index="none")
    reopened_repo = Repo(reopened)
    selector = Definition(IndexedRecordObject, SKIP_ARGS)

    assert list(reopened_repo.query(selector).stored().defs()) == []
    results = reopened_repo.query(selector).stored(refresh=True).defs()
    assert set(results) == {first.definition, second.definition}


def test_direct_hydration_does_not_open_or_repair_retired_sidecars(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    sidecar = Path(store.base_dir, ".dryml", "query-index.sqlite")
    sidecar.parent.mkdir(exist_ok=True)
    sidecar.write_bytes(b"corrupt retired derived state")

    assert tuple(store.hydrate_index()) == (record.definition,)
    assert sidecar.read_bytes() == b"corrupt retired derived state"
    assert store.query_index_status().state == "dirty"


def test_deleting_derived_paths_cannot_remove_logical_definition_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    derived = Path(store.base_dir, ".dryml")
    derived.mkdir(exist_ok=True)
    (derived / "query-index.sqlite").write_bytes(b"derived")
    (derived / "query-index.sqlite").unlink()
    store.clear_query_index_dirty()
    derived.rmdir()

    assert store.read_definition_record(record.digest) == record
    assert tuple(store.hydrate_index()) == (record.definition,)


def test_deleting_retired_build_claim_cannot_change_direct_definition_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    claim = Path(store.base_dir, ".dryml", "query-index.sqlite.building")
    claim.parent.mkdir(exist_ok=True)
    claim.write_bytes(b"interrupted build")
    claim.unlink()

    assert tuple(store.iter_definition_records()) == (record,)
    assert store.query_index_status().state == "dirty"


def test_replacing_or_deleting_retired_generation_roots_is_not_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    generation = Path(store.base_dir, "objects", "aa", "old", ".state-generations", "1")
    generation.mkdir(parents=True)
    (generation / "def.pkl").write_bytes(b"retired generation")
    (generation / "def.pkl").unlink()
    generation.rmdir()

    assert store.read_definition_record(record.digest) == record
    assert tuple(store.hydrate_index()) == (record.definition,)


def test_direct_definition_record_is_idempotent_after_full_validation(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()

    assert store.write_definition_record(record) == record
    assert store.write_definition_record(record) == record
    assert tuple(store.iter_definition_records()) == (record,)


def test_interrupted_definition_replacement_preserves_existing_authority(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    existing = _record("existing")
    interrupted = _record("interrupted")
    store.write_definition_record(existing)
    target = _definition_path(store, interrupted)
    original_replace = os.replace

    def fail_install(source, destination):
        if Path(destination) == target:
            raise OSError("injected direct-record replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_install)
    with pytest.raises(OSError, match="replacement failure"):
        store.write_definition_record(interrupted)

    assert store.read_definition_record(existing.digest) == existing
    assert store.read_definition_record(interrupted.digest) is None
    assert not list(target.parent.glob(".store-*"))


def test_interrupted_record_write_does_not_create_or_dirty_a_sidecar(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    record = _record()
    original_replace = os.replace

    def interrupt_install(source, destination):
        if Path(destination) == _definition_path(store, record):
            raise KeyboardInterrupt("injected interruption")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt_install)
    with pytest.raises(KeyboardInterrupt, match="interruption"):
        store.write_definition_record(record)

    assert store.read_definition_record(record.digest) is None
    assert store.query_index_status().state == "missing"
    assert not list(Path(store.base_dir, ".dryml").glob("query-index.dirty.*"))


def test_concurrent_immutable_definition_installs_preserve_every_complete_record(tmp_path):
    store = DirStore(tmp_path / "store")
    records = [_record(str(index)) for index in range(8)]
    barrier = threading.Barrier(len(records))
    failures = []

    def install(record):
        try:
            barrier.wait()
            store.write_definition_record(record)
        except BaseException as error:
            failures.append(error)

    workers = [threading.Thread(target=install, args=(record,)) for record in records]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert failures == []
    assert {record.digest for record in store.iter_definition_records()} == {record.digest for record in records}


def test_concurrent_duplicate_installs_are_idempotent(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    barrier = threading.Barrier(4)
    results = []
    failures = []

    def install():
        try:
            barrier.wait()
            results.append(store.write_definition_record(record))
        except BaseException as error:
            failures.append(error)

    workers = [threading.Thread(target=install) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert failures == []
    assert results == [record] * 4
    assert tuple(store.iter_definition_records()) == (record,)


def test_concurrent_reader_observes_only_absent_or_complete_direct_record(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    record = _record()
    target = _definition_path(store, record)
    entered_replace = threading.Barrier(2)
    allow_replace = threading.Event()
    original_replace = os.replace
    observations = []

    def pause_install(source, destination):
        if Path(destination) == target:
            entered_replace.wait()
            allow_replace.wait()
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", pause_install)

    writer = threading.Thread(target=lambda: store.write_definition_record(record))

    def reader():
        entered_replace.wait()
        observations.append(store.read_definition_record(record.digest))
        allow_replace.set()

    reader_thread = threading.Thread(target=reader)
    writer.start()
    reader_thread.start()
    writer.join()
    reader_thread.join()

    assert observations == [None]
    assert store.read_definition_record(record.digest) == record


def test_multiple_handles_share_direct_definition_authority_without_catalog_generation(tmp_path):
    writer = DirStore(tmp_path / "store")
    reader = DirStore(writer.base_dir, query_index="sqlite")
    record = _record()

    writer.write_definition_record(record)

    assert reader.read_definition_record(record.digest) == record
    assert tuple(reader.hydrate_index()) == (record.definition,)
    assert writer.catalog_key() == reader.catalog_key()
    assert writer.query_index_status().generation is None


def test_catalog_key_is_stable_per_store_and_does_not_create_catalog_state(tmp_path):
    first = DirStore(tmp_path / "first")
    first_reopened = DirStore(first.base_dir)
    second = DirStore(tmp_path / "second")

    assert first.catalog_key() == first_reopened.catalog_key()
    assert first.catalog_key() != second.catalog_key()
    assert not list(Path(first.base_dir).rglob("*catalog*"))


def test_direct_hydration_is_read_only_inspection(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    original_atomic_write = store._atomic_write

    def reject_mutation(*args, **kwargs):
        raise AssertionError("inspection must not write Store authority")

    monkeypatch.setattr(store, "_atomic_write", reject_mutation)
    assert tuple(store.iter_definition_records()) == (record,)
    assert tuple(store.hydrate_index()) == (record.definition,)
    monkeypatch.setattr(store, "_atomic_write", original_atomic_write)


def test_direct_hydration_after_external_definition_install_is_recoverable(tmp_path):
    first = DirStore(tmp_path / "store")
    second = DirStore(first.base_dir)
    record = _record("external")

    second.write_definition_record(record)

    assert tuple(first.hydrate_index()) == (record.definition,)
    assert first.query_index_status().state == "dirty"


def test_deleted_direct_definition_is_not_resurrected_from_retired_derived_state(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    sidecar = Path(store.base_dir, ".dryml", "query-index.sqlite")
    sidecar.parent.mkdir(exist_ok=True)
    sidecar.write_bytes(record.to_bytes())
    _definition_path(store, record).unlink()

    assert store.read_definition_record(record.digest) is None
    with pytest.raises(StoreAuthorityError, match="StoredRootRecord"):
        tuple(store.hydrate_index())
    assert sidecar.read_bytes() == record.to_bytes()


def test_malformed_retired_sidecar_cannot_block_direct_authority_recovery(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    sidecar = Path(store.base_dir, ".dryml", "query-index.sqlite")
    sidecar.parent.mkdir(exist_ok=True)
    sidecar.write_bytes(b"not sqlite")

    reopened = DirStore(store.base_dir, query_index="sqlite")
    assert tuple(reopened.hydrate_index()) == (record.definition,)
    assert reopened.query_index_status().state == "dirty"


def test_malformed_retired_dirty_marker_cannot_change_direct_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    marker = Path(store.base_dir, ".dryml", "query-index.dirty")
    marker.parent.mkdir(exist_ok=True)
    marker.write_bytes(b"\xff")

    assert tuple(store.iter_definition_records()) == (record,)
    assert marker.read_bytes() == b"\xff"
    assert store.query_index_status().state == "dirty"


@pytest.mark.parametrize("payload", [b"", b"\xff", b"garbage\n"])
def test_retired_sidecar_payloads_are_never_interpreted_as_definition_authority(tmp_path, payload):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    sidecar = Path(store.base_dir, ".dryml", "query-index.sqlite")
    sidecar.parent.mkdir(exist_ok=True)
    sidecar.write_bytes(payload)

    assert tuple(store.hydrate_index()) == (record.definition,)
    assert sidecar.read_bytes() == payload


def test_definition_authority_is_protected_from_legacy_object_file_changes(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    legacy_def = Path(store.base_dir, "objects", record.structural_hash[:2], record.structural_hash, "def.pkl")
    legacy_def.parent.mkdir(parents=True)
    legacy_def.write_bytes(b"old definition")
    before = _definition_path(store, record).read_bytes()
    legacy_def.write_bytes(b"changed old definition")

    assert _definition_path(store, record).read_bytes() == before
    assert tuple(store.hydrate_index()) == (record.definition,)


def test_direct_definition_path_uses_record_digest_not_structural_hash(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()

    store.write_definition_record(record)

    assert _definition_path(store, record).is_file()
    assert record.digest != record.structural_hash
    assert not Path(store.base_dir, "objects", record.structural_hash[:2], record.structural_hash, "def.pkl").exists()


def test_direct_authority_has_no_generation_roots_or_current_pointers(tmp_path):
    store = DirStore(tmp_path / "store")
    store.write_definition_record(_record())

    paths = [path.name for path in Path(store.base_dir).rglob("*")]

    assert ".state-generations" not in paths
    assert ".state-current.pkl" not in paths
    assert "objects" not in paths


def test_definition_record_digest_is_the_only_read_lookup_key(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)

    assert store.read_definition_record(record.digest) == record
    assert store.read_definition_record(record.structural_hash) is None


def test_direct_authority_survives_repeated_sqlite_index_inspection(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    records = [_record("one"), _record("two")]
    for record in records:
        store.write_definition_record(record)

    index = store.open_query_index()
    assert index is not None
    store.rebuild_query_index()
    for _ in range(3):
        assert store.query_index_status().state == "ready"
        with index.read_view() as view:
            assert all(view.exact_ids(record.definition) for record in records)
        assert tuple(store.hydrate_index()) == tuple(record.definition for record in sorted(records, key=lambda item: item.digest))

    assert not Path(store.base_dir, "objects").exists()


def test_repo_index_admin_rebuilds_derived_sqlite_without_mutating_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    repo = Repo(store)

    statuses = repo.index_status(store=store)
    reports = repo.validate_index(store=store, thorough=True)

    assert statuses[0].backend == "sqlite"
    assert statuses[0].state == "dirty"
    assert not reports[0].ok
    repo.rebuild_index(store=store)
    assert repo.validate_index(store=store, thorough=True)[0].ok
    assert tuple(store.hydrate_index()) == (record.definition,)
    assert Path(store.query_index_path).is_file()


def test_repo_refresh_index_rebuilds_sqlite_from_direct_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)
    repo = Repo(store)
    before = repo.index_status(store=store)[0].generation

    assert repo.rebuild_index(store=store) is repo
    assert repo.index_status(store=store)[0].generation > (before or -1)
    assert repo.refresh_index() is repo
    status = repo.index_status(store=store)[0]
    assert status.backend == "sqlite"
    assert status.generation > (before or -1)
    assert tuple(store.hydrate_index()) == (record.definition,)
    assert Path(store.query_index_path).is_file()


def test_direct_records_from_two_stores_remain_independent(tmp_path):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    first_record = _record("first")
    second_record = _record("second")
    first.write_definition_record(first_record)
    second.write_definition_record(second_record)

    assert tuple(first.hydrate_index()) == (first_record.definition,)
    assert tuple(second.hydrate_index()) == (second_record.definition,)
    assert first.catalog_key() != second.catalog_key()


def test_direct_record_scan_has_no_sqlite_metadata_contract(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    store.write_definition_record(record)

    path = _definition_path(store, record)
    assert path.is_file()
    assert not list(Path(store.base_dir).rglob("*.sqlite"))
    assert tuple(store.hydrate_index()) == (record.definition,)


def test_direct_authority_rejects_nonregular_definition_record(tmp_path):
    store = DirStore(tmp_path / "store")
    record = _record()
    path = _definition_path(store, record)
    path.parent.mkdir(parents=True)
    path.mkdir()

    with pytest.raises(StoreAuthorityError, match="not a regular file"):
        store.read_definition_record(record.digest)


def test_store_format_is_the_only_store_wide_authority_gate(tmp_path):
    store = DirStore(tmp_path / "store")

    assert Path(store.store_format_path).is_file()
    assert StoreFormatRecord.from_bytes(Path(store.store_format_path).read_bytes()) == StoreFormatRecord()
    assert not Path(store.base_dir, ".dryml").exists()


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_sqlite_rebuild_indexes_direct_definition_records_and_versions(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    records = [_record("first"), _record("second")]
    for record in records:
        store.write_definition_record(record)

    report = store.rebuild_query_index()
    status = store.query_index_status()
    sqlite3 = require_sqlite()
    con = sqlite3.connect(store.query_index_path)
    try:
        rows = con.execute("SELECT storage_hash, relative_def_path FROM stored_roots ORDER BY relative_def_path").fetchall()
    finally:
        con.close()

    assert report.definitions_scanned == 2
    assert status.state == "ready"
    assert status.schema_version is not None
    assert status.semantic_versions["query_index_codec_version"] is not None
    assert rows == sorted(
        [(bytes.fromhex(record.digest), f"definitions/{record.digest[:2]}/{record.digest}.record") for record in records],
        key=lambda row: row[1],
    )
    with store.open_query_index().read_view() as view:
        assert all(view.exact_ids(record.definition) for record in records)


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_sqlite_rebuild_scans_records_not_hydrate_index(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    record = _record()
    store.write_definition_record(record)

    monkeypatch.setattr(store, "hydrate_index", lambda: (_ for _ in ()).throw(AssertionError("retired scan")))
    store.rebuild_query_index()

    with store.open_query_index().read_view() as view:
        assert view.exact_ids(record.definition)


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_sqlite_dirty_record_publication_is_visible_to_another_handle_after_rebuild(tmp_path):
    writer = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    first = _record("first")
    writer.write_definition_record(first)
    writer.rebuild_query_index()
    peer = DirStore(writer.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    second = _record("second")

    writer.write_definition_record(second)

    assert peer.query_index_status().state == "dirty"
    report = peer.reconcile_query_index()
    assert report.action == "rebuild"
    with peer.open_query_index().read_view() as view:
        assert view.exact_ids(first.definition)
        assert view.exact_ids(second.definition)
    assert not writer.query_index_is_dirty()


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_deleting_or_corrupting_sidecar_recovers_without_changing_records(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    record = _record()
    store.write_definition_record(record)
    store.rebuild_query_index()
    authority = _definition_path(store, record).read_bytes()
    sidecar = Path(store.query_index_path)

    store.close()
    sidecar.unlink()
    assert store.reconcile_query_index().action == "rebuild"
    store.close()
    sidecar.write_bytes(b"not a sqlite database")
    assert store.query_index_status().state == "corrupt"
    assert store.reconcile_query_index().action == "rebuild"

    assert _definition_path(store, record).read_bytes() == authority
    assert store.validate_query_index(thorough=True).ok
    assert list(sidecar.parent.glob(f"{sidecar.name}.quarantine-*"))


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_interrupted_staged_rebuild_preserves_ready_sidecar_and_record_authority(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    first = _record("first")
    store.write_definition_record(first)
    store.rebuild_query_index()
    index = store.open_query_index()
    before = Path(index.path).read_bytes()
    second = _record("second")
    store.write_definition_record(second)

    def interrupt_validation(self, *, roots):
        raise KeyboardInterrupt("injected staged rebuild interruption")

    monkeypatch.setattr(SQLiteStoreQueryIndex, "_validate_rebuild_before_ready", interrupt_validation)
    with pytest.raises(KeyboardInterrupt, match="staged rebuild interruption"):
        store.rebuild_query_index()

    assert Path(index.path).read_bytes() == before
    assert store.read_definition_record(second.digest) == second
    assert store.query_index_status().state == "dirty"
    assert not list(Path(index.path).parent.glob(f"{index.path.name}.rebuild-*.tmp*"))


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_peer_handle_reopens_after_direct_record_sidecar_replacement(tmp_path):
    writer = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    first = _record("first")
    writer.write_definition_record(first)
    writer.rebuild_query_index()
    peer = DirStore(writer.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    peer_index = peer.open_query_index()
    with peer_index.read_view() as view:
        first_generation = view.generation

    second = _record("second")
    writer.write_definition_record(second)
    writer.rebuild_query_index()

    assert not peer_index._connections._connections
    with peer_index.read_view() as view:
        assert view.generation > first_generation
        assert view.exact_ids(second.definition)


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_sqlite_close_releases_managed_connections(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    store.write_definition_record(_record())
    store.rebuild_query_index()
    index = store.open_query_index()
    index.current_generation()

    store.close()

    assert not index._connections._connections


def test_memory_and_none_policies_never_create_sqlite_sidecars(tmp_path):
    for policy, state in (("memory", "ready"), ("none", "disabled")):
        store = DirStore(tmp_path / policy, query_index=policy)
        store.write_definition_record(_record(policy))

        assert store.open_query_index() is None
        assert store.query_index_status().state == state
        assert not Path(store.query_index_path).exists()
        assert tuple(store.iter_definition_records())
