import pytest

from dryml.core2.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.records import (
    ProductManifest,
    ProductManifestEntry,
    ProductWriteSession,
    RecordExportError,
    RecordPolicyOptions,
    RecordValidationError,
    StorageRef,
    StoredStateRecord,
    copy_record_closure,
    default_object_state_representation_spec,
    validate_product_availability,
)


def _cdef():
    return format_cdef_id("a" * 64)


def _record(manifest):
    return StoredStateRecord(
        subject_cdef_id=_cdef(),
        representation_id=default_object_state_representation_spec()["id"],
        storage=(StorageRef.self_product(role="target-state"),),
        manifest=manifest.to_json(),
    ).to_envelope()


def test_product_write_session_commits_under_record_id(tmp_path):
    store = DirStore(tmp_path / "store")
    with ProductWriteSession(store.records) as session:
        session.write_text("b.txt", "b")
        session.write_text("a/data.txt", "a")
        manifest = session.manifest()
        result = session.commit_record(_record(manifest))

    assert [entry.path for entry in result.manifest.entries] == ["a/data.txt", "b.txt"]
    assert (store.records.products_dir / result.located.record_id / "a" / "data.txt").read_text(encoding="utf-8") == "a"
    assert not any(path.name.startswith(".staging-") for path in store.records.products_dir.iterdir())


def test_manifest_digest_changes_and_failed_validation_cleans_staging(tmp_path):
    store = DirStore(tmp_path / "store")
    with ProductWriteSession(store.records) as session:
        session.write_text("x.txt", "one")
        one = session.manifest().entries[0].sha256
        session.write_text("x.txt", "two")
        two = session.manifest().entries[0].sha256
        assert one != two
        bad = _record(session.manifest())
        bad["unexpected"] = True
        with pytest.raises(RecordValidationError):
            session.commit_record(bad)

    assert not any(path.name.startswith(".staging-") for path in store.records.products_dir.iterdir())


def test_validate_product_availability_reports_missing_product_root(tmp_path):
    store = DirStore(tmp_path / "store")
    manifest = ProductManifest((ProductManifestEntry("state.txt", 5, "0" * 64),))
    ref = store.records.write_record(_record(manifest))

    issues = validate_product_availability(store.records, store.records.read_record(ref.record_id))

    assert {issue.code for issue in issues} == {"missing_product_path", "missing_manifest_entry"}
    assert all(issue.record_id == ref.record_id for issue in issues)


def test_product_manifest_rejects_string_entries():
    with pytest.raises(RecordValidationError):
        ProductManifest.from_json({"entries": "state.txt"})


def test_copy_record_closure_rejects_dangling_product_records_before_sidecars(tmp_path):
    source = DirStore(tmp_path / "source")
    dest = DirStore(tmp_path / "dest")
    manifest = ProductManifest((ProductManifestEntry("state.txt", 5, "0" * 64),))
    source.records.write_spec(default_object_state_representation_spec(), family="representation")
    ref = source.records.write_record(_record(manifest))

    with pytest.raises(RecordExportError):
        copy_record_closure(
            source,
            dest,
            seed_records=(ref.record_id,),
            policy="closure",
            options=RecordPolicyOptions(include_products=True),
        )

    assert not dest.records.has_record(ref.record_id)
