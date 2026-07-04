import dryml
import pytest
from dryml.core2.repo import Repo
from dryml.core2.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.records import (
    RepresentationRequirement,
    SpecValidationError,
    StorageRef,
    StoredStateRecord,
    find_compatible_state_record,
    find_stored_state_records,
    make_representation_spec,
)


def _cdef(char="a"):
    return format_cdef_id(char * 64)


def _write_state(store, cdef, spec, *, role="state"):
    store.records.write_spec(spec, family="representation")
    record = StoredStateRecord(cdef, spec["id"], (StorageRef.self_product(role=role),))
    ref = store.records.write_record(record.to_envelope())
    store.records.product_root(ref.record_id, create=True).joinpath("state.txt").write_text("state", encoding="utf-8")
    return ref


def test_find_stored_states_and_exact_match_wins(tmp_path):
    store = DirStore(tmp_path / "store")
    raw = make_representation_spec("fake.raw_state", storage_kinds=("product-dir",))
    normalized = make_representation_spec("fake.normalized_state", storage_kinds=("product-dir",))
    raw_ref = _write_state(store, _cdef(), raw)
    normalized_ref = _write_state(store, _cdef(), normalized)
    repo = Repo(stores=[store])

    found = find_stored_state_records(repo, _cdef())
    assert {item.ref.record_id for item in found} == {raw_ref.record_id, normalized_ref.record_id}
    result = find_compatible_state_record(repo, _cdef(), RepresentationRequirement(representation_id=normalized["id"]))
    assert result.status == "ok"
    assert result.selected.ref.record_id == normalized_ref.record_id


def test_resolution_reports_not_found_and_works_without_index(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=[store])
    missing = find_compatible_state_record(repo, _cdef(), RepresentationRequirement(kind="fake.raw_state"))
    assert missing.status == "not_found"

    spec = make_representation_spec("fake.raw_state", storage_kinds=("product-dir",))
    _write_state(store, _cdef(), spec)
    store.records.rebuild_ref_index()
    store.records.ref_index_path.unlink()
    result = find_compatible_state_record(repo, _cdef(), RepresentationRequirement(kind="fake.raw_state"))
    assert result.status == "ok"


def test_resolution_reports_missing_representation_spec(tmp_path):
    store = DirStore(tmp_path / "store")
    spec = make_representation_spec("fake.raw_state", storage_kinds=("product-dir",))
    record = StoredStateRecord(_cdef(), spec["id"], (StorageRef.self_product(role="state"),))
    ref = store.records.write_record(record.to_envelope())
    repo = Repo(stores=[store])

    result = find_compatible_state_record(repo, _cdef(), RepresentationRequirement(kind="fake.raw_state"))

    assert result.status == "failed"
    assert result.report.issues[0].code == "missing_representation_spec"
    assert result.report.issues[0].record_id == ref.record_id
    assert result.report.issues[0].representation_id == spec["id"]


def test_representation_requirement_rejects_string_sequences():
    for field in ("required_traits", "storage_kinds"):
        with pytest.raises(SpecValidationError):
            RepresentationRequirement.from_json({field: "gpu"})


def test_resolution_reporting_details(tmp_path):
    store = DirStore(tmp_path / "store")
    spec = make_representation_spec("fake.raw_state", storage_kinds=("product-dir",))
    _write_state(store, _cdef(), spec)
    repo = Repo(stores=[store])
    capture = dryml.reporting.CaptureReporter()

    with dryml.config(reporting={"level": "details", "reporter": capture}):
        find_compatible_state_record(repo, _cdef(), RepresentationRequirement(kind="fake.raw_state"))

    assert "dryml.records.representation.check" in {event.name for event in capture.events}
