import dryml

from dryml.core.repo import Repo
from dryml.core.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.records import RepresentationRequirement, StorageRef, StoredStateRecord, find_compatible_state_record, make_representation_spec


def _seed(tmp_path):
    store = DirStore(tmp_path / "store")
    spec = make_representation_spec("fake.raw_state", storage_kinds=("product-dir",))
    store.records.write_spec(spec, family="representation")
    cdef = format_cdef_id("a" * 64)
    ref = store.records.write_record(StoredStateRecord(cdef, spec["id"], (StorageRef.self_product(),)).to_envelope())
    store.records.product_root(ref.record_id, create=True)
    return Repo(stores=[store]), cdef


def test_quiet_mode_suppresses_record_events(tmp_path):
    repo, cdef = _seed(tmp_path)
    capture = dryml.reporting.CaptureReporter()
    with dryml.config(reporting={"level": "quiet", "reporter": capture}):
        find_compatible_state_record(repo, cdef, RepresentationRequirement(kind="fake.raw_state"))
    assert capture.events == []


def test_details_mode_includes_record_resolution_events(tmp_path):
    repo, cdef = _seed(tmp_path)
    capture = dryml.reporting.CaptureReporter()
    with dryml.config(reporting={"level": "details", "reporter": capture}):
        find_compatible_state_record(repo, cdef, RepresentationRequirement(kind="fake.raw_state"))
    assert "dryml.records.state.find" in {event.name for event in capture.events}
    assert any(event.record_id for event in capture.events)
