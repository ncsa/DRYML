import pytest

from dryml.formats.refs import format_cdef_id, format_ref_cdef
from dryml.records import RecordValidationError, SpecValidationError, attach_record_id, make_record, scan_json_refs, scan_record_refs


def cdef(char="a"):
    return format_cdef_id(char * 64)


def cid(prefix, char="a"):
    return f"{prefix}-v1-{char * 64}"


def test_scanner_finds_nested_cdef_refs_and_semantics():
    mentions = scan_json_refs({"args": [cdef(), {"x": format_ref_cdef(cdef("b"))}], "kwargs": {"m": cdef("c")}}, source_kind="spec", source_id=cid("op"), source_family="operation")

    by_path = {mention.path: mention for mention in mentions}
    assert by_path["/payload/args/0"].cdef_semantics == "materialize"
    assert by_path["/payload/args/1/x"].cdef_semantics == "reference"
    assert by_path["/payload/kwargs/m"].target_id == cdef("c")


def test_scanner_ignores_literal_escape_and_rejects_malformed_refs():
    mentions = scan_json_refs({"x": {"$literal": cdef()}, "y": "ordinary"}, source_kind="record", source_id=cid("record"))
    assert mentions == ()

    with pytest.raises(RecordValidationError):
        scan_json_refs({"x": {"$literal": cdef(), "extra": True}}, source_kind="record", source_id=cid("record"))
    with pytest.raises(RecordValidationError):
        scan_json_refs({"x": "op-v1-short"}, source_kind="record", source_id=cid("record"))


def test_scan_json_refs_validates_source_even_without_mentions():
    with pytest.raises(RecordValidationError):
        scan_json_refs({}, source_kind="record", source_id="not-a-record-id")
    with pytest.raises(RecordValidationError):
        scan_json_refs({}, source_kind="record", source_id="record-v2-" + "a" * 64)
    with pytest.raises(RecordValidationError):
        scan_json_refs({}, source_kind="record", source_id=cid("record"), source_family="operation")
    with pytest.raises(RecordValidationError):
        scan_json_refs({}, source_kind="unknown", source_id=cid("record"))
    with pytest.raises(SpecValidationError):
        scan_json_refs({}, source_kind="spec", source_id=cid("repr"), source_family="operation")


def test_typed_keys_are_validated_and_paths_are_escaped():
    record = attach_record_id(
        make_record(
            kind="stored_state",
            payload={
                "subject_cdef_id": cdef(),
                "owner_cdef_id": cdef("b"),
                "input_cdef_ids": [cdef("c")],
                "output_cdef_ids": [cdef("d")],
                "operation_id": cid("op"),
                "representation_id": cid("repr"),
                "environment_requirement_id": cid("envreq"),
                "world_requirement_id": cid("worldreq"),
                "world_id": cid("world"),
                "runtime_id": cid("runtime"),
                "derived_from": [cid("record", "b")],
                "a/b~c": cdef("e"),
            },
        )
    )

    mentions = scan_record_refs(record)
    by_path = {mention.path: mention for mention in mentions}
    assert by_path["/payload/subject_cdef_id"].typed_role == "subject"
    assert by_path["/payload/owner_cdef_id"].typed_role == "owner"
    assert by_path["/payload/input_cdef_ids/0"].typed_role == "input"
    assert by_path["/payload/output_cdef_ids/0"].typed_role == "output"
    assert by_path["/payload/operation_id"].prefix == "op"
    assert by_path["/payload/representation_id"].prefix == "repr"
    assert by_path["/payload/environment_requirement_id"].prefix == "envreq"
    assert by_path["/payload/world_requirement_id"].prefix == "worldreq"
    assert by_path["/payload/runtime_id"].prefix == "runtime"
    assert by_path["/payload/derived_from/0"].typed_role == "derived_from"
    assert by_path["/payload/a~1b~0c"].target_id == cdef("e")
    assert list(mentions) == sorted(mentions, key=lambda m: (m.source_kind, m.source_family or "", m.source_id, m.path, m.target_kind, m.target_id, m.ref_kind, m.typed_key or "", m.typed_role or ""))


def test_typed_key_shape_and_prefix_mismatches_are_rejected():
    with pytest.raises(RecordValidationError):
        scan_json_refs({"input_cdef_ids": cdef()}, source_kind="record", source_id=cid("record"))
    with pytest.raises(RecordValidationError):
        scan_json_refs({"operation_id": cid("repr")}, source_kind="record", source_id=cid("record"))
    with pytest.raises(RecordValidationError):
        scan_json_refs({"operation_id": "op-v2-" + "a" * 64}, source_kind="record", source_id=cid("record"))
