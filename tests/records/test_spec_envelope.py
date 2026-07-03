import pytest

from dryml.records import SpecValidationError, attach_spec_id, compute_spec_id, make_spec, spec_family_for_id, spec_id_prefix, validate_spec
from dryml.records.kinds import SPEC_FAMILIES


@pytest.mark.parametrize("family", sorted(SPEC_FAMILIES))
def test_make_spec_and_compute_id_for_each_family(family):
    kwargs = {"schema": "example.generic.v1"} if family == "generic" else {}
    spec = attach_spec_id(make_spec(family=family, kind="placeholder", payload={"family": family}, **kwargs), family=family)

    assert spec["id"].startswith(f"{spec_id_prefix(family)}-v1-")
    assert compute_spec_id(spec, family=family) == spec["id"]
    assert spec_family_for_id(spec["id"]) == family
    assert validate_spec(spec, family=family) is spec


def test_spec_metadata_does_not_affect_id_but_payload_does():
    left = attach_spec_id(make_spec(family="representation", kind="torch.state_dict", payload={"a": 1}, metadata={"writer": "a"}))
    right = attach_spec_id(make_spec(family="representation", kind="torch.state_dict", payload={"a": 1}, metadata={"writer": "b"}))
    changed = attach_spec_id(make_spec(family="representation", kind="torch.state_dict", payload={"a": 2}, metadata={"writer": "a"}))

    assert left["id"] == right["id"]
    assert left["id"] != changed["id"]


def test_known_spec_family_prefixes_are_enforced():
    representation = attach_spec_id(make_spec(family="representation", kind="repr", payload={}))

    wrong_schema_for_id = dict(representation, schema="dryml.operation.v1")
    with pytest.raises(SpecValidationError, match="prefix"):
        validate_spec(wrong_schema_for_id, family="operation")

    with pytest.raises(SpecValidationError, match="prefix"):
        make_spec(family="operation", kind="op", payload={}, id=representation["id"])


def test_generic_specs_require_caller_schema():
    with pytest.raises(SpecValidationError, match="schema"):
        make_spec(family="generic", kind="freeform", payload={})

    spec = attach_spec_id(make_spec(family="generic", kind="freeform", schema="example.freeform.v1", payload={}))
    assert spec["schema"] == "example.freeform.v1"
    assert spec["id"].startswith("spec-v1-")


def test_operation_specs_are_placeholders_not_executable_calls():
    spec = attach_spec_id(make_spec(family="operation", kind="placeholder", payload={"note": "not executable"}))

    assert spec["schema"] == "dryml.operation.v1"
    assert spec["payload"] == {"note": "not executable"}
    assert "function_call" not in spec
    assert "method_call" not in spec


def test_spec_rejects_mismatched_existing_id():
    spec = make_spec(family="representation", kind="repr", payload={"x": 1})
    wrong_id = compute_spec_id(make_spec(family="representation", kind="repr", payload={"x": 2}), family="representation")
    wrong = dict(spec, id=wrong_id)

    with pytest.raises(SpecValidationError, match="does not match"):
        attach_spec_id(wrong, family="representation")

    with pytest.raises(SpecValidationError, match="does not match"):
        validate_spec(wrong, family="representation")
