import pytest

from dryml.records import (
    RepresentationRequirement,
    RepresentationSpec,
    SpecValidationError,
    default_object_state_representation_spec,
    make_representation_spec,
    representation_satisfies,
)


def test_stable_representation_ids_and_payload_changes():
    left = make_representation_spec("fake.raw_state", version="1", parameters={"dtype": "float32"}, traits=("loadable-state",), storage_kinds=("object-dir",))
    right = make_representation_spec("fake.raw_state", version="1", parameters={"dtype": "float32"}, traits=("loadable-state",), storage_kinds=("object-dir",))
    changed = make_representation_spec("fake.raw_state", version="1", parameters={"dtype": "float64"}, traits=("loadable-state",), storage_kinds=("object-dir",))

    assert left["id"].startswith("repr-v1-")
    assert left["id"] == right["id"]
    assert left["id"] != changed["id"]


def test_default_object_state_round_trips_and_compatibility():
    spec = RepresentationSpec(default_object_state_representation_spec())
    assert spec.kind == "dryml.object_state"
    assert spec.storage_kinds == ("object-dir",)

    exact = RepresentationRequirement(representation_id=spec.id)
    assert representation_satisfies(spec, exact).compatible
    wrong = RepresentationRequirement(representation_id=make_representation_spec("fake.raw_state")["id"])
    assert not representation_satisfies(spec, wrong).compatible


def test_kind_version_parameter_trait_and_storage_compatibility():
    spec = RepresentationSpec.create("fake.normalized_state", version="2", parameters={"layout": "single"}, traits=("loadable-state", "normalized"), storage_kinds=("product-dir",))
    req = RepresentationRequirement(kind="fake.normalized_state", version="2", parameters={"layout": "single"}, required_traits=("normalized",), storage_kinds=("product-dir",))
    assert representation_satisfies(spec, req).compatible

    assert not representation_satisfies(spec, RepresentationRequirement(kind="fake.raw_state")).compatible
    assert not representation_satisfies(spec, RepresentationRequirement(version="3")).compatible
    assert not representation_satisfies(spec, RepresentationRequirement(parameters={"layout": "sharded"})).compatible
    assert not representation_satisfies(spec, RepresentationRequirement(required_traits=("missing",))).compatible
    assert not representation_satisfies(spec, RepresentationRequirement(storage_kinds=("object-dir",))).compatible


def test_malformed_representation_specs_raise_structured_errors():
    with pytest.raises(SpecValidationError):
        RepresentationSpec.create("fake.raw_state", traits=("",))
    malformed = make_representation_spec("fake.raw_state")
    malformed["payload"]["parameters"] = []
    with pytest.raises(SpecValidationError):
        RepresentationSpec(malformed)
