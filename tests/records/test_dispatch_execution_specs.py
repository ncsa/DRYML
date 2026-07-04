import pytest

import dryml.operations as ops
from dryml.dispatch import (
    DispatchSpecError,
    attach_dispatch_id,
    attach_recipe_id,
    compute_dispatch_id,
    compute_recipe_id,
    make_dispatch_spec,
    make_execution_recipe,
    validate_dispatch_spec,
    validate_execution_recipe,
)
from dryml.formats.ids import content_id
from dryml.formats.refs import format_cdef_id
from dryml.records import make_spec, spec_family_for_id


def _op(char="a"):
    return content_id("op", 1, {"op": char})


def _dispatch(char="a"):
    return content_id("dispatch", 1, {"dispatch": char})


def _cdef(char="a"):
    return format_cdef_id(char * 64)


def test_dispatch_spec_stable_ids_and_policy_validation():
    left = attach_dispatch_id(make_dispatch_spec(operation_id=_op(), records={"provenance": True, "record_policy": "descriptive"}, execution={"backend": "local_subprocess"}))
    right = attach_dispatch_id(make_dispatch_spec(operation_id=_op(), execution={"backend": "local_subprocess"}, records={"record_policy": "descriptive", "provenance": True}))
    changed = attach_dispatch_id(make_dispatch_spec(operation_id=_op(), records={"record_policy": "provenance"}))

    assert left["id"].startswith("dispatch-v1-")
    assert left["id"] == right["id"]
    assert left["id"] != changed["id"]
    assert compute_dispatch_id(left) == left["id"]
    assert spec_family_for_id(left["id"]) == "dispatch"
    assert validate_dispatch_spec(left)["payload"]["operation_id"] == _op()

    with pytest.raises(DispatchSpecError, match="record_policy"):
        make_dispatch_spec(operation_id=_op(), records={"record_policy": "invalid"})
    with pytest.raises(DispatchSpecError, match="unknown"):
        validate_dispatch_spec({**left, "payload": {**left["payload"], "surprise": True}})


def test_dispatch_embedded_operation_must_match():
    operation = ops.attach_operation_id(ops.make_function_call_spec("pkg.mod:fn", args=[_cdef()]))
    dispatch = attach_dispatch_id(make_dispatch_spec(operation_id=operation["id"], operation=operation))

    assert dispatch["payload"]["operation"]["id"] == operation["id"]

    with pytest.raises(DispatchSpecError, match="match"):
        make_dispatch_spec(operation_id=_op("b"), operation=operation)


def test_method_call_dispatch_and_specs_are_not_objects():
    operation = ops.attach_operation_id(ops.make_method_call_spec(_cdef(), "train", kwargs={"epochs": 1}))
    dispatch = make_dispatch_spec(operation_id=operation["id"], operation=operation)

    assert dispatch["kind"] == "dispatch"
    assert dispatch["schema"] == "dryml.dispatch.v1"
    assert not hasattr(dispatch, "definition")


def test_execution_recipe_stable_ids_and_prefix_validation():
    left = attach_recipe_id(
        make_execution_recipe(
            dispatch_id=_dispatch(),
            operation_id=_op(),
            backend={"name": "dryml.local_subprocess", "kind": "local_subprocess"},
            input_plan={"materialize_cdefs": [_cdef()]},
            environment_spec_id=content_id("envspec", 1, {"env": 1}),
            world_allocation_id=content_id("worldalloc", 1, {"world": 1}),
        )
    )
    right = attach_recipe_id(
        make_execution_recipe(
            operation_id=_op(),
            dispatch_id=_dispatch(),
            backend={"kind": "local_subprocess", "name": "dryml.local_subprocess"},
            environment_spec_id=content_id("envspec", 1, {"env": 1}),
            world_allocation_id=content_id("worldalloc", 1, {"world": 1}),
            input_plan={"materialize_cdefs": [_cdef()]},
        )
    )

    assert left["id"].startswith("recipe-v1-")
    assert left["id"] == right["id"]
    assert left["id"] not in {_op(), _dispatch()}
    assert compute_recipe_id(left) == left["id"]
    assert spec_family_for_id(left["id"]) == "execution_recipe"
    assert validate_execution_recipe(left)["payload"]["backend"]["name"] == "dryml.local_subprocess"

    with pytest.raises(DispatchSpecError, match="backend"):
        make_execution_recipe(dispatch_id=_dispatch(), operation_id=_op(), backend={"kind": "missing-name"})
    with pytest.raises(DispatchSpecError, match="prefix"):
        make_execution_recipe(dispatch_id=_op(), operation_id=_op(), backend={"name": "x"})


def test_spec_family_metadata_accepts_dispatch_and_recipe():
    dispatch = make_spec(family="dispatch", kind="dispatch", payload={"operation_id": _op()})
    recipe = make_spec(family="execution_recipe", kind="execution_recipe", payload={"dispatch_id": _dispatch(), "operation_id": _op(), "backend": {"name": "x"}})

    assert dispatch["schema"] == "dryml.dispatch.v1"
    assert recipe["schema"] == "dryml.execution_recipe.v1"
