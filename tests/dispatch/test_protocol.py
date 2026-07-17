import pytest

from dryml.dispatch import (
    ExecutionEnvelope,
    WorkerHandshakeResponse,
    WorkerResponse,
    WorkerStoreRef,
    attach_dispatch_id,
    attach_recipe_id,
    make_dispatch_spec,
    make_execution_recipe,
    validate_dispatch_spec,
    validate_execution_recipe,
)
from dryml.dispatch.protocol import DISPATCH_WORKER_PROTOCOL_SCHEMA
from dryml.formats.ids import content_id
from dryml.operations import attach_operation_id, make_function_call_spec


def _envelope(tmp_path):
    op = attach_operation_id(make_function_call_spec("pkg.mod:fn"))
    dispatch = {"schema": "dryml.dispatch.v1", "schema_version": 1, "kind": "dispatch", "id": content_id("dispatch", 1, {"d": 1}), "payload": {"operation_id": op["id"]}}
    recipe = {"schema": "dryml.execution_recipe.v1", "schema_version": 1, "kind": "execution_recipe", "id": content_id("recipe", 1, {"r": 1}), "payload": {"dispatch_id": dispatch["id"], "operation_id": op["id"], "backend": {"name": "dryml.local_subprocess"}}}
    return ExecutionEnvelope(dispatch_spec=dispatch, execution_recipe=recipe, operation_spec=op, store_refs=(WorkerStoreRef("dir_store", "shared", str(tmp_path)),))


def test_protocol_models_round_trip(tmp_path):
    envelope = ExecutionEnvelope.from_json(_envelope(tmp_path).to_json())
    response = WorkerResponse.from_json(WorkerResponse(status="ok", operation_id=envelope.operation_id).to_json())
    handshake = WorkerHandshakeResponse.from_json(
        {
            "status": "ok",
            "protocol_schema": DISPATCH_WORKER_PROTOCOL_SCHEMA,
            "protocol_version": 1,
            "python_version": "3.x",
            "platform": "linux",
            "pid": 1,
            "features": ["operation.function_call"],
            "operation_kinds": ["function_call"],
            "call_transports": ["import_ref"],
            "store_ref_kinds": ["dir_store"],
            "record_schemas": {"record": 1},
            "runtime_modes": ["worker"],
        }
    )

    assert envelope.store_refs[0].path == str(tmp_path)
    assert response.status == "ok"
    assert handshake.protocol_version == 1


def test_planning_metadata_v2_carriers_remain_compatible(tmp_path):
    planning_v2 = {
        "dryml.dispatch.planning_version": 2,
        "dryml.code_analysis": {
            "target": {"kind": "function", "import_path": "pkg.mod:fn"},
            "facts": [{"kind": "callable", "source": {}, "data": {"name": "fn"}}],
            "diagnostics": [],
        },
        "dryml.code_probe": {
            "bootstrap_environment": {"kind": "current", "schema_version": 1},
            "bootstrap_probe": None,
            "final_probe": None,
        },
        "dryml.environment_selection": {
            "kind": "environment",
            "candidate": {"kind": "current", "schema_version": 1},
            "source": "explicit",
            "considered": [],
            "diagnostics": [],
        },
        "dryml.environment_check": {
            "kind": "environment",
            "status": "not_required",
            "compatible": None,
            "requirement": None,
            "candidate": {"kind": "current", "schema_version": 1},
            "details": [],
            "diagnostics": [],
        },
    }
    operation = attach_operation_id(make_function_call_spec("pkg.mod:fn"))
    dispatch = attach_dispatch_id(make_dispatch_spec(
        operation_id=operation["id"],
        metadata=planning_v2,
    ))
    recipe = attach_recipe_id(make_execution_recipe(
        dispatch_id=dispatch["id"],
        operation_id=operation["id"],
        backend={"name": "dryml.local_subprocess"},
        annotation_report=planning_v2,
    ))
    envelope = ExecutionEnvelope(
        dispatch_spec=dispatch,
        execution_recipe=recipe,
        operation_spec=operation,
        store_refs=(WorkerStoreRef("dir_store", "shared", str(tmp_path)),),
        reporting={"planning": planning_v2},
    )

    assert validate_dispatch_spec(dispatch) == dispatch
    assert validate_execution_recipe(recipe) == recipe
    serialized = envelope.to_json()
    assert ExecutionEnvelope.from_json(serialized).to_json() == serialized
    assert serialized["dispatch_spec"]["payload"]["metadata"] == planning_v2
    assert serialized["execution_recipe"]["payload"]["annotation_report"] == planning_v2
    assert serialized["reporting"]["planning"] == planning_v2


def test_protocol_invalid_shapes_reject_strings(tmp_path):
    data = _envelope(tmp_path).to_json()
    data["store_refs"] = "not-a-list"
    with pytest.raises(Exception, match="store_refs"):
        ExecutionEnvelope.from_json(data)
    with pytest.raises(Exception, match="result_cdef_ids"):
        WorkerResponse.from_json({"status": "ok", "result_cdef_ids": "not-a-list"})
    with pytest.raises(Exception, match="protocol"):
        WorkerHandshakeResponse.from_json({"status": "ok", "protocol_schema": "wrong", "protocol_version": 999, "pid": 1, "features": [], "operation_kinds": [], "call_transports": [], "store_ref_kinds": [], "record_schemas": {}, "runtime_modes": []})


def test_worker_response_status_context_invariants():
    with pytest.raises(Exception, match="ok worker responses"):
        WorkerResponse(status="ok", error={"message": "bad"})
    with pytest.raises(Exception, match="require error or diagnostics"):
        WorkerResponse(status="failed")
    with pytest.raises(Exception, match="require cancellation"):
        WorkerResponse(status="cancelled")
    with pytest.raises(Exception, match="only valid"):
        WorkerResponse(status="failed", diagnostics=({"message": "bad"},), cancellation={"requested": True})

    assert WorkerResponse(status="unsupported", diagnostics=({"message": "unsupported"},)).status == "unsupported"


def test_coordination_metadata_validates_worker_key_and_paths(tmp_path):
    allocation = {"role": "trainer", "replica": 0, "rank": 1, "local_rank": 1}
    envelope = ExecutionEnvelope(
        dispatch_spec=_envelope(tmp_path).dispatch_spec,
        execution_recipe=_envelope(tmp_path).execution_recipe,
        operation_spec=_envelope(tmp_path).operation_spec,
        allocation_view=allocation,
        store_refs=(WorkerStoreRef("dir_store", "shared", str(tmp_path)),),
        launch={"coordination": {"worker_key": dict(allocation), "start_path": str(tmp_path / "start.json"), "cancel_path": str(tmp_path / "cancel.json")}},
    )

    assert envelope.launch["coordination"]["worker_key"]["role"] == "trainer"

    bad = envelope.to_json()
    bad["launch"]["coordination"]["start_path"] = "relative.json"
    with pytest.raises(Exception, match="absolute"):
        ExecutionEnvelope.from_json(bad)

    bad = envelope.to_json()
    bad["launch"]["coordination"]["worker_key"]["rank"] = 99
    with pytest.raises(Exception, match="worker_key"):
        ExecutionEnvelope.from_json(bad)
