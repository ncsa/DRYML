import pytest

from dryml.dispatch import ExecutionEnvelope, WorkerHandshakeResponse, WorkerResponse, WorkerStoreRef
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
