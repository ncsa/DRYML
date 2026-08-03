import json
import os

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
from dryml.dispatch.protocol import write_json_file
from dryml.dispatch.planner import _allocation_to_json, allocation_from_json
from dryml.formats.ids import content_id
from dryml.operations import attach_operation_id, make_function_call_spec
from dryml.runtime import RuntimeAllocationView
from dryml.worlds import LocalResourceInventory


def _envelope(tmp_path):
    op = attach_operation_id(make_function_call_spec("pkg.mod:fn"))
    dispatch = {"schema": "dryml.dispatch.v1", "schema_version": 1, "kind": "dispatch", "id": content_id("dispatch", 1, {"d": 1}), "payload": {"operation_id": op["id"]}}
    recipe = {"schema": "dryml.execution_recipe.v1", "schema_version": 1, "kind": "execution_recipe", "id": content_id("recipe", 1, {"r": 1}), "payload": {"dispatch_id": dispatch["id"], "operation_id": op["id"], "backend": {"name": "dryml.local_subprocess"}}}
    return ExecutionEnvelope(
        dispatch_spec=dispatch,
        execution_recipe=recipe,
        operation_spec=op,
        environment_spec={"kind": "current", "schema_version": 1},
        world_spec={"roles": {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}},
        runtime_spec={"mode": "worker", "device_visibility": {"policy": "assigned"}, "world_allocation_id": "worldalloc-v1-test"},
        allocation_view={"world_allocation_id": "worldalloc-v1-test", "role": "worker", "replica": 0, "rank": 0, "local_rank": 0, "cpus": [0], "accelerators": {}, "env": {}, "metadata": {}},
        requirement_policy="strict",
        requirement_axes=["environment", "world", "runtime"],
        handshake={"min_protocol": 1, "required_features": ["runtime.worker_session.v2"]},
        store_refs=(WorkerStoreRef("dir_store", "shared", str(tmp_path)),),
    )


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


def test_v1_or_incomplete_envelopes_require_explicit_v2_replanning(tmp_path):
    data = _envelope(tmp_path).to_json()
    data["schema"] = "dryml.execution_envelope.v1"
    data["schema_version"] = 1
    with pytest.raises(Exception, match="replan with execution-envelope v2"):
        ExecutionEnvelope.from_json(data)

    data = _envelope(tmp_path).to_json()
    data.pop("world_spec")
    with pytest.raises(Exception, match="missing required field"):
        ExecutionEnvelope.from_json(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("replica", 1, "replica"),
        ("rank", 1, "ranks"),
        ("local_rank", 1, "ranks"),
        ("cpus", [0, 1], "CPU count"),
        ("accelerators", {"gpu": [0]}, "accelerators"),
    ],
)
def test_v2_envelope_rejects_allocation_incoherent_with_world(
    tmp_path, field, value, message
):
    data = _envelope(tmp_path).to_json()
    data["allocation_view"][field] = value

    with pytest.raises(Exception, match=message):
        ExecutionEnvelope.from_json(data)


def test_v2_envelope_rejects_runtime_allocation_identity_mismatch(tmp_path):
    data = _envelope(tmp_path).to_json()
    data["runtime_spec"]["world_allocation_id"] = "worldalloc-v1-other"

    with pytest.raises(Exception, match="identities do not match"):
        ExecutionEnvelope.from_json(data)


def test_protocol_json_write_retries_transient_replace_conflict(tmp_path, monkeypatch):
    path = tmp_path / "mailbox.json"
    path.write_text("{}", encoding="utf-8")
    real_replace = os.replace
    attempts = 0

    def replace_with_reader_conflict(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("destination is briefly open by a reader")
        real_replace(source, target)

    monkeypatch.setattr("dryml.dispatch.protocol.os.replace", replace_with_reader_conflict)

    write_json_file(str(path), {"sequence": 1})

    assert attempts == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"sequence": 1}


def test_managed_worker_result_round_trip_is_structured(tmp_path):
    managed_result = {
        "schema": "dryml.managed.operation_result.v1",
        "schema_version": 1,
        "status": "ok",
        "effects": {},
        "checkpoint_head": None,
    }

    response = WorkerResponse.from_json(
        WorkerResponse(
            status="ok",
            operation_id=_envelope(tmp_path).operation_id,
            managed_result=managed_result,
        ).to_json()
    )

    assert response.managed_result == managed_result
    with pytest.raises(Exception, match="managed_result"):
        WorkerResponse.from_json({"status": "ok", "managed_result": "not-a-mapping"})
    with pytest.raises(Exception, match="schema"):
        WorkerResponse(
            status="ok",
            managed_result={"schema": "wrong", "schema_version": 1, "status": "ok"},
        )


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
        environment_spec=_envelope(tmp_path).environment_spec,
        world_spec=_envelope(tmp_path).world_spec,
        runtime_spec=_envelope(tmp_path).runtime_spec,
        allocation_view=_envelope(tmp_path).allocation_view,
        requirement_policy="strict",
        requirement_axes=["environment", "world", "runtime"],
        handshake={"min_protocol": 1, "required_features": ["runtime.worker_session.v2"]},
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
    allocation = {**_envelope(tmp_path).allocation_view, "role": "trainer"}
    envelope = ExecutionEnvelope(
        dispatch_spec=_envelope(tmp_path).dispatch_spec,
        execution_recipe=_envelope(tmp_path).execution_recipe,
        operation_spec=_envelope(tmp_path).operation_spec,
        environment_spec=_envelope(tmp_path).environment_spec,
        world_spec={"roles": {"trainer": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}},
        runtime_spec=_envelope(tmp_path).runtime_spec,
        requirement_policy="strict",
        requirement_axes=["environment", "world", "runtime"],
        handshake={"min_protocol": 1, "required_features": ["runtime.worker_session.v2"]},
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


def test_accelerator_memory_allocation_requires_negotiation_and_round_trips(tmp_path):
    allocation = {
        **_allocation_to_json(
        RuntimeAllocationView(
            role="worker",
            replica=0,
            rank=0,
            local_rank=0,
            cpus=(0,),
            accelerators={"gpu": ("gpu-a",)},
            accelerator_memory={"gpu": {"gpu-a": 1024}},
        )
    ),
        "world_allocation_id": "worldalloc-v1-test",
    }
    world_spec = {
        "roles": {
            "worker": {
                "replicas": 1,
                "process": {
                    "resources": {
                        "cpus": 1,
                        "accelerators": {"gpu": 1},
                        "accelerator_memory": {"gpu": [1024]},
                    }
                },
            }
        }
    }

    assert allocation_from_json(allocation).accelerator_memory == {"gpu": {"gpu-a": 1024}}
    with pytest.raises(Exception, match="accelerator-memory allocation requires"):
        ExecutionEnvelope(
            dispatch_spec=_envelope(tmp_path).dispatch_spec,
            execution_recipe=_envelope(tmp_path).execution_recipe,
            operation_spec=_envelope(tmp_path).operation_spec,
            environment_spec=_envelope(tmp_path).environment_spec,
            world_spec=world_spec,
            runtime_spec=_envelope(tmp_path).runtime_spec,
            allocation_view=allocation,
            requirement_policy="strict",
            requirement_axes=["environment", "world", "runtime"],
            handshake={"min_protocol": 1, "required_features": ["runtime.worker_session.v2"]},
        )

    envelope = ExecutionEnvelope(
        dispatch_spec=_envelope(tmp_path).dispatch_spec,
        execution_recipe=_envelope(tmp_path).execution_recipe,
        operation_spec=_envelope(tmp_path).operation_spec,
        environment_spec=_envelope(tmp_path).environment_spec,
        world_spec=world_spec,
        runtime_spec=_envelope(tmp_path).runtime_spec,
        requirement_policy="strict",
        requirement_axes=["environment", "world", "runtime"],
        allocation_view=allocation,
        handshake={
            "min_protocol": 1,
            "required_features": ["runtime.accelerator_memory.v1", "runtime.worker_session.v2"],
        },
    )
    assert ExecutionEnvelope.from_json(envelope.to_json()).allocation_view == allocation


def test_planner_requests_accelerator_memory_feature_when_allocating_limits(tmp_path):
    from dryml.core.store.dir import DirStore
    from dryml.dispatch import Dispatcher

    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        make_function_call_spec("operator:add", args=[1, 2]),
        world={
            "roles": {
                "main": {
                    "replicas": 1,
                    "process": {
                        "resources": {
                            "accelerators": {"gpu": 1},
                            "accelerator_memory": {"gpu": ["1GiB"]},
                        }
                    },
                }
            }
        },
        inventory=LocalResourceInventory(
            (0,),
            {"gpu": ("gpu-a",)},
            accelerator_memory={"gpu": {"gpu-a": "2GiB"}},
        ),
        requirement_policy="ignore",
    )

    assert plan.envelope.allocation_view["accelerator_memory"] == {
        "gpu": [{"device": "gpu-a", "memory": 1024**3}]
    }
    assert "runtime.accelerator_memory.v1" in plan.envelope.handshake["required_features"]
