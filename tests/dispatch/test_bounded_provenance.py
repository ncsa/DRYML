from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from dryml.code import DynamicTracePolicy
from dryml.code.facts import CodeFact, DiagnosticFact
from dryml.core.repo import Repo
from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher, ExecutionEnvelope
from dryml.environments import CurrentEnvironmentSpec, PythonExecutableSpec
from dryml.formats import json_ready
from dryml.operations import make_function_call_spec
from dryml.records import ExecutionRecord
from dryml.runtime import RuntimeContextSpec, RuntimeMode
from dryml.worlds import LocalResourceInventory


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


FORBIDDEN = (
    "LIVE_OBJECT_SENTINEL",
    "OVERSIZED_SOURCE_SENTINEL",
    "RAW_PROBE_OUTPUT_SENTINEL",
    "RAW_PROCESS_OUTPUT_SENTINEL",
    "CREDENTIAL_SENTINEL",
    "ENV_DUMP_SENTINEL",
    "PRIVATE_BENCHMARK_HOST_SENTINEL",
    "UNRESTRICTED_TRACE_ARGS_SENTINEL",
    "RUNTIME_CANDIDATE_ENV_SENTINEL",
    "RUNTIME_CANDIDATE_METADATA_SENTINEL",
    "RUNTIME_CANDIDATE_SECRET_SENTINEL",
)

LAUNCH_CONFIG_SENTINELS = (
    "ENVIRONMENT_LAUNCH_SECRET_SENTINEL",
    "ENVIRONMENT_PRIVATE_PATH_SENTINEL",
    "WORLD_LAUNCH_SECRET_SENTINEL",
    "WORLD_METADATA_SECRET_SENTINEL",
    "RUNTIME_ENV_SECRET_SENTINEL",
    "RUNTIME_FRAMEWORK_SECRET_SENTINEL",
    "RUNTIME_METADATA_SECRET_SENTINEL",
    "DEVICE_VISIBILITY_ID_SECRET_SENTINEL",
    "ALLOCATION_ACCELERATOR_ID_SECRET_SENTINEL",
)

INVENTORY_METADATA_SENTINEL = "INVENTORY_METADATA_SECRET_SENTINEL"


class _LiveSentinel:
    def __repr__(self):
        return "LIVE_OBJECT_SENTINEL"


@pytest.mark.parametrize("target_kind", ("function", "method"))
def test_bounded_provenance_matrix(tmp_path, monkeypatch, target_kind):
    import dryml.dispatch.backends as backends
    import dryml.dispatch.requirements as requirements

    monkeypatch.syspath_prepend(str(FIXTURE_ROOT))
    from provenance_targets import ProvenanceBox, provenance_add

    original_analyze = requirements.analyze
    original_worker_command = backends.build_worker_command

    def analyze_with_forbidden_payload(*args, **kwargs):
        result = original_analyze(*args, **kwargs)
        payload = CodeFact(
            kind="audit_payload",
            data={
                "live_object": _LiveSentinel(),
                "source": "OVERSIZED_SOURCE_SENTINEL" * 10_000,
                "probe_stdout": "RAW_PROBE_OUTPUT_SENTINEL",
                "process_output": "RAW_PROCESS_OUTPUT_SENTINEL",
                "credential": "CREDENTIAL_SENTINEL",
                "env": {"TOKEN": "ENV_DUMP_SENTINEL"},
                "benchmark_host": "PRIVATE_BENCHMARK_HOST_SENTINEL",
                "args": ["UNRESTRICTED_TRACE_ARGS_SENTINEL"],
            },
        )
        diagnostic = DiagnosticFact(
            code="audit.sensitive_payload",
            severity="warning",
            message="RAW_PROCESS_OUTPUT_SENTINEL",
            source={"host": "PRIVATE_BENCHMARK_HOST_SENTINEL"},
            data={"credential": "CREDENTIAL_SENTINEL", "env": {"TOKEN": "ENV_DUMP_SENTINEL"}},
        )
        return type(result)(result.target, result.facts + (payload,), result.diagnostics + (diagnostic,))

    monkeypatch.setattr(requirements, "analyze", analyze_with_forbidden_payload)

    def worker_command_with_fixture(environment_spec):
        command, env = original_worker_command(environment_spec)
        env["PYTHONPATH"] = os.pathsep.join((str(FIXTURE_ROOT), env.get("PYTHONPATH", "")))
        return command, env

    monkeypatch.setattr(backends, "build_worker_command", worker_command_with_fixture)

    store = DirStore(tmp_path / "store", query_index="none")
    dispatcher = Dispatcher(store=store)
    if target_kind == "function":
        target = provenance_add
        method_name = None
        args = (2, 3)
    else:
        box = ProvenanceBox(2)
        Repo(stores=[store]).save(box, store=store, record_policy="none")
        target = box.definition
        method_name = "plus"
        args = (3,)

    plan = dispatcher.plan(
        target,
        method_name,
        args=args,
        environment=CurrentEnvironmentSpec(),
        analysis_policy=(
            {"dynamic_trace": DynamicTracePolicy(require_proxy_only_args=False)}
            if target_kind == "function"
            else None
        ),
    )
    operation = plan.envelope.operation_spec
    dispatch = plan.dispatch_spec
    recipe = plan.execution_recipe
    metadata = dispatch["payload"]["metadata"]
    outcomes = metadata["dryml.dispatch.analysis_outcomes"]

    assert operation["metadata"]["dryml.code_target"]
    assert operation["metadata"]["dryml.dispatch.transport"] in {"import_path", "method_call"}
    assert operation["id"] == dispatch["payload"]["operation_id"] == recipe["payload"]["operation_id"]
    assert operation["payload"].get("method") == method_name
    assert metadata["dryml.dispatch.planning_version"] == 4
    assert metadata["dryml.requirements"]["world_requirement"]
    assert metadata["dryml.requirement_sources"]
    assert metadata["dryml.environment_selection"]["source"] == "explicit"
    assert metadata["dryml.world_selection"]["source"] == "synthesized"
    assert metadata["dryml.environment_check"]["status"] == "not_required"
    assert metadata["dryml.world_check"]["status"] == "satisfied"
    assert metadata["dryml.runtime_check"]["status"] == "not_required"
    assert metadata["dryml.requirement_policy"] == "strict"
    assert metadata["dryml.runtime_enforcement"] == "strict"
    assert outcomes["code_probe"] == {"outcome": "not_required"}
    assert outcomes["environment_probe"] == {"outcome": "not_required"}
    if target_kind == "function":
        assert outcomes["dynamic_trace"] == {
            "requested": True,
            "completed": True,
            "outcome": "complete",
        }
        assert metadata["dryml.dispatch.dynamic_trace"]["status"] == "complete"
    else:
        assert outcomes["dynamic_trace"] == {
            "requested": False,
            "completed": False,
            "outcome": "not_requested",
        }
        assert "dryml.dispatch.dynamic_trace" not in metadata
    assert metadata["dryml.code_analysis"]["fact_counts"]["other"] == 1
    assert metadata["dryml.dispatch.diagnostics"][-1] == {
        "code": "audit.sensitive_payload",
        "severity": "warning",
    }
    assert recipe["payload"]["annotation_report"] == metadata
    assert recipe["payload"]["backend"]["kind"] == "local_subprocess"

    explanation = dispatcher.explain(
        target,
        method_name,
        args=args,
        environment=CurrentEnvironmentSpec(),
        runtime=RuntimeContextSpec(
            env={"TOKEN": "RUNTIME_CANDIDATE_ENV_SENTINEL"},
            frameworks={
                "audit": {"token": "RUNTIME_CANDIDATE_SECRET_SENTINEL"},
            },
            metadata={"audit": "RUNTIME_CANDIDATE_METADATA_SENTINEL"},
        ).to_data(),
    )
    public_resolution = explanation.to_data()["resolution"]
    assert "OVERSIZED_SOURCE_SENTINEL" in json.dumps(public_resolution["code_analysis"])
    assert public_resolution["runtime_selection"]["candidate"]["env"] == {
        "TOKEN": "RUNTIME_CANDIDATE_ENV_SENTINEL",
    }
    assert public_resolution["runtime_check"]["candidate"]["metadata"] == {
        "audit": "RUNTIME_CANDIDATE_METADATA_SENTINEL",
    }
    assert public_resolution["runtime_check"]["candidate"]["frameworks"] == {
        "audit": {"token": "RUNTIME_CANDIDATE_SECRET_SENTINEL"},
    }

    persisted_with_runtime_sentinels = explanation.resolution.metadata()
    assert not any(
        value in json.dumps(persisted_with_runtime_sentinels, sort_keys=True)
        for value in FORBIDDEN
    )

    envelope_data = plan.envelope.to_json()
    restored = ExecutionEnvelope.from_json(envelope_data)
    assert restored.to_json() == envelope_data
    assert restored.dispatch_spec["id"] == dispatch["id"]
    assert restored.execution_recipe["id"] == recipe["id"]
    assert restored.operation_spec["id"] == operation["id"]
    assert restored.reporting["planning"]["dryml.dispatch.analysis_outcomes"] == outcomes

    assert store.records.read_spec(operation["id"], family="operation") == envelope_data["operation_spec"]
    assert store.records.read_spec(dispatch["id"], family="dispatch") == envelope_data["dispatch_spec"]
    assert store.records.read_spec(recipe["id"], family="execution_recipe") == envelope_data["execution_recipe"]

    response = dispatcher.submit(plan).result(timeout=20)
    assert response.status == "ok", response.error
    assert response.result_canonical == 5
    record_envelope = store.records.read_record(response.execution_record_id)
    record = ExecutionRecord.from_envelope(record_envelope)
    assert record.operation_id == operation["id"]
    assert record.dispatch_id == dispatch["id"]
    assert record.recipe_id == recipe["id"]
    assert record.backend["kind"] == recipe["payload"]["backend"]["kind"]
    assert record.extra["worker_key"] == {"role": "main", "replica": 0, "rank": 0, "local_rank": 0}

    carriers = (
        envelope_data["operation_spec"],
        envelope_data["dispatch_spec"],
        envelope_data["execution_recipe"],
        envelope_data,
        record_envelope,
    )
    serialized = json.dumps(carriers, sort_keys=True)
    assert not any(value in serialized for value in FORBIDDEN)
    assert not _contains_live_object(carriers)


@pytest.mark.parametrize("local_world", (False, True))
def test_launch_configuration_is_redacted_from_persistent_specs(tmp_path, local_world):
    from dryml.dispatch.backends import _write_execution_record

    def build(store_name, suffix):
        store = DirStore(tmp_path / store_name, query_index="none")
        private_pythonpath = tmp_path / f"{LAUNCH_CONFIG_SENTINELS[1]}-{suffix}"
        private_pythonpath.mkdir()
        environment = PythonExecutableSpec(
            sys.executable,
            env={"API_TOKEN": f"{LAUNCH_CONFIG_SENTINELS[0]}-{suffix}"},
            pythonpath_policy="explicit",
            extra_pythonpath=(str(private_pythonpath),),
        ).to_data()
        runtime = RuntimeContextSpec(
            mode=RuntimeMode.WORKER,
            device_visibility={
                "policy": "assigned",
                "accelerators": [f"{LAUNCH_CONFIG_SENTINELS[7]}-{suffix}"],
            },
            env={"SERVICE_TOKEN": f"{LAUNCH_CONFIG_SENTINELS[4]}-{suffix}"},
            frameworks={"plain": {"credential": f"{LAUNCH_CONFIG_SENTINELS[5]}-{suffix}"}},
            metadata={"credential": f"{LAUNCH_CONFIG_SENTINELS[6]}-{suffix}"},
        ).to_data()
        backend_kind = "local_world" if local_world else "local_subprocess"
        world = {
            "roles": {
                "main": {
                    "replicas": 1,
                    "process": {
                        "env": {
                            "SERVICE_TOKEN": f"{LAUNCH_CONFIG_SENTINELS[2]}-{suffix}",
                            "SECONDARY_TOKEN": f"{LAUNCH_CONFIG_SENTINELS[3]}-{suffix}",
                        },
                        "resources": {"accelerators": {"gpu": 1}},
                    },
                }
            },
            "backend": {"kind": backend_kind, "parameters": {}},
        }
        planner = Dispatcher(store=store)
        kwargs = {
            "environment": environment,
            "runtime": runtime,
            "world": world,
            "inventory": LocalResourceInventory(
                (0,),
                accelerators={"gpu": (f"{LAUNCH_CONFIG_SENTINELS[8]}-{suffix}",)},
                metadata={"private": f"{INVENTORY_METADATA_SENTINEL}-{suffix}"},
            ),
            "requirement_policy": "ignore",
        }
        operation = make_function_call_spec("operator:add", args=[1, 2])
        planned = planner.plan_world(operation, **kwargs) if local_world else planner.plan(operation, **kwargs)
        envelope = planned.worker_plans[0].envelope if local_world else planned.envelope
        return store, planned, envelope

    store, planned, envelope = build("first", "one")
    _, comparison, _ = build("second", "two")

    launch_only = json.dumps(
        {
            "environment": envelope.environment_spec,
            "runtime": envelope.runtime_spec,
            "allocation": envelope.allocation_view,
            "world": envelope.launch["world_spec"],
            "world_allocation": envelope.launch["world_allocation_spec"],
        },
        sort_keys=True,
    )
    if not all(value in launch_only for value in LAUNCH_CONFIG_SENTINELS):
        pytest.fail("launch envelope did not retain all launch-only configuration")

    provenance_specs = (
        envelope.operation_spec,
        planned.dispatch_spec,
        planned.execution_recipe,
        envelope.launch["provenance_world_spec"],
        envelope.launch["provenance_world_allocation_spec"],
        store.records.read_spec(planned.dispatch_spec["id"], family="dispatch"),
        store.records.read_spec(planned.execution_recipe["id"], family="execution_recipe"),
        store.records.read_spec(envelope.launch["world_id"], family="world"),
        store.records.read_spec(envelope.launch["world_allocation_id"], family="world_allocation"),
    )
    serialized = json.dumps(json_ready(provenance_specs), sort_keys=True)
    forbidden_persistence = (*LAUNCH_CONFIG_SENTINELS, INVENTORY_METADATA_SENTINEL)
    if any(value in serialized for value in forbidden_persistence):
        pytest.fail("persistent provenance retained launch-only configuration")
    if "dryml.dispatch.persistence_projection.v1" not in serialized or "__dryml_redacted__" not in serialized:
        pytest.fail("persistent provenance omitted explicit redaction markers")

    record_id = _write_execution_record(
        store,
        envelope,
        status="failed",
        error={"type": "RuntimeError"},
        diagnostics=({"message": "worker execution failed"},),
    )
    record_data = json.dumps(json_ready(store.records.read_record(record_id)), sort_keys=True)
    if any(value in record_data for value in forbidden_persistence):
        pytest.fail("execution record retained launch-only configuration")

    if planned.dispatch_spec["id"] != comparison.dispatch_spec["id"]:
        pytest.fail("launch-only secret values changed dispatch identity")
    comparison_envelope = comparison.worker_plans[0].envelope if local_world else comparison.envelope
    if envelope.launch["world_id"] != comparison_envelope.launch["world_id"]:
        pytest.fail("launch-only secret values changed persisted world identity")
    if envelope.launch["world_allocation_id"] != comparison_envelope.launch["world_allocation_id"]:
        pytest.fail("launch-only secret values changed persisted allocation identity")


def _contains_live_object(value):
    if isinstance(value, dict):
        return any(_contains_live_object(key) or _contains_live_object(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_live_object(item) for item in value)
    return value is not None and type(value) not in {str, int, float, bool}
