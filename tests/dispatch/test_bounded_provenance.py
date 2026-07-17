from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dryml.code import DynamicTracePolicy
from dryml.code.facts import CodeFact, DiagnosticFact
from dryml.core2.repo import Repo
from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, ExecutionEnvelope
from dryml.environments import CurrentEnvironmentSpec
from dryml.records import ExecutionRecord
from dryml.runtime import RuntimeContextSpec


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
    assert metadata["dryml.dispatch.planning_version"] == 3
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


def _contains_live_object(value):
    if isinstance(value, dict):
        return any(_contains_live_object(key) or _contains_live_object(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_live_object(item) for item in value)
    return value is not None and type(value) not in {str, int, float, bool}
