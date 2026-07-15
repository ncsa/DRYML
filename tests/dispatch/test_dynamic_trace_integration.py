"""Focused Sprint 9C dispatch/current-process trace integration contracts."""

from __future__ import annotations

import pytest

import dryml
from dryml.code import CodeAnalysisContext, CodeAnalysisResult, DynamicTracePolicy
from dryml.code.facts import DiagnosticFact
from dryml.code.targets import CodeTargetSpec
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.dispatch.errors import DispatchPlanningError
from dryml.dispatch.normalize import normalize_user_operation
from dryml.dispatch.requirements import resolve_dispatch_plan
from dryml.core2.utils.general import pickle_save
from dryml.formats.refs import format_cdef_id


_calls: list[str] = []


def traceable_target():
    _calls.append("trace")


def failing_trace_target():
    _calls.append("failed")
    raise RuntimeError("trusted target failure")


@dryml.env.req(requirements=("trace-class>=1",))
class TraceRequirementModel:
    @dryml.env.req(requirements=("trace-method>=1",))
    def train(self):
        raise AssertionError("trace must use a proxy")


def nested_requirement_target(model):
    model.train()


def _store(tmp_path):
    return DirStore(tmp_path / "store", query_index="none")


def test_dynamic_trace_policy_is_validated_before_normalization(monkeypatch, tmp_path):
    import dryml.dispatch.planner as planner

    monkeypatch.setattr(
        planner,
        "normalize_user_operation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not normalize")),
    )
    with pytest.raises(DispatchPlanningError, match="dynamic_trace"):
        Dispatcher(store=_store(tmp_path)).plan(traceable_target, analysis_policy={"dynamic_trace": False})


def test_opt_in_plan_traces_once_with_private_forced_collect_context(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    _calls.clear()
    observed = []
    original = requirements.trace

    def trace_spy(*args, **kwargs):
        observed.append((kwargs["context"], kwargs["policy"]))
        return original(*args, **kwargs)

    caller_context = CodeAnalysisContext(diagnostics_policy="raise")
    monkeypatch.setattr(requirements, "trace", trace_spy)
    plan = Dispatcher(store=_store(tmp_path)).plan(
        traceable_target,
        analysis_policy={"context": caller_context, "dynamic_trace": True},
        requirement_policy="ignore",
    )

    trace_context, trace_policy = observed.pop()
    assert _calls == ["trace"]
    assert trace_context.allow_dynamic_execution is True
    assert trace_context.algorithms == ("dynamic_trace",)
    assert trace_context.diagnostics_policy == "collect"
    assert caller_context.diagnostics_policy == "raise"
    assert isinstance(trace_policy, DynamicTracePolicy)
    evidence = plan.resolution.dynamic_trace.to_data()
    assert evidence["status"] == "complete"
    assert evidence["trace_input_id"]
    assert evidence["trace_run_id"]
    assert "dryml.dispatch.dynamic_trace" in plan.dispatch_spec["payload"]["metadata"]
    assert "dryml.dispatch.dynamic_trace" not in plan.envelope.operation_spec.get("metadata", {})


def test_explain_traces_once_without_persisting_and_no_opt_in_never_executes(tmp_path):
    _calls.clear()
    store = _store(tmp_path)
    dispatcher = Dispatcher(store=store)

    ordinary = dispatcher.explain(traceable_target, requirement_policy="ignore")
    assert ordinary.launchable
    assert _calls == []
    traced = dispatcher.explain(
        traceable_target,
        analysis_policy={"dynamic_trace": DynamicTracePolicy()},
        requirement_policy="ignore",
    )
    assert traced.launchable
    assert _calls == ["trace"]
    assert traced.resolution.dynamic_trace.data["status"] == "complete"
    assert not store.records.specs_dir.exists()


def test_requested_trace_failure_blocks_plan_and_explain_under_ignore(tmp_path):
    _calls.clear()
    dispatcher = Dispatcher(store=_store(tmp_path))

    with pytest.raises(DispatchPlanningError) as excinfo:
        dispatcher.plan(
            failing_trace_target,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )
    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "failed"
    assert evidence["trace_input_id"] and evidence["trace_run_id"]

    explanation = dispatcher.explain(
        failing_trace_target,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )
    assert not explanation.launchable
    assert explanation.resolution.dynamic_trace.data["status"] == "failed"
    assert _calls == ["failed", "failed"]


def test_trace_uses_worker_resolver_cdef_invocation_and_adds_nested_fragments(tmp_path):
    cdef = ConcreteDefinition(TraceRequirementModel)
    cdef_id = format_cdef_id(cdef.stable_hash())

    class CDefStore:
        def object_dir_for_cdef_id(self, value):
            assert value == cdef_id
            return str(tmp_path)

        def has(self, value):
            return value == cdef

    pickle_save(cdef, tmp_path / "def.pkl")
    normalized = normalize_user_operation(
        nested_requirement_target,
        store=CDefStore(),
        args=(cdef,),
        trace_enabled=True,
    )
    resolution = resolve_dispatch_plan(
        normalized,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )

    assert normalized.operation_spec["payload"]["args"] == [cdef_id]
    assert resolution.dynamic_trace.data["status"] == "complete"
    assert resolution.requirements.environment_requirement.requirements == (
        "trace-class>=1",
        "trace-method>=1",
    )


@pytest.mark.parametrize(
    ("result", "status", "run_present"),
    (
        (
            CodeAnalysisResult(
                CodeTargetSpec("function"),
                diagnostics=(DiagnosticFact(
                    severity="error",
                    code="dryml.code.dynamic_trace_unsupported_argument",
                    message="bounded pre-execution diagnostic",
                ),),
            ),
            "pre_execution_failed",
            False,
        ),
        (CodeAnalysisResult(CodeTargetSpec("function")), "evidence_rejected", True),
    ),
)
def test_trace_result_admission_keeps_preexecution_and_rejected_identity_rules(monkeypatch, tmp_path, result, status, run_present):
    import dryml.dispatch.requirements as requirements

    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: result)
    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(
            traceable_target,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )
    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == status
    assert evidence["trace_input_id"]
    assert bool(evidence["trace_run_id"]) is run_present
    assert evidence["execution_started"] is (False if status == "pre_execution_failed" else None)
