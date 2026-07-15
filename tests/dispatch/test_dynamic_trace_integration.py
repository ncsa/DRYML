"""Focused Sprint 9C dispatch/current-process trace integration contracts."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

import dryml
from dryml.code import CodeAnalysisContext, CodeAnalysisResult, DynamicTracePolicy, target_from_callable
from dryml.code.facts import AnnotationFact, CodeFact, DiagnosticFact, DynamicCallFact
from dryml.core2.definition import ConcreteDefinition, Definition
from dryml.core2.object import Object
from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.dispatch.errors import DispatchPlanningError
from dryml.dispatch.operations import PickledCallable
from dryml.dispatch.protocol import WorkerResponse
from dryml.dispatch.normalize import normalize_callable_operation, normalize_user_operation
from dryml.dispatch.requirements import DynamicTraceProvenance, _effective_trace_invocation, parse_analysis_policy, resolve_dispatch_plan
from dryml.operations import resolve_call_arguments
from dryml.core2.utils.general import pickle_save
from dryml.formats.refs import format_cdef_id
from dryml.formats import json_ready
from dryml.operations import attach_operation_id, make_function_call_spec


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


class StoredTraceObject(Object):
    @dryml.env.req(requirements=("stored-object-method>=1",))
    def train(self):
        raise AssertionError("trace must not invoke a stored method target")


def nested_requirement_target(model):
    model.train()


def argument_target(*args, **kwargs):
    _calls.append((args, kwargs))


def interrupting_target():
    raise KeyboardInterrupt()


class ConstructorSentinel:
    def __init__(self):
        raise AssertionError("trace must not construct classes")


class CallableSentinel:
    def __call__(self):
        raise AssertionError("trace must not call callable instances")


class InspectionHookCallable:
    """Callable sentinel whose generic callable metadata is unsafe to inspect."""

    def __getattribute__(self, name):
        if name in {"__name__", "__module__", "__qualname__", "__reduce__", "__reduce_ex__"}:
            raise AssertionError(f"trace eligibility must not inspect {name}")
        return super().__getattribute__(name)

    def __call__(self):
        raise AssertionError("trace eligibility must not invoke callable instances")


class DescriptorSentinel:
    def __get__(self, _instance, _owner):
        raise AssertionError("trace eligibility must not invoke descriptors")


def generator_target():
    yield None


async def async_target():
    return None


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


def test_analysis_policy_mixed_type_unknown_keys_raise_dispatch_error():
    with pytest.raises(DispatchPlanningError, match="unsupported fields"):
        parse_analysis_policy({"dynamic_trace": True, "unknown": None, 1: None})


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
                target_from_callable(traceable_target).spec,
                diagnostics=(DiagnosticFact(
                    severity="error",
                    code="dryml.code.dynamic_trace_unsupported_argument",
                    message="bounded pre-execution diagnostic",
                ),),
            ),
            "pre_execution_failed",
            False,
        ),
        (CodeAnalysisResult(target_from_callable(traceable_target).spec), "evidence_rejected", True),
    ),
)
def test_trace_result_admission_keeps_preexecution_and_rejected_identity_rules(monkeypatch, tmp_path, result, status, run_present):
    import dryml.dispatch.requirements as requirements

    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=result.facts,
            diagnostics=result.diagnostics,
        ),
    )
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


def test_explicit_operation_constructs_input_identity_before_unsupported_target(tmp_path):
    operation = attach_operation_id(make_function_call_spec("builtins:len", args=[[1]]))

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(
            operation,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "pre_execution_failed"
    assert evidence["trace_input_id"]
    assert evidence["trace_run_id"] is None


def test_pickled_preliminary_noncurrent_environment_never_traces_or_invokes(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    _calls.clear()
    traced = []
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: traced.append(True))
    foreign_python = str(tmp_path / "other-python")

    with pytest.raises(DispatchPlanningError):
        Dispatcher(store=_store(tmp_path)).plan(
            PickledCallable(traceable_target),
            environment=foreign_python,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    assert traced == []
    assert _calls == []


def test_complete_provenance_requires_valid_summary_and_strict_status_invariants():
    with pytest.raises(ValueError):
        DynamicTraceProvenance({
            "schema": "dryml.dispatch.dynamic_trace.v1",
            "schema_version": 1,
            "requested": True,
            "trace_input_id": "trace-input-v1-test",
            "trace_run_id": "trace-run-v1-test",
            "execution_location": "current_process",
            "execution_started": True,
            "target": {"target_kind": "function", "transport": "import_path"},
            "policy": {"max_calls": 1, "require_proxy_only_args": True, "collect_requirements": True},
            "status": "complete",
            "summary": None,
            "calls": [],
            "accepted_fragments": [],
            "duplicate_observations": [],
            "diagnostics": [],
        })


def test_existing_operation_strips_forged_trace_planning_metadata(tmp_path):
    operation = attach_operation_id(make_function_call_spec(
        "builtins:len",
        args=[[1]],
        metadata={"dryml.dispatch.dynamic_trace": {"forged": True}},
    ))

    normalized = normalize_user_operation(operation, store=_store(tmp_path))

    assert "dryml.dispatch.dynamic_trace" not in normalized.operation_spec["metadata"]


def test_effective_invocation_matches_worker_resolver_for_nested_reference_and_literal_forms(tmp_path):
    cdef = ConcreteDefinition(TraceRequirementModel)
    cdef_id = format_cdef_id(cdef.stable_hash())

    class CDefStore:
        def object_dir_for_cdef_id(self, value):
            assert value == cdef_id
            return str(tmp_path)

        def has(self, value):
            return value == cdef

    pickle_save(cdef, tmp_path / "def.pkl")
    caller_args = (cdef, [cdef_id, f"ref({cdef_id})", {"$literal": cdef_id}, {"nested": (cdef,)}])
    caller_kwargs = {"values": {"raw": cdef_id, "escaped": {"$literal": cdef_id}}}
    normalized = normalize_user_operation(
        argument_target,
        store=CDefStore(),
        args=caller_args,
        kwargs=caller_kwargs,
        trace_enabled=True,
    )

    trace_args, trace_kwargs = _effective_trace_invocation(normalized)

    assert caller_args[1][3]["nested"] == (cdef,)
    assert caller_kwargs["values"]["escaped"] == {"$literal": cdef_id}
    assert trace_args[0] == cdef
    assert trace_args[1] == [cdef, cdef_id, cdef_id, {"nested": [cdef]}]
    assert trace_kwargs == {"values": {"raw": cdef, "escaped": cdef_id}}
    assert normalized.trace_cdef_positions == (("args/0", cdef_id), ("args/1/3/nested/0", cdef_id))

    worker_calls = []

    def worker_materialize(value):
        worker_calls.append(("materialize", value))
        return cdef

    worker = resolve_call_arguments(
        normalized.operation_spec,
        materialize_cdef=worker_materialize,
        make_cdef_ref=lambda value: value,
    )
    assert worker.args == trace_args
    assert worker.kwargs == trace_kwargs
    assert worker_calls == [
        ("materialize", cdef_id),
        ("materialize", cdef_id),
        ("materialize", cdef_id),
        ("materialize", cdef_id),
    ]


@pytest.mark.parametrize("args", [(), ("retained",)])
def test_pickle_small_marker_is_retained_for_identity_and_stripped_for_effective_invocation(tmp_path, args):
    normalized = normalize_user_operation(
        PickledCallable(argument_target),
        store=_store(tmp_path),
        args=args,
        trace_enabled=True,
    )

    effective_args, effective_kwargs = _effective_trace_invocation(normalized)

    assert effective_args == args
    assert effective_kwargs == {}
    assert len(normalized.operation_spec["payload"]["args"]) == len(args) + 1
    malformed = replace(normalized, launch={**normalized.launch, "identity_arg_count": len(args) + 1})
    with pytest.raises(ValueError, match="pickle_small"):
        _effective_trace_invocation(malformed)
    malformed_suffix = replace(
        normalized,
        operation_spec={
            **normalized.operation_spec,
            "payload": {**normalized.operation_spec["payload"], "args": [*normalized.operation_spec["payload"]["args"][:-1], {"$literal": "wrong"}]},
        },
    )
    with pytest.raises(ValueError, match="pickle_small"):
        _effective_trace_invocation(malformed_suffix)


@pytest.mark.parametrize("argument", ["ref(cdef-v4-0123456789abcdef)", {"$literal": "cdef-v4-0123456789abcdef"}])
def test_resolver_scalar_forms_follow_default_and_scalar_enabled_trace_policy(argument, tmp_path):
    _calls.clear()
    dispatcher = Dispatcher(store=_store(tmp_path))

    with pytest.raises(DispatchPlanningError) as excinfo:
        dispatcher.plan(argument_target, args=(argument,), analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")
    assert excinfo.value.context["dynamic_trace"]["status"] == "pre_execution_failed"
    assert _calls == []

    dispatcher.plan(
        argument_target,
        args=(argument,),
        analysis_policy={"dynamic_trace": DynamicTracePolicy(require_proxy_only_args=False)},
        requirement_policy="ignore",
    )
    assert _calls == [(("cdef-v4-0123456789abcdef",), {})]


def test_plain_definition_trace_argument_rejects_without_concretize_or_target(monkeypatch, tmp_path):
    definition = Definition(TraceRequirementModel)
    _calls.clear()
    monkeypatch.setattr(Definition, "concretize", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not concretize")))

    with pytest.raises(DispatchPlanningError, match="plain Definition"):
        Dispatcher(store=_store(tmp_path)).plan(
            argument_target,
            args=(definition,),
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )
    assert _calls == []


def test_trace_definition_method_target_rejects_before_concretization(monkeypatch, tmp_path):
    definition = Definition(TraceRequirementModel)
    monkeypatch.setattr(
        Definition,
        "concretize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not concretize")),
    )

    with pytest.raises(DispatchPlanningError, match="plain Definition method"):
        Dispatcher(store=_store(tmp_path)).plan(
            definition,
            "train",
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )


@pytest.mark.parametrize("use_object", [False, True])
def test_stored_method_targets_keep_store_and_input_identity_without_persisting(monkeypatch, tmp_path, use_object):
    subject = StoredTraceObject() if use_object else ConcreteDefinition(TraceRequirementModel)
    cdef = subject.definition if use_object else subject
    cdef_id = format_cdef_id(cdef.stable_hash())

    class StoredCDefStore:
        def has(self, value):
            return value == cdef

        def object_dir_for_cdef_id(self, value):
            assert value == cdef_id
            return str(tmp_path)

    store = StoredCDefStore()
    pickle_save(cdef, tmp_path / "def.pkl")
    if use_object:
        monkeypatch.setattr(
            "dryml.dispatch.normalize.Repo.save",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not persist")),
        )

    normalized = normalize_user_operation(
        subject,
        "train",
        store=store,
        trace_enabled=True,
        persist_object=True,
    )
    resolution = resolve_dispatch_plan(
        normalized,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )

    evidence = resolution.dynamic_trace.to_data()
    assert normalized.trace_store is store
    assert evidence["status"] == "pre_execution_failed"
    assert evidence["trace_input_id"]
    assert evidence["trace_run_id"] is None


def test_admission_rejects_unexpected_fact_even_with_complete_zero_call_summary(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    result = CodeAnalysisResult(
        target_from_callable(traceable_target).spec,
        facts=(
            CodeFact("unexpected", data={}),
            CodeFact(
                "dynamic_trace_summary",
                source={"analyzer": "dynamic_trace", "target_kind": "function"},
                data={"complete": True, "outcome": "complete", "calls_recorded": 0, "max_calls": 10_000},
            ),
        ),
    )
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: result)

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["trace_input_id"] and evidence["trace_run_id"]
    assert evidence["execution_started"] is True
    assert evidence["summary"]["data"]["calls_recorded"] == 0
    assert evidence["calls"] == []


def test_admission_rejects_recognized_preexecution_diagnostic_mixed_with_summary(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    result = CodeAnalysisResult(
        target_from_callable(traceable_target).spec,
        facts=(_summary(),),
        diagnostics=(DiagnosticFact(
            severity="error",
            code="dryml.code.dynamic_trace_unsupported_argument",
            message="recognized only when diagnostic-only",
        ),),
    )
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: result)

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["execution_started"] is True
    assert evidence["summary"]["data"]["calls_recorded"] == 0


def test_complete_trace_with_nonpreexecution_diagnostic_returns_failed_plan_and_explain(monkeypatch, tmp_path):
    """Inconsistent post-start evidence remains a rejected, non-launchable carrier."""
    import dryml.dispatch.requirements as requirements

    def complete_result_with_diagnostic(*_args, **kwargs):
        return CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=(_summary(),),
            diagnostics=(DiagnosticFact(
                severity="error",
                code="dryml.code.dynamic_trace_algorithm_failed",
                message="must reject inconsistent complete evidence",
            ),),
        )

    monkeypatch.setattr(requirements, "trace", complete_result_with_diagnostic)
    dispatcher = Dispatcher(store=_store(tmp_path))

    with pytest.raises(DispatchPlanningError) as excinfo:
        dispatcher.plan(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["execution_started"] is True
    assert evidence["summary"]["data"]["complete"] is True
    assert evidence["diagnostics"][0]["code"] == "dryml.dispatch.dynamic_trace_evidence_rejected"

    explanation = dispatcher.explain(
        traceable_target,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )
    assert explanation.launchable is False
    assert explanation.resolution.dynamic_trace.data["status"] == "evidence_rejected"


def test_complete_summary_with_257_diagnostics_remains_bounded_rejected_evidence(monkeypatch, tmp_path):
    """Rejected diagnostic overflow cannot expose a private provenance exception."""
    import dryml.dispatch.requirements as requirements

    diagnostics = tuple(
        DiagnosticFact(
            severity="error",
            code="dryml.code.dynamic_trace_algorithm_failed",
            message="must not be projected",
        )
        for _ in range(257)
    )
    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=(_summary(),),
            diagnostics=diagnostics,
        ),
    )
    dispatcher = Dispatcher(store=_store(tmp_path))

    with pytest.raises(DispatchPlanningError) as excinfo:
        dispatcher.plan(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["trace_input_id"] and evidence["trace_run_id"]
    assert evidence["execution_started"] is True
    assert evidence["summary"]["data"]["calls_recorded"] == 0
    assert evidence["calls"] == []
    assert evidence["diagnostics"] == [{
        "code": "dryml.dispatch.dynamic_trace_evidence_rejected",
        "severity": "error",
        "data": {"trace_diagnostic_codes": []},
    }]

    explanation = dispatcher.explain(
        traceable_target,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )
    assert explanation.launchable is False
    explanation_evidence = explanation.resolution.dynamic_trace.to_data()
    assert explanation_evidence["status"] == evidence["status"]
    assert explanation_evidence["trace_input_id"] and explanation_evidence["trace_run_id"]
    assert explanation_evidence["summary"] == evidence["summary"]
    assert explanation_evidence["calls"] == evidence["calls"]
    assert explanation_evidence["diagnostics"] == evidence["diagnostics"]


def test_oversized_complete_trace_with_diagnostic_remains_rejected_for_plan_and_explain(monkeypatch, tmp_path):
    """Inconsistent complete evidence takes precedence over projection overflow."""
    import dryml.dispatch.requirements as requirements

    def oversized_complete_result_with_diagnostic(*_args, **kwargs):
        return CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=tuple(_dynamic_call(index) for index in range(257)) + (_summary(calls=257),),
            diagnostics=(DiagnosticFact(
                severity="error",
                code="dryml.code.dynamic_trace_algorithm_failed",
                message="must reject inconsistent complete evidence before overflow",
            ),),
        )

    monkeypatch.setattr(requirements, "trace", oversized_complete_result_with_diagnostic)
    dispatcher = Dispatcher(store=_store(tmp_path))

    with pytest.raises(DispatchPlanningError) as excinfo:
        dispatcher.plan(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["trace_input_id"] and evidence["trace_run_id"]
    assert evidence["execution_started"] is True
    assert evidence["summary"] is None
    assert evidence["calls"] == []
    assert evidence["diagnostics"][0]["code"] == "dryml.dispatch.dynamic_trace_evidence_rejected"

    explanation = dispatcher.explain(
        traceable_target,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )
    assert explanation.launchable is False
    explanation_evidence = explanation.resolution.dynamic_trace.data
    assert explanation_evidence["status"] == "evidence_rejected"
    assert explanation_evidence["summary"] is None
    assert explanation_evidence["calls"] == []


def test_admission_accepts_context_metadata_in_returned_target(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=(_summary(),),
        ),
    )

    plan = Dispatcher(store=_store(tmp_path)).plan(
        traceable_target,
        analysis_policy={"context": CodeAnalysisContext(metadata={"run_id": "audit"}), "dynamic_trace": True},
        requirement_policy="ignore",
    )

    assert plan.resolution.dynamic_trace.data["status"] == "complete"
    assert "audit" not in json.dumps(plan.resolution.dynamic_trace.to_data())


def test_trace_result_requires_current_input_and_run_correlation(monkeypatch, tmp_path):
    """A stale same-function result without this facade run's metadata rejects."""
    import dryml.dispatch.requirements as requirements

    stale = CodeAnalysisResult(target_from_callable(traceable_target).spec, facts=(_summary(),))
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: stale)

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(
            traceable_target,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["trace_input_id"] and evidence["trace_run_id"]
    assert "_dryml_dispatch_trace" not in json.dumps(evidence)


def test_trace_input_identity_includes_complete_facade_target_metadata(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=(_summary(),),
        ),
    )
    dispatcher = Dispatcher(store=_store(tmp_path))
    first = dispatcher.explain(
        traceable_target,
        analysis_policy={"context": CodeAnalysisContext(metadata={"variant": "one"}), "dynamic_trace": True},
        requirement_policy="ignore",
    )
    second = dispatcher.explain(
        traceable_target,
        analysis_policy={"context": CodeAnalysisContext(metadata={"variant": "two"}), "dynamic_trace": True},
        requirement_policy="ignore",
    )

    assert first.resolution.dynamic_trace.data["trace_input_id"] != second.resolution.dynamic_trace.data["trace_input_id"]


def test_parsed_trace_request_is_not_an_accepted_public_policy_form(tmp_path):
    parsed = parse_analysis_policy({"dynamic_trace": True})

    with pytest.raises(DispatchPlanningError, match="public CodeAnalysisContext or mapping"):
        parse_analysis_policy(parsed)
    with pytest.raises(DispatchPlanningError, match="public CodeAnalysisContext or mapping"):
        Dispatcher(store=_store(tmp_path)).plan(traceable_target, analysis_policy=parsed)


def test_rejected_provenance_requires_true_when_valid_summary_proves_start():
    data = {
        "schema": "dryml.dispatch.dynamic_trace.v1",
        "schema_version": 1,
        "requested": True,
        "trace_input_id": "trace-input-v1-test",
        "trace_run_id": "trace-run-v1-test",
        "execution_location": "current_process",
        "execution_started": None,
        "target": {"target_kind": "function", "transport": "import_path"},
        "policy": {"max_calls": 1, "require_proxy_only_args": True, "collect_requirements": True},
        "status": "evidence_rejected",
        "summary": _summary().to_data(),
        "calls": [],
        "accepted_fragments": [],
        "duplicate_observations": [],
        "diagnostics": [{
            "code": "dryml.dispatch.dynamic_trace_evidence_rejected",
            "severity": "error",
            "data": {"trace_diagnostic_codes": []},
        }],
    }
    with pytest.raises(ValueError, match="proves execution start"):
        DynamicTraceProvenance(data)


def test_trace_diagnostics_use_fixed_schema_and_preserve_underlying_codes(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    result = CodeAnalysisResult(
        target_from_callable(traceable_target).spec,
        facts=(_summary(complete=False, outcome="algorithm_failed"),),
        diagnostics=(DiagnosticFact(
            severity="error",
            code="dryml.code.dynamic_trace_algorithm_failed",
            message="must not be projected",
        ),),
    )

    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=result.facts,
            diagnostics=result.diagnostics,
        ),
    )
    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(
            traceable_target,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    diagnostic = excinfo.value.context["dynamic_trace"]["diagnostics"][0]
    assert set(diagnostic) == {"code", "severity", "data"}
    assert diagnostic["code"] == "dryml.dispatch.dynamic_trace_incomplete"
    assert diagnostic["data"] == {"trace_diagnostic_codes": ["dryml.code.dynamic_trace_algorithm_failed"]}


def _summary(*, complete=True, outcome="complete", calls=0, max_calls=10_000):
    return CodeFact(
        "dynamic_trace_summary",
        source={"analyzer": "dynamic_trace", "target_kind": "function"},
        data={"complete": complete, "outcome": outcome, "calls_recorded": calls, "max_calls": max_calls},
    )


def _dynamic_call(sequence):
    return DynamicCallFact(
        source={"analyzer": "dynamic_trace", "target_kind": "function"},
        data={
            "sequence": sequence,
            "receiver_kind": "concrete_definition",
            "receiver_ref": "cdef-v4-0123456789abcdef",
            "receiver_class": None,
            "method_name": "train",
            "args": [],
            "kwargs": {},
            "method_facts": [],
        },
    )


def test_stale_envelope_retains_independently_validated_start_evidence(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    stale = CodeAnalysisResult(
        target_from_callable(failing_trace_target).spec,
        facts=(_dynamic_call(0), _summary(calls=1)),
    )
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: stale)

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["execution_started"] is True
    assert evidence["summary"]["data"]["calls_recorded"] == 1
    assert [call["data"]["sequence"] for call in evidence["calls"]] == [0]


def test_stale_envelope_over_call_limit_remains_rejected(monkeypatch, tmp_path):
    """Stale complete evidence is rejected before its calls exceed projection bounds."""
    import dryml.dispatch.requirements as requirements

    stale = CodeAnalysisResult(
        target_from_callable(failing_trace_target).spec,
        facts=tuple(_dynamic_call(index) for index in range(257)) + (_summary(calls=257),),
    )
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: stale)

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(
            traceable_target,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["trace_input_id"] and evidence["trace_run_id"]
    assert evidence["execution_started"] is True
    assert evidence["summary"] is None
    assert evidence["calls"] == []
    assert evidence["diagnostics"][0]["code"] == "dryml.dispatch.dynamic_trace_identity_mismatch"


def test_unknown_9b_summary_outcome_is_rejected_not_incomplete(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    result = CodeAnalysisResult(
        target_from_callable(traceable_target).spec,
        facts=(_summary(complete=False, outcome="future_outcome"),),
    )
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: result)

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["execution_started"] is None


def test_policy_mismatch_retains_independently_validated_calls_that_prove_start(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    result = CodeAnalysisResult(
        target_from_callable(traceable_target).spec,
        facts=(_dynamic_call(0), _summary(calls=1, max_calls=1)),
    )
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: result)

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["execution_started"] is True
    assert evidence["summary"] is None
    assert [call["data"]["sequence"] for call in evidence["calls"]] == [0]


def test_257_call_projection_overflow_keeps_summary_in_bounded_carrier(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=tuple(_dynamic_call(index) for index in range(257)) + (_summary(calls=257),),
        ),
    )

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "provenance_limit_exceeded"
    assert evidence["summary"]["data"]["calls_recorded"] == 257
    assert evidence["calls"] == []


def test_byte_overflow_is_projected_without_raw_validation_error(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    calls = []
    for index in range(256):
        call = _dynamic_call(index)
        call.data["receiver_ref"] = "cdef-v4-" + "a" * 4_088
        calls.append(call)
    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=tuple(calls) + (_summary(calls=256),),
        ),
    )

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "provenance_limit_exceeded"
    assert evidence["calls"] == []
    assert evidence["summary"]["data"]["calls_recorded"] == 256


def test_provenance_policy_uses_exact_dynamic_trace_bounds():
    data = {
        "schema": "dryml.dispatch.dynamic_trace.v1",
        "schema_version": 1,
        "requested": True,
        "trace_input_id": "trace-input-v1-test",
        "trace_run_id": "trace-run-v1-test",
        "execution_location": "current_process",
        "execution_started": True,
        "target": {"target_kind": "function", "transport": "import_path"},
        "policy": {"max_calls": 10_000, "require_proxy_only_args": True, "collect_requirements": True},
        "status": "provenance_limit_exceeded",
        "summary": _summary(calls=257).to_data(),
        "calls": [],
        "accepted_fragments": [],
        "duplicate_observations": [],
        "diagnostics": [{
            "code": "dryml.dispatch.dynamic_trace_provenance_limit_exceeded",
            "severity": "error",
            "data": {"trace_diagnostic_codes": []},
        }],
    }
    assert DynamicTraceProvenance.from_data(data).to_data() == data
    for max_calls in (0, 10_001):
        with pytest.raises(ValueError):
            DynamicTraceProvenance({**data, "policy": {**data["policy"], "max_calls": max_calls}})


def test_provenance_preserves_65_valid_calls_without_generic_metadata_truncation():
    calls = [
        DynamicCallFact(
            source={"analyzer": "dynamic_trace", "target_kind": "function"},
            data={
                "sequence": index,
                "receiver_kind": "concrete_definition",
                "receiver_ref": "cdef-v4-0123456789abcdef",
                "receiver_class": None,
                "method_name": "train",
                "args": [],
                "kwargs": {},
                "method_facts": [],
            },
        ).to_data()
        for index in range(65)
    ]
    provenance = DynamicTraceProvenance({
        "schema": "dryml.dispatch.dynamic_trace.v1",
        "schema_version": 1,
        "requested": True,
        "trace_input_id": "trace-input-v1-test",
        "trace_run_id": "trace-run-v1-test",
        "execution_location": "current_process",
        "execution_started": True,
        "target": {"target_kind": "function", "transport": "import_path"},
        "policy": {"max_calls": 65, "require_proxy_only_args": True, "collect_requirements": True},
        "status": "complete",
        "summary": CodeFact(
            "dynamic_trace_summary",
            source={"analyzer": "dynamic_trace", "target_kind": "function"},
            data={"complete": True, "outcome": "complete", "calls_recorded": 65, "max_calls": 65},
        ).to_data(),
        "calls": calls,
        "accepted_fragments": [],
        "duplicate_observations": [],
        "diagnostics": [],
    })

    assert len(provenance.to_data()["calls"]) == 65


def test_unsafe_full_call_evidence_blocks_all_carriers_and_sidecars(monkeypatch, tmp_path):
    """V1 evidence is rejected rather than rewritten when it cannot persist safely."""

    import dryml.dispatch.requirements as requirements

    call = _dynamic_call(0)
    call.data["args"] = ["synthetic-positional-secret"]
    call.data["kwargs"] = {"api_key": "synthetic-secret-token"}
    fragment_data = dryml.annotations.fragments_for(TraceRequirementModel)[0].to_data()
    fragment_data["source"]["metadata"] = {"environment": {"API_TOKEN": "synthetic-method-fact-secret"}}
    call.data["method_facts"] = [AnnotationFact(
        data=fragment_data,
        source={
            "analyzer": "direct_annotations",
            "target_kind": "function",
            "annotation_source": fragment_data["source"],
        },
    ).to_data()]

    def trace_with_sensitive_argument(*_args, **kwargs):
        return CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=(call, _summary(calls=1)),
        )

    monkeypatch.setattr(requirements, "trace", trace_with_sensitive_argument)
    store = _store(tmp_path)
    dispatcher = Dispatcher(store=store)
    explanation = dispatcher.explain(
        traceable_target,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )
    with pytest.raises(DispatchPlanningError) as excinfo:
        dispatcher.plan(
            traceable_target,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    assert call.data["args"] == ["synthetic-positional-secret"]
    assert call.data["kwargs"] == {"api_key": "synthetic-secret-token"}
    assert not explanation.launchable
    carriers = (explanation.to_data()["resolution"]["dynamic_trace"], excinfo.value.context["dynamic_trace"])
    for carrier in carriers:
        assert carrier["status"] == "evidence_rejected"
        assert carrier["execution_started"] is True
        assert carrier["calls"] == []
        assert carrier["summary"] is None
        assert "synthetic-positional-secret" not in repr(carrier)
        assert "synthetic-secret-token" not in repr(carrier)
        assert "synthetic-method-fact-secret" not in repr(carrier)
    assert not store.records.specs_dir.exists()


@pytest.mark.parametrize("field,value", [
    ("args", ["bounded-value"]),
    ("kwargs", {"value": "bounded-value"}),
])
def test_provenance_restores_full_v1_call_wires(field, value):
    """The v1 carrier retains the complete established DynamicCallFact wire."""

    raw = _dynamic_call(0).to_data()
    raw["data"][field] = value
    data = {
        "schema": "dryml.dispatch.dynamic_trace.v1",
        "schema_version": 1,
        "requested": True,
        "trace_input_id": "trace-input-v1-test",
        "trace_run_id": "trace-run-v1-test",
        "execution_location": "current_process",
        "execution_started": True,
        "target": {"target_kind": "function", "transport": "import_path"},
        "policy": {"max_calls": 1, "require_proxy_only_args": True, "collect_requirements": True},
        "status": "complete",
        "summary": _summary(calls=1, max_calls=1).to_data(),
        "calls": [raw],
        "accepted_fragments": [],
        "duplicate_observations": [],
        "diagnostics": [],
    }
    assert DynamicTraceProvenance.from_data(data).to_data() == data


@pytest.mark.parametrize("field,value", [("args", "malformed"), ("kwargs", "malformed")])
def test_provenance_restoration_rejects_malformed_full_call_wires(field, value):
    raw = _dynamic_call(0).to_data()
    raw["data"][field] = value
    data = {
        "schema": "dryml.dispatch.dynamic_trace.v1",
        "schema_version": 1,
        "requested": True,
        "trace_input_id": "trace-input-v1-test",
        "trace_run_id": "trace-run-v1-test",
        "execution_location": "current_process",
        "execution_started": True,
        "target": {"target_kind": "function", "transport": "import_path"},
        "policy": {"max_calls": 1, "require_proxy_only_args": True, "collect_requirements": True},
        "status": "complete",
        "summary": _summary(calls=1, max_calls=1).to_data(),
        "calls": [raw],
        "accepted_fragments": [],
        "duplicate_observations": [],
        "diagnostics": [],
    }
    with pytest.raises(ValueError):
        DynamicTraceProvenance.from_data(data)


@pytest.mark.parametrize("field,value", [
    ("args", "malformed"),
    ("kwargs", "malformed"),
])
def test_admission_rejects_malformed_raw_call_arguments_before_projection(monkeypatch, tmp_path, field, value):
    """A malformed facade result cannot be rewritten into persisted redaction."""

    import dryml.dispatch.requirements as requirements

    call = _dynamic_call(0)
    call.data[field] = value
    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=(call, _summary(calls=1)),
        ),
    )

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(
            traceable_target,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "evidence_rejected"
    assert evidence["calls"] == []


@pytest.mark.parametrize("observation", [
    {"sequence": 99, "method": "unrelated", "fragment_key": "not-a-digest"},
    {"sequence": 99, "method": "unrelated", "fragment_key": hashlib.sha256(b"unrelated").hexdigest()},
])
def test_provenance_rejects_observations_unrelated_to_calls_or_annotation_facts(observation):
    call = _dynamic_call(0).to_data()
    summary = _summary(calls=1).to_data()
    data = {
        "schema": "dryml.dispatch.dynamic_trace.v1",
        "schema_version": 1,
        "requested": True,
        "trace_input_id": "trace-input-v1-test",
        "trace_run_id": "trace-run-v1-test",
        "execution_location": "current_process",
        "execution_started": True,
        "target": {"target_kind": "function", "transport": "import_path"},
        "policy": {"max_calls": 1, "require_proxy_only_args": True, "collect_requirements": True},
        "status": "complete",
        "summary": summary,
        "calls": [call],
        "accepted_fragments": [observation],
        "duplicate_observations": [],
        "diagnostics": [],
    }
    with pytest.raises(ValueError):
        DynamicTraceProvenance(data)


def test_provenance_rejects_call_target_kind_that_differs_from_carrier_target():
    call = DynamicCallFact(
        source={"analyzer": "dynamic_trace", "target_kind": "local_function"},
        data=_dynamic_call(0).data,
    ).to_data()
    data = {
        "schema": "dryml.dispatch.dynamic_trace.v1",
        "schema_version": 1,
        "requested": True,
        "trace_input_id": "trace-input-v1-test",
        "trace_run_id": "trace-run-v1-test",
        "execution_location": "current_process",
        "execution_started": True,
        "target": {"target_kind": "function", "transport": "import_path"},
        "policy": {"max_calls": 1, "require_proxy_only_args": True, "collect_requirements": True},
        "status": "complete",
        "summary": _summary(calls=1).to_data(),
        "calls": [call],
        "accepted_fragments": [],
        "duplicate_observations": [],
        "diagnostics": [],
    }
    with pytest.raises(ValueError):
        DynamicTraceProvenance(data)


def test_store_live_cdef_mismatch_blocks_before_trace_or_target(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    supplied = ConcreteDefinition(TraceRequirementModel)
    stored = ConcreteDefinition(dict)
    cdef_id = format_cdef_id(supplied.stable_hash())

    class MismatchStore:
        def object_dir_for_cdef_id(self, value):
            assert value == cdef_id
            return str(tmp_path)

        def has(self, value):
            return value == supplied

    pickle_save(stored, tmp_path / "def.pkl")
    traced = []
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: traced.append(True))
    _calls.clear()

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=MismatchStore()).plan(
            argument_target,
            args=(supplied,),
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    assert excinfo.value.context["dynamic_trace"]["trace_input_id"] is None
    assert traced == []
    assert _calls == []


def test_final_pickle_environment_rejection_retains_completed_trace_carrier(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    checks = iter((True, False))
    monkeypatch.setattr(requirements, "_same_python_environment", lambda _value: next(checks))
    _calls.clear()

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(
            PickledCallable(traceable_target),
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    evidence = excinfo.value.context["dynamic_trace"]
    assert _calls == ["trace"]
    assert evidence["status"] == "complete"
    assert evidence["trace_input_id"] and evidence["trace_run_id"]


def test_overlapping_and_nested_dispatch_traces_keep_request_evidence_isolated(tmp_path):
    barrier = Barrier(2)

    def concurrent_target():
        barrier.wait(timeout=5)

    def nested_target():
        from dryml.code import trace

        trace(traceable_target, context=CodeAnalysisContext(allow_dynamic_execution=True))

    def explain_once(index):
        return Dispatcher(store=_store(tmp_path / str(index))).explain(
            PickledCallable(concurrent_target),
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        ).resolution.dynamic_trace.to_data()

    with ThreadPoolExecutor(max_workers=2) as pool:
        evidence = list(pool.map(explain_once, range(2)))

    assert all(item["trace_input_id"] for item in evidence)
    assert len({item["trace_run_id"] for item in evidence}) == 2
    nested = Dispatcher(store=_store(tmp_path / "nested")).explain(
        PickledCallable(nested_target),
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )
    assert nested.launchable
    assert nested.resolution.dynamic_trace.data["status"] == "complete"


def test_interruption_cleans_pickle_launch_artifact(monkeypatch, tmp_path):
    import dryml.dispatch.planner as planner

    original = planner.normalize_user_operation
    captured = []

    def capture(*args, **kwargs):
        normalized = original(*args, **kwargs)
        captured.append(normalized)
        return normalized

    monkeypatch.setattr(planner, "normalize_user_operation", capture)
    with pytest.raises(KeyboardInterrupt):
        Dispatcher(store=_store(tmp_path)).plan(
            PickledCallable(interrupting_target),
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    assert captured
    assert all(not os.path.exists(path) for path in captured[0].launch["cleanup_paths"])


def test_submit_and_run_trace_only_through_their_single_planning_call(tmp_path):
    class Future:
        def __init__(self, plan):
            self.plan = plan

        def result(self, timeout=None):
            del timeout
            return WorkerResponse(
                "ok",
                operation_id=self.plan.dispatch_spec["payload"]["operation_id"],
                dispatch_id=self.plan.dispatch_spec["id"],
                recipe_id=self.plan.execution_recipe["id"],
            )

    class Backend:
        def submit(self, plan):
            return Future(plan)

    _calls.clear()
    dispatcher = Dispatcher(store=_store(tmp_path), backend=Backend())
    future = dispatcher.submit(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")
    assert isinstance(future, Future)
    assert _calls == ["trace"]

    _calls.clear()
    result = dispatcher.run(traceable_target, analysis_policy={"dynamic_trace": True}, requirement_policy="ignore")
    assert result.status == "ok"
    assert _calls == ["trace"]


def test_plan_world_traces_through_its_single_planning_pipeline(tmp_path):
    _calls.clear()
    plan = Dispatcher(store=_store(tmp_path)).plan_world(
        traceable_target,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
        record_policy="none",
    )

    assert _calls == ["trace"]
    assert plan.dispatch_spec["payload"]["metadata"]["dryml.dispatch.dynamic_trace"]["status"] == "complete"


def test_traced_and_untraced_operation_sidecar_bytes_are_identical(tmp_path):
    store = _store(tmp_path)
    dispatcher = Dispatcher(store=store)

    ordinary = dispatcher.plan(traceable_target, requirement_policy="ignore")
    operation_path = store.records.spec_family_dir("operation") / f"{ordinary.envelope.operation_spec['id']}.json"
    before = operation_path.read_bytes()
    traced = dispatcher.plan(
        traceable_target,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )

    assert traced.envelope.operation_spec["id"] == ordinary.envelope.operation_spec["id"]
    assert operation_path.read_bytes() == before
    assert "dryml.dispatch.dynamic_trace" not in traced.envelope.operation_spec["metadata"]
    assert traced.dispatch_spec["id"] != ordinary.dispatch_spec["id"]


def _pre_execution_provenance_data():
    return {
        "schema": "dryml.dispatch.dynamic_trace.v1",
        "schema_version": 1,
        "requested": True,
        "trace_input_id": "trace-input-v1-test",
        "trace_run_id": None,
        "execution_location": "current_process",
        "execution_started": False,
        "target": {"target_kind": "function", "transport": "import_path"},
        "policy": {"max_calls": 1, "require_proxy_only_args": True, "collect_requirements": True},
        "status": "pre_execution_failed",
        "summary": None,
        "calls": [],
        "accepted_fragments": [],
        "duplicate_observations": [],
        "diagnostics": [{
            "code": "dryml.dispatch.dynamic_trace_unsupported_input",
            "severity": "error",
            "data": {"trace_diagnostic_codes": []},
        }],
    }


def _complete_provenance_data(calls, *, accepted=(), duplicates=()):
    count = len(calls)
    return {
        "schema": "dryml.dispatch.dynamic_trace.v1",
        "schema_version": 1,
        "requested": True,
        "trace_input_id": "trace-input-v1-test",
        "trace_run_id": "trace-run-v1-test",
        "execution_location": "current_process",
        "execution_started": True,
        "target": {"target_kind": "function", "transport": "import_path"},
        "policy": {"max_calls": max(1, count), "require_proxy_only_args": True, "collect_requirements": True},
        "status": "complete",
        "summary": _summary(calls=count, max_calls=max(1, count)).to_data(),
        "calls": [call.to_data() for call in calls],
        "accepted_fragments": list(accepted),
        "duplicate_observations": list(duplicates),
        "diagnostics": [],
    }


def _assert_exact_and_over_provenance_bounds(exact, over):
    for restore in (DynamicTraceProvenance, DynamicTraceProvenance.from_data):
        assert restore(exact).to_data() == exact
        with pytest.raises(ValueError):
            restore(over)


def test_provenance_call_count_exact_bound_and_n_plus_one():
    exact = _complete_provenance_data([_dynamic_call(index) for index in range(256)])
    over = {
        **exact,
        "policy": {**exact["policy"], "max_calls": 257},
        "summary": _summary(calls=257, max_calls=257).to_data(),
        "calls": [*exact["calls"], _dynamic_call(256).to_data()],
    }

    _assert_exact_and_over_provenance_bounds(exact, over)


def _fragment_fact(priority):
    fragment = {
        "namespace": "environment",
        "kind": "requirement",
        "fragment": {},
        "source": {
            "kind": "synthetic",
            "target": None,
            "label": None,
            "namespace": None,
            "path": None,
            "metadata": {},
        },
        "priority": priority,
        "merge_policy": None,
        "schema_version": 1,
    }
    fact = AnnotationFact(
        data=fragment,
        source={"analyzer": "direct_annotations", "target_kind": "function", "annotation_source": fragment["source"]},
    ).to_data()
    key = hashlib.sha256(json.dumps(fragment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return fact, key


def test_provenance_accepted_fragment_count_exact_bound_and_n_plus_one():
    facts_and_keys = [_fragment_fact(index) for index in range(1024)]
    calls = []
    accepted = []
    for sequence in range(4):
        batch = facts_and_keys[sequence * 256:(sequence + 1) * 256]
        call = _dynamic_call(sequence)
        call.data["method_facts"] = [fact for fact, _key in batch]
        calls.append(call)
        accepted.extend(
            {"sequence": sequence, "method": "train", "fragment_key": key}
            for _fact, key in batch
        )
    exact = _complete_provenance_data(calls, accepted=accepted)
    over = {**exact, "accepted_fragments": [*accepted, accepted[0]]}

    _assert_exact_and_over_provenance_bounds(exact, over)


def test_provenance_duplicate_count_exact_bound_and_n_plus_one():
    fact, key = _fragment_fact(0)
    calls = []
    accepted = [{"sequence": 0, "method": "train", "fragment_key": key}]
    duplicates = []
    for sequence, count in enumerate((256, 256, 256, 256, 1)):
        call = _dynamic_call(sequence)
        call.data["method_facts"] = [fact] * count
        calls.append(call)
        duplicates.extend(
            {"sequence": sequence, "method": "train", "fragment_key": key, "first": "trace:0"}
            for _index in range(count - (sequence == 0))
        )
    exact = _complete_provenance_data(calls, accepted=accepted, duplicates=duplicates)
    over = {**exact, "duplicate_observations": [*duplicates, duplicates[0]]}

    _assert_exact_and_over_provenance_bounds(exact, over)


def test_provenance_diagnostic_count_exact_bound_and_n_plus_one():
    diagnostic = _pre_execution_provenance_data()["diagnostics"][0]
    exact = {**_pre_execution_provenance_data(), "diagnostics": [diagnostic.copy() for _index in range(256)]}
    over = {**exact, "diagnostics": [*exact["diagnostics"], diagnostic]}

    _assert_exact_and_over_provenance_bounds(exact, over)


def test_provenance_scalar_length_exact_bound_and_n_plus_one():
    exact = _pre_execution_provenance_data()
    exact["diagnostics"][0]["code"] = "x" * 4096
    over = json.loads(json.dumps(exact))
    over["diagnostics"][0]["code"] += "x"

    _assert_exact_and_over_provenance_bounds(exact, over)


def test_provenance_nesting_depth_exact_bound_and_n_plus_one():
    def nested_argument(layers):
        value = "x"
        for _index in range(layers):
            value = [value]
        return value

    exact_call = _dynamic_call(0)
    exact_call.data["args"] = [nested_argument(27)]
    over_call = _dynamic_call(0)
    over_call.data["args"] = [nested_argument(28)]

    _assert_exact_and_over_provenance_bounds(
        _complete_provenance_data([exact_call]),
        _complete_provenance_data([over_call]),
    )


def test_provenance_canonical_json_bytes_exact_bound_and_n_plus_one():
    exact = _pre_execution_provenance_data()
    codes = ["x"] * 256
    exact["diagnostics"][0]["data"]["trace_diagnostic_codes"] = codes
    remaining = 1_048_576 - len(json.dumps(exact, sort_keys=True, separators=(",", ":")).encode())
    for index in range(len(codes)):
        growth = min(4095, remaining)
        codes[index] += "x" * growth
        remaining -= growth
    assert remaining == 0
    assert len(json.dumps(exact, sort_keys=True, separators=(",", ":")).encode()) == 1_048_576

    over = json.loads(json.dumps(exact))
    adjustable = next(index for index, code in enumerate(over["diagnostics"][0]["data"]["trace_diagnostic_codes"]) if len(code) < 4096)
    over["diagnostics"][0]["data"]["trace_diagnostic_codes"][adjustable] += "x"

    _assert_exact_and_over_provenance_bounds(exact, over)


@pytest.mark.parametrize("target", [ConstructorSentinel, CallableSentinel(), len, generator_target, async_target])
def test_unsupported_live_target_rows_do_not_construct_or_invoke(target, tmp_path):
    try:
        explanation = Dispatcher(store=_store(tmp_path)).explain(
            target,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )
    except DispatchPlanningError:
        # Non-importable callable instances retain ordinary normalization
        # rejection; trace must not make them dispatchable.
        return

    assert not explanation.launchable
    assert explanation.resolution.dynamic_trace.data["status"] == "pre_execution_failed"


@pytest.mark.parametrize("factory", [InspectionHookCallable, lambda: InspectionHookCallable, DescriptorSentinel])
def test_opt_in_unsupported_targets_are_rejected_before_generic_inspection(factory, tmp_path):
    target = factory()
    with pytest.raises(DispatchPlanningError, match="exact synchronous Python function"):
        Dispatcher(store=_store(tmp_path)).explain(
            target,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )


def test_explicit_pickle_unsupported_callable_is_rejected_before_serialization(tmp_path):
    with pytest.raises(DispatchPlanningError, match="exact synchronous Python function"):
        Dispatcher(store=_store(tmp_path)).explain(
            PickledCallable(InspectionHookCallable()),
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )


def test_run_world_traces_only_through_its_planning_call(tmp_path):
    _calls.clear()
    Dispatcher(store=_store(tmp_path)).run_world(
        traceable_target,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
        record_policy="none",
        timeout=60,
    )

    assert _calls == ["trace"]


def test_provenance_correlates_duplicate_fragments_across_methods_and_same_call_order():
    fragment = dryml.annotations.fragments_for(TraceRequirementModel)[0]
    fragment_data = fragment.to_data()
    method_fact = AnnotationFact(
        data=fragment_data,
        source={
            "analyzer": "direct_annotations",
            "target_kind": "function",
            "annotation_source": fragment_data["source"],
        },
    ).to_data()
    first = _dynamic_call(0)
    first.data.update({"method_name": "prepare", "method_facts": [method_fact, method_fact]})
    later = _dynamic_call(1)
    later.data.update({"method_name": "train", "method_facts": [method_fact]})
    fragment_key = hashlib.sha256(
        json.dumps(fragment.to_data(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    provenance = DynamicTraceProvenance({
        "schema": "dryml.dispatch.dynamic_trace.v1",
        "schema_version": 1,
        "requested": True,
        "trace_input_id": "trace-input-v1-test",
        "trace_run_id": "trace-run-v1-test",
        "execution_location": "current_process",
        "execution_started": True,
        "target": {"target_kind": "function", "transport": "import_path"},
        "policy": {"max_calls": 2, "require_proxy_only_args": True, "collect_requirements": True},
        "status": "complete",
        "summary": _summary(calls=2, max_calls=2).to_data(),
        "calls": [first.to_data(), later.to_data()],
        "accepted_fragments": [{"sequence": 0, "method": "prepare", "fragment_key": fragment_key}],
        "duplicate_observations": [
            {"sequence": 0, "method": "prepare", "fragment_key": fragment_key, "first": "trace:0"},
            {"sequence": 1, "method": "train", "fragment_key": fragment_key, "first": "trace:0"},
        ],
        "diagnostics": [],
    })

    assert provenance.to_data()["duplicate_observations"][1]["first"] == "trace:0"


def test_full_v1_calls_are_preserved_in_every_carrier(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    call = _dynamic_call(0)

    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=(call, _summary(calls=1)),
        ),
    )
    store = _store(tmp_path)
    plan = Dispatcher(store=store).plan(
        traceable_target,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )

    expected_calls = [call.to_data()]
    carriers = (
        plan.resolution.dynamic_trace.to_data(),
        plan.dispatch_spec["payload"]["metadata"]["dryml.dispatch.dynamic_trace"],
        plan.execution_recipe["payload"]["annotation_report"]["dryml.dispatch.dynamic_trace"],
        plan.envelope.reporting["planning"]["dryml.dispatch.dynamic_trace"],
        store.records.read_spec(plan.dispatch_spec["id"], family="dispatch")["payload"]["metadata"]["dryml.dispatch.dynamic_trace"],
        store.records.read_spec(plan.execution_recipe["id"], family="execution_recipe")["payload"]["annotation_report"]["dryml.dispatch.dynamic_trace"],
    )
    assert [json_ready(carrier)["calls"] for carrier in carriers] == [expected_calls] * len(carriers)


def test_safe_semantic_metadata_is_unchanged_for_authoritative_resolution():
    import dryml.dispatch.requirements as requirements

    fragments = []
    for mode in ("base", "override"):
        fragment_data = dryml.annotations.fragments_for(TraceRequirementModel)[0].to_data()
        fragment_data["source"]["metadata"] = {"legacy_environment_fragment_mode": mode}
        fragments.append(fragment_data)
    call = _dynamic_call(0)
    call.data["method_facts"] = [
        AnnotationFact(
            data=fragment,
            source={
                "analyzer": "direct_annotations",
                "target_kind": "function",
                "annotation_source": fragment["source"],
            },
        ).to_data()
        for fragment in fragments
    ]

    requirements._validate_trace_call_persistence([call])
    combined, accepted, duplicates = requirements._combine_trace_fragments((), [call])
    expected = dryml.annotations.resolve_fragments(
        tuple(dryml.annotations.AnnotationFragment.from_data(fragment) for fragment in fragments),
        source="dryml.dispatch.dynamic_trace",
    )
    actual = dryml.annotations.resolve_fragments(combined, source="dryml.dispatch.dynamic_trace")

    assert actual.to_data() == expected.to_data()
    assert [item.source.metadata["legacy_environment_fragment_mode"] for item in combined] == ["base", "override"]
    assert len(accepted) == 2
    assert duplicates == []


@pytest.mark.parametrize("private_request", [False, True])
def test_exported_resolver_does_not_trace_ordinarily_normalized_callable(monkeypatch, private_request):
    import dryml.dispatch.requirements as requirements

    _calls.clear()
    normalized = normalize_user_operation(traceable_target, args=[])
    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not trace")),
    )
    kwargs = {"requirement_policy": "ignore"}
    if private_request:
        kwargs["_analysis_request"] = parse_analysis_policy({"dynamic_trace": True})
    else:
        kwargs["analysis_policy"] = {"dynamic_trace": True}

    resolution = dryml.dispatch.resolve_dispatch_plan(normalized, **kwargs)

    assert normalized.trace_live_target is None
    assert not resolution.launchable
    assert resolution.dynamic_trace.data["status"] == "pre_execution_failed"
    assert _calls == []


def test_direct_callable_normalizer_cannot_create_trace_ready_target(monkeypatch):
    import dryml.dispatch.requirements as requirements

    _calls.clear()
    normalized = normalize_callable_operation(traceable_target)
    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not trace")),
    )

    resolution = resolve_dispatch_plan(
        normalized,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )

    assert normalized.trace_live_target is None
    assert not resolution.launchable
    assert resolution.dynamic_trace.data["status"] == "pre_execution_failed"
    assert _calls == []


def test_explanation_serialization_preserves_valid_65_call_trace_provenance(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    result = CodeAnalysisResult(
        target_from_callable(traceable_target).spec,
        facts=tuple(_dynamic_call(index) for index in range(65)) + (_summary(calls=65, max_calls=65),),
    )
    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=result.facts,
        ),
    )

    explanation = Dispatcher(store=_store(tmp_path)).explain(
        traceable_target,
        analysis_policy={"dynamic_trace": DynamicTracePolicy(max_calls=65)},
        requirement_policy="ignore",
    )

    assert len(explanation.to_data()["resolution"]["dynamic_trace"]["calls"]) == 65


def test_over_limit_recognized_preexecution_diagnostics_use_bounded_failure_carrier(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements

    diagnostics = tuple(
        DiagnosticFact(
            severity="error",
            code="dryml.code.dynamic_trace_unsupported_argument",
            message="must not be projected",
        )
        for _ in range(257)
    )
    monkeypatch.setattr(
        requirements,
        "trace",
        lambda *_args, **kwargs: CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            diagnostics=diagnostics,
        ),
    )

    with pytest.raises(DispatchPlanningError) as excinfo:
        Dispatcher(store=_store(tmp_path)).plan(
            traceable_target,
            analysis_policy={"dynamic_trace": True},
            requirement_policy="ignore",
        )

    evidence = excinfo.value.context["dynamic_trace"]
    assert evidence["status"] == "pre_execution_failed"
    assert evidence["diagnostics"] == [{
        "code": "dryml.dispatch.dynamic_trace_provenance_limit_exceeded",
        "severity": "error",
        "data": {"trace_diagnostic_codes": []},
    }]


@pytest.mark.parametrize("entrypoint", ["plan", "plan_world", "explain"])
def test_entrypoints_use_one_validated_policy_snapshot_through_resolution(monkeypatch, tmp_path, entrypoint):
    import dryml.dispatch.planner as planner

    policy = {"dynamic_trace": True}
    original = planner.normalize_user_operation

    def mutate_policy(*args, **kwargs):
        policy.clear()
        return original(*args, **kwargs)

    monkeypatch.setattr(planner, "normalize_user_operation", mutate_policy)
    _calls.clear()
    dispatcher = Dispatcher(store=_store(tmp_path))
    if entrypoint == "plan":
        dispatcher.plan(traceable_target, analysis_policy=policy, requirement_policy="ignore")
    elif entrypoint == "plan_world":
        dispatcher.plan_world(
            traceable_target,
            analysis_policy=policy,
            requirement_policy="ignore",
            record_policy="none",
        )
    else:
        dispatcher.explain(traceable_target, analysis_policy=policy, requirement_policy="ignore")

    assert policy == {}
    assert _calls == ["trace"]


def test_explicit_none_analysis_policy_context_is_not_omission(monkeypatch, tmp_path):
    with pytest.raises(DispatchPlanningError, match="context"):
        parse_analysis_policy({"context": None})
    import dryml.dispatch.planner as planner

    monkeypatch.setattr(
        planner,
        "normalize_user_operation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not normalize")),
    )
    with pytest.raises(DispatchPlanningError, match="context"):
        Dispatcher(store=_store(tmp_path)).plan(traceable_target, analysis_policy={"context": None})


@pytest.mark.parametrize("entrypoint", ["plan", "explain"])
def test_context_metadata_snapshot_survives_mutation_during_normalization_and_before_trace(monkeypatch, tmp_path, entrypoint):
    """One request uses its validation-time context snapshot across trace phases."""

    import dryml.dispatch.planner as planner
    import dryml.dispatch.requirements as requirements

    caller_context = CodeAnalysisContext(metadata={"variant": "initial", "nested": {"phase": "initial"}})
    observed_discovery = []
    observed_trace = []
    original_normalize = planner.normalize_user_operation
    original_analyze = requirements.analyze
    original_input_id = requirements._trace_input_id

    def mutate_during_normalization(*args, **kwargs):
        caller_context.metadata["variant"] = "mutated-after-validation"
        caller_context.metadata["nested"]["phase"] = "normalization"
        return original_normalize(*args, **kwargs)

    def analyze_spy(*args, **kwargs):
        observed_discovery.append(json.loads(json.dumps(kwargs["context"].metadata)))
        return original_analyze(*args, **kwargs)

    def mutate_between_identity_and_trace(*args, **kwargs):
        result = original_input_id(*args, **kwargs)
        caller_context.metadata["nested"]["phase"] = "before-trace"
        return result

    def trace_spy(*_args, **kwargs):
        observed_trace.append(json.loads(json.dumps(kwargs["context"].metadata)))
        return CodeAnalysisResult(
            target_from_callable(traceable_target, metadata=kwargs["context"].metadata).spec,
            facts=(_summary(),),
        )

    monkeypatch.setattr(planner, "normalize_user_operation", mutate_during_normalization)
    monkeypatch.setattr(requirements, "analyze", analyze_spy)
    monkeypatch.setattr(requirements, "_trace_input_id", mutate_between_identity_and_trace)
    monkeypatch.setattr(requirements, "trace", trace_spy)

    result = getattr(Dispatcher(store=_store(tmp_path)), entrypoint)(
        traceable_target,
        analysis_policy={"context": caller_context, "dynamic_trace": True},
        requirement_policy="ignore",
    )

    assert observed_discovery == [{"variant": "initial", "nested": {"phase": "initial"}}]
    assert observed_trace == [{
        "variant": "initial",
        "nested": {"phase": "initial"},
        "_dryml_dispatch_trace_input_id": result.resolution.dynamic_trace.data["trace_input_id"],
        "_dryml_dispatch_trace_run_id": result.resolution.dynamic_trace.data["trace_run_id"],
    }]
    assert caller_context.metadata["nested"]["phase"] == "before-trace"
    assert result.resolution.dynamic_trace.data["status"] == "complete"
