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
from dryml.dispatch.normalize import normalize_user_operation
from dryml.dispatch.requirements import DynamicTraceProvenance, _effective_trace_invocation, parse_analysis_policy, resolve_dispatch_plan
from dryml.operations import resolve_call_arguments
from dryml.core2.utils.general import pickle_save
from dryml.formats.refs import format_cdef_id
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

    result = CodeAnalysisResult(
        target_from_callable(traceable_target).spec,
        facts=tuple(_dynamic_call(index) for index in range(257)) + (_summary(calls=257),),
    )
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: result)

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
    result = CodeAnalysisResult(
        target_from_callable(traceable_target).spec,
        facts=tuple(calls) + (_summary(calls=256),),
    )
    monkeypatch.setattr(requirements, "trace", lambda *_args, **_kwargs: result)

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


def test_provenance_omits_admitted_call_arguments_from_all_carriers(monkeypatch, tmp_path):
    """Accepted call arguments stay transient and never reach provenance copies."""

    import dryml.dispatch.requirements as requirements

    call = _dynamic_call(0)
    call.data["args"] = ["synthetic-positional-secret"]
    call.data["kwargs"] = {"api_key": "synthetic-secret-token"}

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
    plan = dispatcher.plan(
        traceable_target,
        analysis_policy={"dynamic_trace": True},
        requirement_policy="ignore",
    )

    assert call.data["args"] == ["synthetic-positional-secret"]
    assert call.data["kwargs"] == {"api_key": "synthetic-secret-token"}
    carriers = (
        explanation.to_data()["resolution"]["dynamic_trace"],
        plan.resolution.dynamic_trace.to_data(),
        plan.dispatch_spec["payload"]["metadata"]["dryml.dispatch.dynamic_trace"],
        plan.execution_recipe["payload"]["annotation_report"]["dryml.dispatch.dynamic_trace"],
        plan.envelope.reporting["planning"]["dryml.dispatch.dynamic_trace"],
        store.records.read_spec(plan.dispatch_spec["id"], family="dispatch")["payload"]["metadata"]["dryml.dispatch.dynamic_trace"],
        store.records.read_spec(plan.execution_recipe["id"], family="execution_recipe")["payload"]["annotation_report"]["dryml.dispatch.dynamic_trace"],
    )
    for carrier in carriers:
        assert not carrier["calls"][0]["data"]["args"]
        assert not carrier["calls"][0]["data"]["kwargs"]
        assert "synthetic-positional-secret" not in repr(carrier)
        assert "synthetic-secret-token" not in repr(carrier)


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


def test_provenance_rejects_each_top_level_count_and_scalar_overflow():
    base = {
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
    overlong = {**base, "diagnostics": [{"code": "x" * 4097, "severity": "error"}]}
    with pytest.raises(ValueError):
        DynamicTraceProvenance(overlong)
    with pytest.raises(ValueError):
        DynamicTraceProvenance({**base, "diagnostics": base["diagnostics"] * 257})


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
