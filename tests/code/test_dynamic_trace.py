from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

import dryml
import dryml.code as code
from dryml.core2 import ConcreteDefinition, Definition
from dryml.core2.dtype import DType
from dryml.core2.methods import Method, Traits, traits
from dryml.core2.object import Object
from dryml.core2.symbol import ImportRef, SourceSpec


@dryml.env.req(requirements=("trace-class>=1",))
class TraceModel:
    built = False
    called = False

    def __init__(self):
        type(self).built = True
        raise AssertionError("trace must not build receivers")

    @dryml.env.req(requirements=("trace-method>=1",))
    def train(self, value=None):
        type(self).called = True
        raise AssertionError("trace must not invoke real methods")

    def build(self):
        raise AssertionError("trace proxy must not expose build")

    @staticmethod
    def static(value=None):
        raise AssertionError("trace must not invoke static methods")

    @classmethod
    def class_method(cls, value=None):
        raise AssertionError("trace must not invoke class methods")

    @property
    def property_value(self):
        raise AssertionError("trace must not invoke properties")

    value = 3


@dryml.env.req(requirements=("x" * 4_097,))
class OversizedAnnotationTraceModel:
    def train(self):
        raise AssertionError("trace must not invoke real methods")


class TraceMethodModel(Method):
    @traits(backend=None)
    def run(self):
        raise AssertionError("trace must not invoke Method subclasses")


def module_trace_target(model):
    model.train()


class UnboundOrchestration:
    def run(model):
        model.train()


def _enabled(**kwargs):
    return code.CodeAnalysisContext(allow_dynamic_execution=True, **kwargs)


def _summary(result):
    return result.facts_of_kind("dynamic_trace_summary")[0]


def _assign_item(value):
    value[0] = 1


def _use_context_manager(value):
    with value:
        pass


def _await_value(value):
    import asyncio

    async def use():
        await value

    asyncio.run(use())


def _async_iterate(value):
    import asyncio

    async def use():
        async for _ in value:
            pass

    asyncio.run(use())


def _use_async_context_manager(value):
    import asyncio

    async def use():
        async with value:
            pass

    asyncio.run(use())


def test_policy_validation_is_strict():
    assert code.DynamicTracePolicy().max_calls == 10_000
    assert code.DynamicTracePolicy(max_calls=10_000).max_calls == 10_000
    for value in (True, 1.0, "1"):
        with pytest.raises(TypeError):
            code.DynamicTracePolicy(max_calls=value)
    for value in (0, -1, 10_001):
        with pytest.raises(ValueError):
            code.DynamicTracePolicy(max_calls=value)
    with pytest.raises(TypeError):
        code.DynamicTracePolicy(require_proxy_only_args=1)
    with pytest.raises(TypeError):
        code.DynamicTracePolicy(collect_requirements=1)


def test_disabled_trace_does_not_resolve_import_path():
    result = code.trace("module_that_must_not_be_imported_for_disabled_trace:target")
    assert result.diagnostics_of_code("dryml.code.dynamic_trace_disabled")
    assert not result.facts


def test_facade_types_and_context_fail_before_execution():
    called = []

    def target():
        called.append(True)

    for call in (
        lambda: code.trace(target, args=[], context=_enabled()),
        lambda: code.trace(target, kwargs=[], context=_enabled()),
        lambda: code.trace(target, kwargs={1: None}, context=_enabled()),
        lambda: code.trace(target, context=object()),
        lambda: code.trace(target, policy={}, context=_enabled()),
    ):
        with pytest.raises(TypeError):
            call()
    result = code.trace(target, context=code.CodeAnalysisContext(allow_dynamic_execution=True, algorithms=("source",)))
    assert result.diagnostics_of_code("dryml.code.dynamic_trace_invalid_context")
    assert called == []


def test_unhashable_trace_algorithm_selection_fails_before_execution():
    called = []

    def target():
        called.append(True)

    result = code.trace(
        target,
        context=code.CodeAnalysisContext(
            allow_dynamic_execution=True,
            algorithms=([],),
        ),
    )

    assert result.diagnostics_of_code("dryml.code.dynamic_trace_invalid_context")
    assert called == []


def test_supported_live_targets_execute_once_and_import_path_works():
    calls = []

    def local():
        calls.append("local")

    closure_value = "closure"

    def closure():
        calls.append(closure_value)

    for target in (local, closure, lambda: calls.append("lambda"), code.CodeTarget(code.CodeTargetSpec("function"), obj=local)):
        result = code.trace(target, context=_enabled())
        assert result.ok
        assert _summary(result).data["outcome"] == "complete"
    imported = code.trace(f"{__name__}:module_trace_target", args=(Definition(TraceModel),), context=_enabled())
    assert imported.ok
    assert imported.facts_of_kind("dynamic_call")
    assert calls == ["local", "closure", "lambda", "local"]


def test_unbound_method_function_runs_as_an_ordinary_function():
    result = code.trace(
        UnboundOrchestration.run,
        args=(Definition(TraceModel),),
        context=_enabled(),
    )
    assert result.ok
    assert len(result.facts_of_kind("dynamic_call")) == 1


def test_enabled_import_path_respects_import_permission_and_redacts_failures(monkeypatch):
    denied = code.trace(f"{__name__}:module_trace_target", args=(Definition(TraceModel),), context=_enabled(allow_import=False))
    assert denied.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_target")

    import dryml.code.targets as targets

    def fail_import(name):
        raise RuntimeError("IMPORT-SECRET-MUST-NOT-LEAK")

    monkeypatch.setattr(targets.importlib, "import_module", fail_import)
    failed = code.trace("missing.module:target", context=_enabled())
    encoded = json.dumps(failed.to_data())
    assert failed.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_target")
    assert "IMPORT-SECRET-MUST-NOT-LEAK" not in encoded


@pytest.mark.parametrize(
    "target",
    [
        TraceModel().train if False else object(),
        TraceModel,
        len,
        (lambda: None)(),
    ],
)
def test_unsupported_nonfunction_targets_do_not_execute(target):
    result = code.trace(target, context=_enabled())
    assert result.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_target")
    assert not result.facts


def test_bound_callable_async_and_generator_targets_are_rejected_without_execution():
    called = []

    class Callable:
        def __call__(self):
            called.append("callable")

    class Owner:
        def method(self):
            called.append("bound")

    async def async_target():
        called.append("async")

    async def async_generator():
        called.append("async_generator")
        yield 1

    def generator():
        called.append("generator")
        yield 1

    for target in (Callable(), Owner().method, async_target, async_generator, generator):
        result = code.trace(target, context=_enabled())
        assert result.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_target")
    assert called == []


def test_source_spec_only_target_is_rejected():
    target = code.CodeTargetSpec("source_spec", source_spec=SourceSpec.from_source("lambda: None").__stable_leaf_bytes__().decode())
    result = code.trace(target, context=_enabled())
    assert result.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_target")


def test_analyze_dynamic_trace_is_non_invoking_and_requires_facade():
    called = []

    def target():
        called.append(True)

    assert code.analyze(target).ok
    selected = code.analyze(target, algorithms=("dynamic_trace",))
    assert selected.diagnostics_of_code("dryml.code.dynamic_trace_requires_trace_facade")
    assert called == []


def test_inline_and_worker_probe_selection_remain_non_invoking(requirement_targets):
    inline = code.probe_target(
        requirement_targets.plain_importable_function,
        algorithms=("dynamic_trace",),
        include_environment_record=False,
    )
    worker = code.probe_target(
        "dryml_requirement_targets:plain_importable_function",
        algorithms=("dynamic_trace",),
        include_environment_record=False,
        timeout=5,
    )
    assert inline.analysis.diagnostics_of_code("dryml.code.dynamic_trace_requires_trace_facade")
    assert worker.analysis.diagnostics_of_code("dryml.code.dynamic_trace_requires_trace_facade")


def test_definition_and_cdef_calls_record_without_build_or_real_method():
    TraceModel.built = False
    TraceModel.called = False

    def target(defn, cdef):
        defn.train(cdef, nested={"items": [defn]})
        cdef.static()
        cdef.class_method()

    result = code.trace(target, args=(Definition(TraceModel), ConcreteDefinition(TraceModel)), context=_enabled())
    calls = result.facts_of_kind("dynamic_call")
    assert result.ok
    assert [fact.data["sequence"] for fact in calls] == [0, 1, 2]
    assert [fact.data["receiver_kind"] for fact in calls] == ["definition", "concrete_definition", "concrete_definition"]
    assert calls[0].data["args"][0]["definition_kind"] == "concrete_definition"
    assert calls[0].data["kwargs"]["nested"]["items"][0]["definition_kind"] == "definition"
    assert calls[0].data["receiver_class"] == f"{__name__}:TraceModel"
    assert TraceModel.built is False
    assert TraceModel.called is False
    json.dumps(result.to_data())


def test_trace_rejects_identity_subclass_leaf_hook_before_invocation():
    class HookDType(DType):
        hook_called = False

        def __stable_leaf_bytes__(self):
            type(self).hook_called = True
            return b"unbounded-custom-leaf"

    executed = []

    def target(model):
        executed.append(True)
        model.train()

    result = code.trace(
        target,
        args=(Definition(TraceModel, dtype=HookDType("float", 32)),),
        context=_enabled(),
    )

    assert result.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_argument")
    assert HookDType.hook_called is False
    assert executed == []


def test_current_annotation_facts_are_attached_and_switchable():
    result = code.trace(module_trace_target, args=(Definition(TraceModel),), context=_enabled())
    method_facts = result.facts_of_kind("dynamic_call")[0].data["method_facts"]
    requirements = [fact for fact in method_facts if fact["kind"] == "requirement"]
    assert len(requirements) == 2
    assert all("resolution" in fact["data"] for fact in requirements)
    restored = code.CodeAnalysisResult.from_data(json.loads(json.dumps(result.to_data())))
    assert isinstance(restored.facts_of_kind("dynamic_call")[0], code.DynamicCallFact)
    assert restored.facts_of_kind("dynamic_call")[0].data["method_facts"] == method_facts

    no_requirements = code.trace(
        module_trace_target,
        args=(Definition(TraceModel),),
        context=_enabled(),
        policy=code.DynamicTracePolicy(collect_requirements=False),
    )
    assert no_requirements.facts_of_kind("dynamic_call")[0].data["method_facts"] == []
    no_annotations = code.trace(
        module_trace_target,
        args=(Definition(TraceModel),),
        context=_enabled(include_annotations=False, include_method_contracts=False),
    )
    assert no_annotations.facts_of_kind("dynamic_call")[0].data["method_facts"] == []


def test_legacy_compute_spec_metadata_is_ignored():
    sentinel = object()
    original = getattr(TraceModel.train, "__dry_compute_spec__", sentinel)
    TraceModel.train.__dry_compute_spec__ = type(
        "LegacySpec",
        (),
        {"compute_reqs": {"must": "not appear"}},
    )()
    try:
        result = code.trace(
            module_trace_target,
            args=(Definition(TraceModel),),
            context=_enabled(),
        )
    finally:
        if original is sentinel:
            del TraceModel.train.__dry_compute_spec__
        else:
            TraceModel.train.__dry_compute_spec__ = original

    encoded = json.dumps(result.to_data())
    assert result.ok
    assert "compute_reqs" not in encoded
    assert "must" not in encoded


def test_applicable_core_method_contract_fact_is_typed_and_repr_redacted(monkeypatch):
    secret = "TRAIT-REPR-SECRET-MUST-NOT-LEAK"
    monkeypatch.setattr(Traits, "__repr__", lambda self: secret)
    result = code.trace(
        lambda method: method.run(),
        args=(Definition(TraceMethodModel),),
        context=_enabled(include_annotations=False),
    )
    method_facts = result.facts_of_kind("dynamic_call")[0].data["method_facts"]
    assert result.ok
    assert [fact["kind"] for fact in method_facts] == ["method_contract"]
    assert method_facts[0]["data"]["trait_impls"] == [{
        "name": "run",
        "traits": {"backend": None, "batch_mode": None},
    }]
    assert secret not in json.dumps(result.to_data())


def test_property_nonmethod_missing_and_dunder_attributes_are_diagnostic_without_hooks():
    for access in (
        lambda model: model.property_value(),
        lambda model: model.value(),
        lambda model: model.missing(),
        lambda model: model.build(),
        lambda model: model.__class__(),
    ):
        result = code.trace(access, args=(Definition(TraceModel),), context=_enabled())
        assert result.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_receiver_attribute")
        assert _summary(result).data["outcome"] == "unsupported_receiver_attribute"


def test_method_name_exact_boundary_and_limit_metadata():
    accepted_name = "m" * 512
    rejected_name = "m" * 513

    def method(self):
        raise AssertionError("trace must not invoke dynamically named methods")

    model_type = type(
        "LongMethodModel",
        (),
        {accepted_name: method, rejected_name: method},
    )
    context = _enabled(include_annotations=False, include_method_contracts=False)
    accepted = code.trace(
        lambda model: getattr(model, accepted_name)(),
        args=(Definition(model_type),),
        context=context,
    )
    assert accepted.ok
    assert accepted.facts_of_kind("dynamic_call")[0].data["method_name"] == accepted_name
    assert accepted.facts_of_kind("dynamic_call")[0].data["receiver_class"] is None

    rejected = code.trace(
        lambda model: getattr(model, rejected_name)(),
        args=(Definition(model_type),),
        context=context,
    )
    diagnostic = rejected.diagnostics_of_code(
        "dryml.code.dynamic_trace_unsupported_receiver_attribute"
    )[0]
    assert diagnostic.data == {
        "limit_name": "method_name_chars",
        "limit": 512,
        "observed_lower_bound": 513,
    }
    assert _summary(rejected).data["complete"] is False


def test_custom_descriptor_and_metaclass_hooks_are_not_invoked():
    events = []

    class Meta(type):
        def __getattribute__(cls, name):
            if name == "descriptor":
                events.append("metaclass")
            return super().__getattribute__(name)

    class Descriptor:
        def __get__(self, instance, owner):
            events.append("descriptor")
            raise AssertionError("descriptor must not bind")

    class Model(metaclass=Meta):
        descriptor = Descriptor()

    result = code.trace(lambda model: model.descriptor(), args=(Definition(Model),), context=_enabled())
    assert result.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_receiver_attribute")
    assert events == []


def test_invocation_grammar_aliases_identity_and_caller_containers():
    definition = Definition(TraceModel)
    shared = [definition]
    outer = [shared, shared]
    observed = []

    def target(value, same):
        observed.extend((value is same, value[0] is value[1], value[0][0] is same[0][0]))
        value[0][0].train()

    result = code.trace(target, args=(outer, outer), context=_enabled())
    assert result.ok
    assert observed == [True, True, True]
    assert outer == [shared, shared]
    assert outer[0][0] is definition


def test_scalar_invocation_requires_opt_in_and_remains_bounded():
    called = []

    def target(*values, **kwargs):
        called.append((values, kwargs))

    rejected = code.trace(target, args=(None,), context=_enabled())
    assert rejected.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_argument")
    accepted = code.trace(
        target,
        args=(None, True, 3, 1.5, "x"),
        kwargs={"value": 4},
        context=_enabled(),
        policy=code.DynamicTracePolicy(require_proxy_only_args=False),
    )
    assert accepted.ok
    assert len(called) == 1
    for value in (float("nan"), float("inf"), b"x", {1}, object()):
        result = code.trace(
            target,
            args=(value,),
            context=_enabled(),
            policy=code.DynamicTracePolicy(require_proxy_only_args=False),
        )
        assert result.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_argument")
    oversized = code.trace(
        target,
        args=("x" * 4_097,),
        context=_enabled(),
        policy=code.DynamicTracePolicy(require_proxy_only_args=False),
    )
    assert oversized.diagnostics_of_code("dryml.code.dynamic_trace_argument_limit_exceeded")


def test_unsupported_invocation_categories_do_not_execute_hooks_or_target():
    events = []

    class CustomMapping(Mapping):
        def __iter__(self):
            events.append("mapping_iter")
            raise AssertionError("custom mappings must not be inspected")

        def __len__(self):
            events.append("mapping_len")
            raise AssertionError("custom mappings must not be inspected")

        def __getitem__(self, key):
            events.append("mapping_getitem")
            raise AssertionError("custom mappings must not be inspected")

    class CustomSequence(Sequence):
        def __len__(self):
            events.append("sequence_len")
            raise AssertionError("custom sequences must not be inspected")

        def __getitem__(self, key):
            events.append("sequence_getitem")
            raise AssertionError("custom sequences must not be inspected")

    @dataclass
    class Record:
        value: int

    class DryObject(Object):
        pass

    def iterator():
        events.append("generator_iterated")
        yield 1

    def target(value):
        events.append("target")

    values = (
        b"bytes",
        bytearray(b"bytes"),
        {1},
        frozenset({1}),
        range(1),
        iter((1,)),
        iterator(),
        CustomMapping(),
        CustomSequence(),
        Record(1),
        np.int64(1),
        object.__new__(DryObject),
        object(),
    )
    policy = code.DynamicTracePolicy(require_proxy_only_args=False)
    for value in values:
        result = code.trace(
            target,
            args=(value,),
            context=_enabled(),
            policy=policy,
        )
        assert result.diagnostics_of_code(
            "dryml.code.dynamic_trace_unsupported_argument"
        )
    assert events == []


def test_empty_exact_containers_are_valid_in_proxy_only_mode():
    observed = []

    def target(value, *, mapping):
        observed.append((value, mapping))

    result = code.trace(
        target,
        args=(([], ()),),
        kwargs={"mapping": {}},
        context=_enabled(),
    )
    assert result.ok
    assert observed == [(([], ()), {})]


def test_cycle_and_non_string_key_fail_before_execution():
    called = []

    def target(value):
        called.append(value)

    cycle = []
    cycle.append(cycle)
    for value in (cycle, {1: Definition(TraceModel)}):
        result = code.trace(target, args=(value,), context=_enabled())
        assert result.diagnostics
    assert called == []


def test_invocation_depth_entry_string_and_integer_exact_boundaries():
    definition = Definition(TraceModel)

    def nested(depth):
        value = definition
        for _ in range(depth):
            value = [value]
        return value

    assert code.trace(lambda value: None, args=(nested(31),), context=_enabled()).ok
    depth_failure = code.trace(lambda value: None, args=(nested(32),), context=_enabled())
    assert depth_failure.diagnostics_of_code("dryml.code.dynamic_trace_argument_limit_exceeded")

    assert code.trace(lambda value: None, args=([definition] * 9_999,), context=_enabled()).ok
    entry_failure = code.trace(lambda value: None, args=([definition] * 10_000,), context=_enabled())
    assert entry_failure.diagnostics_of_code("dryml.code.dynamic_trace_argument_limit_exceeded")

    scalar_policy = code.DynamicTracePolicy(require_proxy_only_args=False)
    assert code.trace(lambda value: None, args=("x" * 4_096,), context=_enabled(), policy=scalar_policy).ok
    assert code.trace(lambda value: None, args=(1 << 4_095,), context=_enabled(), policy=scalar_policy).ok
    integer_failure = code.trace(lambda value: None, args=(1 << 4_096,), context=_enabled(), policy=scalar_policy)
    assert integer_failure.diagnostics_of_code("dryml.code.dynamic_trace_argument_limit_exceeded")


def test_observed_call_entry_boundary_and_unsupported_value_abort_current_call():
    quiet_context = _enabled(include_annotations=False, include_method_contracts=False)
    complete = code.trace(
        lambda model: model.train([None] * 9_999),
        args=(Definition(TraceModel),),
        context=quiet_context,
    )
    assert complete.ok and len(complete.facts_of_kind("dynamic_call")) == 1
    exceeded = code.trace(
        lambda model: model.train([None] * 10_000),
        args=(Definition(TraceModel),),
        context=quiet_context,
    )
    assert exceeded.diagnostics_of_code("dryml.code.dynamic_trace_argument_limit_exceeded")
    assert not exceeded.facts_of_kind("dynamic_call")
    unsupported = code.trace(
        lambda model: model.train(object()),
        args=(Definition(TraceModel),),
        context=quiet_context,
    )
    assert unsupported.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_argument")
    assert _summary(unsupported).data["outcome"] == "unsupported_argument"


def test_live_definition_created_outside_invocation_is_flattened_in_observed_call():
    external = ConcreteDefinition(TraceModel)
    result = code.trace(
        lambda model: model.train({"external": external}),
        args=(Definition(TraceModel),),
        context=_enabled(include_annotations=False, include_method_contracts=False),
    )
    observed = result.facts_of_kind("dynamic_call")[0].data["args"][0]["external"]
    assert observed["definition_kind"] == "concrete_definition"
    assert observed["definition_ref"].startswith("cdef-v4-")


def test_observed_ordinary_definition_shaped_dictionary_round_trips():
    ordinary = {"definition_kind": "ordinary", "definition_ref": "ordinary"}
    result = code.trace(
        lambda model: model.train(ordinary),
        args=(Definition(TraceModel),),
        context=_enabled(include_annotations=False, include_method_contracts=False),
    )

    assert result.ok
    assert result.facts_of_kind("dynamic_call")[0].data["args"] == [ordinary]
    restored = code.CodeAnalysisResult.from_data(json.loads(json.dumps(result.to_data())))
    assert restored.facts_of_kind("dynamic_call")[0].data["args"] == [ordinary]


def test_call_limit_retains_prior_facts_and_aborts_target():
    continued = []

    def target(model):
        model.train()
        model.train()
        continued.append(True)

    result = code.trace(
        target,
        args=(Definition(TraceModel),),
        context=_enabled(),
        policy=code.DynamicTracePolicy(max_calls=1),
    )
    assert len(result.facts_of_kind("dynamic_call")) == 1
    assert result.diagnostics_of_code("dryml.code.dynamic_trace_call_limit_exceeded")
    assert _summary(result).data == {"complete": False, "outcome": "call_limit_exceeded", "calls_recorded": 1, "max_calls": 1}
    assert continued == []


def test_hard_call_ceiling_records_exactly_ten_thousand_calls():
    continued = []

    def target(model):
        for _ in range(10_001):
            model.train()
        continued.append(True)

    result = code.trace(
        target,
        args=(Definition(TraceModel),),
        context=_enabled(include_annotations=False, include_method_contracts=False),
    )
    assert len(result.facts_of_kind("dynamic_call")) == 10_000
    assert result.diagnostics_of_code(
        "dryml.code.dynamic_trace_call_limit_exceeded"
    )[0].data == {
        "limit_name": "calls",
        "limit": 10_000,
        "observed_lower_bound": 10_001,
    }
    assert continued == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda value: bool(value),
        lambda value: len(value),
        lambda value: list(value),
        _async_iterate,
        lambda value: value[0],
        _assign_item,
        lambda value: value.attribute,
        lambda value: value(),
        lambda value: value + 1,
        lambda value: value @ 1,
        lambda value: -value,
        lambda value: value == 1,
        _await_value,
        _use_context_manager,
        _use_async_context_manager,
        lambda value: hash(value),
        lambda value: int(value),
        lambda value: float(value),
        lambda value: complex(value),
        lambda value: bytes(value),
        lambda value: f"{value}",
    ],
)
def test_unsupported_return_operations_are_explicit(operation):
    def target(model):
        operation(model.train())

    result = code.trace(target, args=(Definition(TraceModel),), context=_enabled())
    assert result.diagnostics_of_code("dryml.code.dynamic_trace_unsupported_return_operation")
    assert _summary(result).data["outcome"] == "unsupported_return_operation"
    assert len(result.facts_of_kind("dynamic_call")) == 1


def test_ignored_and_directly_returned_unsupported_values_complete():
    def ignored(model):
        model.train()

    def returned(model):
        return model.train()

    assert code.trace(ignored, args=(Definition(TraceModel),), context=_enabled()).ok
    assert code.trace(returned, args=(Definition(TraceModel),), context=_enabled()).ok


def test_caught_private_abort_cannot_make_trace_successful():
    def target(model):
        try:
            bool(model.train())
        except Exception:
            pass

    result = code.trace(target, args=(Definition(TraceModel),), context=_enabled())
    assert not result.ok
    assert _summary(result).data["complete"] is False


def test_target_failure_is_redacted_and_cleanup_allows_next_trace():
    secret = "TRACE-SECRET-MUST-NOT-LEAK"

    def target(model):
        model.train()
        raise RuntimeError(secret)

    result = code.trace(target, args=(Definition(TraceModel),), context=_enabled())
    encoded = json.dumps(result.to_data())
    assert secret not in encoded
    assert "traceback" not in encoded.lower()
    assert result.diagnostics_of_code("dryml.code.dynamic_trace_target_failed")
    assert _summary(result).data["outcome"] == "target_failed"
    assert code.trace(module_trace_target, args=(Definition(TraceModel),), context=_enabled()).ok


def test_method_fact_failure_retains_no_ambiguous_call_and_cleans_up(monkeypatch):
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    def fail(*args, **kwargs):
        raise RuntimeError("private collection failure")

    monkeypatch.setattr(dynamic_trace, "fragments_for_definition_method", fail)
    result = code.trace(module_trace_target, args=(Definition(TraceModel),), context=_enabled())
    assert result.diagnostics_of_code("dryml.code.dynamic_trace_method_fact_collection_failed")
    assert not result.facts_of_kind("dynamic_call")
    assert _summary(result).data["outcome"] == "method_fact_collection_failed"
    monkeypatch.undo()
    assert code.trace(module_trace_target, args=(Definition(TraceModel),), context=_enabled()).ok


def test_oversized_annotation_method_fact_is_a_bounded_collection_failure():
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    def target(first, oversized):
        first.train()
        oversized.train()

    result = code.trace(
        target,
        args=(Definition(TraceModel), Definition(OversizedAnnotationTraceModel)),
        context=_enabled(),
    )

    diagnostic = result.diagnostics_of_code(
        "dryml.code.dynamic_trace_method_fact_collection_failed"
    )[0]
    assert diagnostic.data == {
        "limit_name": "method_fact_string_chars",
        "limit": dynamic_trace.MAX_STRING_CHARS,
        "observed_lower_bound": dynamic_trace.MAX_STRING_CHARS + 1,
    }
    assert not result.diagnostics_of_code("dryml.code.algorithm_failed")
    assert len(result.facts_of_kind("dynamic_call")) == 1
    assert _summary(result).data == {
        "complete": False,
        "outcome": "method_fact_collection_failed",
        "calls_recorded": 1,
        "max_calls": 10_000,
    }


def test_method_fact_count_exact_boundary_and_n_plus_one(monkeypatch):
    import dryml.code.algorithms.dynamic_trace as dynamic_trace
    import dryml.code.algorithms.method_contracts as method_contracts

    def result_with_count(target, context, count):
        return code.CodeAnalysisResult(
            target=target.spec,
            facts=tuple(
                code.MethodContractFact(
                    source={"analyzer": "method_contracts", "target_kind": "class"},
                    data={
                        "method_contract_detected": True,
                        "class_module": None,
                        "class_qualname": None,
                        "trait_impls": [{
                            "name": f"method_{index}",
                            "traits": {"backend": None, "batch_mode": None},
                        }],
                        "has_user_call": False,
                    },
                )
                for index in range(count)
            ),
        )

    monkeypatch.setattr(
        method_contracts,
        "analyze_target",
        lambda target, context: result_with_count(target, context, 256),
    )
    exact = code.trace(
        module_trace_target,
        args=(Definition(TraceModel),),
        context=_enabled(include_annotations=False),
    )
    assert exact.ok
    assert len(exact.facts_of_kind("dynamic_call")[0].data["method_facts"]) == 256

    monkeypatch.setattr(
        method_contracts,
        "analyze_target",
        lambda target, context: result_with_count(target, context, 257),
    )
    exceeded = code.trace(
        module_trace_target,
        args=(Definition(TraceModel),),
        context=_enabled(include_annotations=False),
    )
    diagnostic = exceeded.diagnostics_of_code(
        "dryml.code.dynamic_trace_method_fact_collection_failed"
    )[0]
    assert diagnostic.data == {
        "limit_name": "method_facts",
        "limit": dynamic_trace.MAX_METHOD_FACTS,
        "observed_lower_bound": 257,
    }
    assert not exceeded.facts_of_kind("dynamic_call")


def test_unexpected_framework_failure_collects_or_raises_after_cleanup(monkeypatch):
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    original = dynamic_trace._encode_observed_call

    def fail(*args, **kwargs):
        raise RuntimeError("private implementation failure")

    monkeypatch.setattr(dynamic_trace, "_encode_observed_call", fail)
    collected = code.trace(module_trace_target, args=(Definition(TraceModel),), context=_enabled())
    assert collected.diagnostics_of_code("dryml.code.algorithm_failed")
    assert _summary(collected).data["outcome"] == "algorithm_failed"
    with pytest.raises(code.CodeAnalysisError):
        code.trace(
            module_trace_target,
            args=(Definition(TraceModel),),
            context=_enabled(diagnostics_policy="raise"),
        )
    monkeypatch.setattr(dynamic_trace, "_encode_observed_call", original)
    assert code.trace(module_trace_target, args=(Definition(TraceModel),), context=_enabled()).ok


def test_keyboard_interrupt_propagates_after_cleanup():
    def interrupted(model):
        model.train()
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        code.trace(interrupted, args=(Definition(TraceModel),), context=_enabled())
    assert code.trace(module_trace_target, args=(Definition(TraceModel),), context=_enabled()).ok


@pytest.mark.parametrize("interruption", [SystemExit, GeneratorExit])
def test_other_base_interruptions_propagate_after_cleanup(interruption):
    def interrupted(model):
        model.train()
        raise interruption()

    with pytest.raises(interruption):
        code.trace(interrupted, args=(Definition(TraceModel),), context=_enabled())
    assert code.trace(
        module_trace_target,
        args=(Definition(TraceModel),),
        context=_enabled(),
    ).ok


def test_cancellation_style_base_exception_propagates_after_cleanup():
    import asyncio

    def cancelled(model):
        model.train()
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        code.trace(cancelled, args=(Definition(TraceModel),), context=_enabled())
    assert code.trace(
        module_trace_target,
        args=(Definition(TraceModel),),
        context=_enabled(),
    ).ok


def test_nested_trace_restores_outer_planner():
    nested_results = []

    def inner(model):
        model.train()

    def outer(model):
        model.train()
        nested_results.append(code.trace(inner, args=(Definition(TraceModel),), context=_enabled()))
        model.train()

    result = code.trace(outer, args=(Definition(TraceModel),), context=_enabled())
    assert [fact.data["sequence"] for fact in result.facts_of_kind("dynamic_call")] == [0, 1]
    assert nested_results[0].ok
    assert len(nested_results[0].facts_of_kind("dynamic_call")) == 1


def test_failed_nested_trace_restores_outer_planner():
    nested_results = []

    def inner(model):
        bool(model.train())

    def outer(model):
        model.train()
        nested_results.append(
            code.trace(inner, args=(Definition(TraceModel),), context=_enabled())
        )
        model.train()

    result = code.trace(
        outer,
        args=(Definition(TraceModel),),
        context=_enabled(),
    )
    assert result.ok
    assert len(result.facts_of_kind("dynamic_call")) == 2
    assert nested_results[0].diagnostics_of_code(
        "dryml.code.dynamic_trace_unsupported_return_operation"
    )


def test_foreign_proxy_aborts_current_nested_trace():
    nested = []

    def outer(model):
        def inner(other):
            model.train()

        nested.append(code.trace(inner, args=(Definition(TraceModel),), context=_enabled()))
        model.train()

    outer_result = code.trace(outer, args=(Definition(TraceModel),), context=_enabled())
    assert nested[0].diagnostics_of_code("dryml.code.dynamic_trace_stale_proxy")
    assert outer_result.ok


def test_active_proxy_in_new_thread_raises_without_mutating_owner():
    errors = []

    def target(model):
        thread = threading.Thread(target=lambda: _call_and_capture(model, errors))
        thread.start()
        thread.join()
        model.train()

    result = code.trace(target, args=(Definition(TraceModel),), context=_enabled())
    assert len(errors) == 1 and isinstance(errors[0], code.DynamicTraceProxyError)
    assert result.ok
    assert len(result.facts_of_kind("dynamic_call")) == 1


def _call_and_capture(proxy, errors):
    try:
        proxy.train()
    except BaseException as exc:
        errors.append(exc)


def test_post_return_proxy_raises_without_mutating_result():
    escaped = []

    def target(model):
        escaped.append(model)
        model.train()

    result = code.trace(target, args=(Definition(TraceModel),), context=_enabled())
    before = result.to_data()
    with pytest.raises(code.DynamicTraceProxyError):
        escaped[0].train()
    assert result.to_data() == before


def test_copied_context_cannot_append_after_close():
    import contextvars

    escaped = []

    def target(model):
        escaped.append((contextvars.copy_context(), model))

    result = code.trace(target, args=(Definition(TraceModel),), context=_enabled())
    before = result.to_data()
    copied, proxy = escaped[0]
    with pytest.raises(code.DynamicTraceProxyError):
        copied.run(lambda: proxy.train())
    assert result.to_data() == before


def test_owner_close_during_attribute_failure_raises_public_proxy_error(monkeypatch):
    """Closing after attribute admission cannot leak the private trace abort."""

    import contextvars
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    admitted = threading.Event()
    closed = threading.Event()
    errors = []
    threads = []
    original_ensure = dynamic_trace._ensure_proxy_owner
    original_close = dynamic_trace._Planner.close
    admissions = 0

    def pause_after_first_admission(owner):
        nonlocal admissions
        original_ensure(owner)
        admissions += 1
        if admissions == 1:
            admitted.set()
            assert closed.wait(timeout=5)

    def close_and_release(planner):
        original_close(planner)
        closed.set()

    monkeypatch.setattr(dynamic_trace, "_ensure_proxy_owner", pause_after_first_admission)
    monkeypatch.setattr(dynamic_trace._Planner, "close", close_and_release)

    def target(model):
        copied = contextvars.copy_context()

        def child():
            try:
                copied.run(lambda: model.missing)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=child)
        threads.append(thread)
        thread.start()
        assert admitted.wait(timeout=5)

    result = code.trace(target, args=(Definition(TraceModel),), context=_enabled())
    threads[0].join(timeout=5)
    assert not threads[0].is_alive()
    assert result.ok
    assert len(errors) == 1
    assert isinstance(errors[0], code.DynamicTraceProxyError)


def test_copied_foreign_context_close_race_raises_public_proxy_error(monkeypatch):
    """A copied foreign planner closing cannot leak a private trace abort."""

    import contextvars
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    close_started = threading.Event()
    child_attempting = threading.Event()
    child_ready = threading.Event()
    errors = []
    threads = []
    racing_planners = []
    original_close = dynamic_trace._Planner.close

    def close_while_child_is_admitting(planner):
        if planner not in racing_planners:
            return original_close(planner)
        with planner.lock:
            close_started.set()
            assert child_attempting.wait(timeout=5)
            planner.state = "closed"

    monkeypatch.setattr(dynamic_trace._Planner, "close", close_while_child_is_admitting)

    def target(model):
        def inner(_other):
            racing_planners.append(dynamic_trace._CURRENT_PLANNER.get())
            copied = contextvars.copy_context()

            def child():
                child_ready.set()
                assert close_started.wait(timeout=5)
                child_attempting.set()
                try:
                    copied.run(lambda: model.train())
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=child)
            threads.append(thread)
            thread.start()
            assert child_ready.wait(timeout=5)

        inner_result = code.trace(inner, args=(Definition(TraceModel),), context=_enabled())
        assert inner_result.ok

    result = code.trace(target, args=(Definition(TraceModel),), context=_enabled())
    threads[0].join(timeout=5)
    assert not threads[0].is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], code.DynamicTraceProxyError)
    assert result.ok
    assert not result.facts_of_kind("dynamic_call")


def test_overlapping_thread_traces_do_not_mix_facts_or_limits():
    barrier = threading.Barrier(2)
    results = []

    def target(model):
        barrier.wait(timeout=5)
        model.train()

    def run():
        results.append(code.trace(target, args=(Definition(TraceModel),), context=_enabled()))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(results) == 2
    assert all(result.ok and len(result.facts_of_kind("dynamic_call")) == 1 for result in results)


def test_copied_child_thread_context_serializes_sequence_without_corruption():
    import contextvars

    def target(model):
        contexts = [contextvars.copy_context(), contextvars.copy_context()]
        threads = [threading.Thread(target=lambda ctx=ctx: ctx.run(lambda: model.train())) for ctx in contexts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

    result = code.trace(target, args=(Definition(TraceModel),), context=_enabled())
    assert result.ok
    assert [fact.data["sequence"] for fact in result.facts_of_kind("dynamic_call")] == [0, 1]


def test_overlapping_task_contexts_share_only_their_own_active_planner():
    import asyncio

    async def child(model):
        await asyncio.sleep(0)
        model.train()

    def target(model):
        async def run_children():
            await asyncio.gather(child(model), child(model))

        asyncio.run(run_children())

    result = code.trace(target, args=(Definition(TraceModel),), context=_enabled())
    assert result.ok
    assert [fact.data["sequence"] for fact in result.facts_of_kind("dynamic_call")] == [0, 1]


def test_independent_async_tasks_do_not_share_trace_planners():
    import asyncio

    barrier = threading.Barrier(2)

    def target(model):
        barrier.wait(timeout=5)
        model.train()

    async def run_traces():
        return await asyncio.gather(*(
            asyncio.to_thread(
                code.trace,
                target,
                args=(Definition(TraceModel),),
                context=_enabled(),
            )
            for _ in range(2)
        ))

    results = asyncio.run(run_traces())
    assert all(result.ok for result in results)
    assert all(len(result.facts_of_kind("dynamic_call")) == 1 for result in results)


def test_import_ref_receiver_permission_and_source_spec_gates():
    import_ref = Definition(ImportRef(__name__, "TraceModel"))
    allowed = code.trace(module_trace_target, args=(import_ref,), context=_enabled())
    assert allowed.ok
    denied = code.trace(module_trace_target, args=(import_ref,), context=_enabled(allow_import=False))
    assert denied.diagnostics_of_code("dryml.code.dynamic_trace_receiver_resolution_failed")

    source = SourceSpec.from_source("class InlineModel:\n    def train(self):\n        raise AssertionError('must not execute')", kind="class", name="InlineModel")
    source_result = code.trace(module_trace_target, args=(Definition(source),), context=_enabled())
    assert source_result.ok
    blocked = code.trace(module_trace_target, args=(Definition(source),), context=_enabled(allow_source=False))
    assert blocked.diagnostics_of_code("dryml.code.dynamic_trace_receiver_resolution_failed")


def test_missing_or_nonclass_receiver_fails_before_target_execution():
    called = []

    def target(model):
        called.append(True)

    for definition in (Definition(), Definition(ImportRef("builtins", "len"))):
        result = code.trace(target, args=(definition,), context=_enabled())
        assert result.diagnostics_of_code("dryml.code.dynamic_trace_receiver_resolution_failed")
        assert not result.facts
    assert called == []


def test_receiver_class_path_exact_boundary_and_n_plus_one():
    method_name = "train"

    def train(self):
        raise AssertionError("trace must not invoke the receiver method")

    def model_with_path_length(length):
        qualname = "BoundaryModel"
        module = "m" * (length - len(qualname) - 1)
        return type(
            qualname,
            (),
            {"__module__": module, method_name: train},
        )

    called = []
    exact = code.trace(
        lambda model: (called.append("exact"), model.train()),
        args=(Definition(model_with_path_length(4_096)),),
        context=_enabled(include_annotations=False, include_method_contracts=False),
    )
    assert exact.ok
    assert called == ["exact"]
    assert exact.facts_of_kind("dynamic_call")[0].data["receiver_class"] is None

    exceeded = code.trace(
        lambda model: called.append("over"),
        args=(Definition(model_with_path_length(4_097)),),
        context=_enabled(include_annotations=False, include_method_contracts=False),
    )
    diagnostic = exceeded.diagnostics_of_code(
        "dryml.code.dynamic_trace_receiver_resolution_failed"
    )[0]
    assert diagnostic.data == {
        "limit_name": "receiver_class_chars",
        "limit": 4_096,
        "observed_lower_bound": 4_097,
    }
    assert called == ["exact"]
    assert not exceeded.facts


def test_invalid_bounded_identity_fails_before_target_execution(monkeypatch):
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    called = []

    def fail_identity(value):
        raise dynamic_trace.StableHashLimitError("depth", 128, 129)

    monkeypatch.setattr(dynamic_trace, "_definition_reference", fail_identity)
    result = code.trace(
        lambda model: called.append(model),
        args=(Definition(TraceModel),),
        context=_enabled(),
    )
    diagnostic = result.diagnostics_of_code(
        "dryml.code.dynamic_trace_argument_limit_exceeded"
    )[0]
    assert diagnostic.data == {
        "limit_name": "hash_depth",
        "limit": 128,
        "observed_lower_bound": 129,
    }
    assert called == []
    assert not result.facts


def test_diagnostic_count_exact_boundary_and_n_plus_one():
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    target = code.CodeTarget(code.CodeTargetSpec("function"), obj=lambda: None)
    request = dynamic_trace._InvocationRequest(
        target=target,
        args=(),
        kwargs={},
        context=_enabled(),
        policy=code.DynamicTracePolicy(),
    )
    planner = dynamic_trace._Planner(request)
    diagnostic = dynamic_trace._diagnostic(
        "dryml.code.test",
        "Bounded test diagnostic.",
        target_kind="function",
    )
    for _ in range(dynamic_trace.MAX_DIAGNOSTICS):
        planner.add_diagnostic(diagnostic)
    assert len(planner.diagnostics) == dynamic_trace.MAX_DIAGNOSTICS
    assert planner.state == "active"

    planner.add_diagnostic(diagnostic)
    assert len(planner.diagnostics) == dynamic_trace.MAX_DIAGNOSTICS
    assert planner.diagnostics[-1].code == (
        "dryml.code.dynamic_trace_diagnostics_limit_exceeded"
    )
    assert planner.diagnostics[-1].data["observed_lower_bound"] == 257
    assert planner.state == "aborted"
    assert planner.outcome == "diagnostics_limit_exceeded"


def test_diagnostic_code_and_message_character_exact_boundaries():
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    diagnostic = dynamic_trace._diagnostic(
        "c" * 1_024,
        "m" * 1_024,
        target_kind="function",
    )
    assert len(diagnostic.code) == 1_024
    assert len(diagnostic.message) == 1_024
    with pytest.raises(RuntimeError, match="constants exceed"):
        dynamic_trace._diagnostic(
            "c" * 1_025,
            "message",
            target_kind="function",
        )
    with pytest.raises(RuntimeError, match="constants exceed"):
        dynamic_trace._diagnostic(
            "code",
            "m" * 1_025,
            target_kind="function",
        )


def test_diagnostic_limit_trace_has_incomplete_summary():
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    def target():
        planner = dynamic_trace._CURRENT_PLANNER.get()
        diagnostic = dynamic_trace._diagnostic(
            "dryml.code.test",
            "Bounded test diagnostic.",
            target_kind="function",
        )
        for _ in range(dynamic_trace.MAX_DIAGNOSTICS + 1):
            planner.add_diagnostic(diagnostic)

    result = code.trace(target, context=_enabled())
    assert result.diagnostics_of_code(
        "dryml.code.dynamic_trace_diagnostics_limit_exceeded"
    )
    assert _summary(result).data == {
        "complete": False,
        "outcome": "diagnostics_limit_exceeded",
        "calls_recorded": 0,
        "max_calls": 10_000,
    }


def test_complete_zero_call_is_distinct_from_disabled_and_incomplete():
    complete = code.trace(lambda: None, context=_enabled())
    disabled = code.trace(lambda: None)
    incomplete = code.trace(lambda model: bool(model.train()), args=(Definition(TraceModel),), context=_enabled())
    assert _summary(complete).data["calls_recorded"] == 0
    assert _summary(complete).data["complete"] is True
    assert not disabled.facts
    assert _summary(incomplete).data["complete"] is False


def test_per_call_and_aggregate_result_limits_fail_closed(monkeypatch):
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    def target(model):
        model.train("x" * 200)

    # Isolate call/result admission from method-fact collection. Invalid or
    # oversized method-fact wire data has its own fail-closed outcome.
    context = _enabled(include_annotations=False, include_method_contracts=False)
    monkeypatch.setattr(dynamic_trace, "MAX_CALL_FACT_BYTES", 200)
    per_call = code.trace(target, args=(Definition(TraceModel),), context=context)
    assert per_call.diagnostics_of_code("dryml.code.dynamic_trace_result_limit_exceeded")
    assert not per_call.facts_of_kind("dynamic_call")
    assert _summary(per_call).data["outcome"] == "result_limit_exceeded"

    monkeypatch.setattr(dynamic_trace, "MAX_CALL_FACT_BYTES", 1_048_576)
    monkeypatch.setattr(dynamic_trace, "MAX_RESULT_BYTES", 250)
    aggregate = code.trace(module_trace_target, args=(Definition(TraceModel),), context=context)
    assert aggregate.diagnostics_of_code("dryml.code.dynamic_trace_result_limit_exceeded")
    assert _summary(aggregate).data["complete"] is False


def test_aggregate_result_admission_exact_boundary_and_n_plus_one(monkeypatch):
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    context = _enabled(include_annotations=False, include_method_contracts=False)
    baseline = code.trace(
        module_trace_target,
        args=(Definition(TraceModel),),
        context=context,
    )
    def encoded_fact_size(fact):
        return len(json.dumps(
            fact.to_data(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"))

    call = baseline.facts_of_kind("dynamic_call")[0]
    reserved_summary_size = dynamic_trace._reserved_summary_size(
        target_kind=call.source["target_kind"],
        calls_recorded=1,
        max_calls=10_000,
    )
    exact_limit = encoded_fact_size(call) + reserved_summary_size

    monkeypatch.setattr(dynamic_trace, "MAX_RESULT_BYTES", exact_limit)
    exact = code.trace(
        module_trace_target,
        args=(Definition(TraceModel),),
        context=context,
    )
    assert exact.ok
    assert len(exact.facts_of_kind("dynamic_call")) == 1
    assert sum(encoded_fact_size(fact) for fact in exact.facts) <= exact_limit

    monkeypatch.setattr(dynamic_trace, "MAX_RESULT_BYTES", exact_limit - 1)
    exceeded = code.trace(
        module_trace_target,
        args=(Definition(TraceModel),),
        context=context,
    )
    diagnostic = exceeded.diagnostics_of_code(
        "dryml.code.dynamic_trace_result_limit_exceeded"
    )[0]
    assert diagnostic.data["limit"] == exact_limit - 1
    assert diagnostic.data["observed_lower_bound"] == exact_limit
    assert not exceeded.facts_of_kind("dynamic_call")
    assert _summary(exceeded).data["outcome"] == "result_limit_exceeded"


def test_failure_summary_fits_reserved_aggregate_result_budget(monkeypatch):
    import dryml.code.algorithms.dynamic_trace as dynamic_trace

    context = _enabled(include_annotations=False, include_method_contracts=False)

    def succeeds_after_call(model):
        model.train()

    baseline = code.trace(
        succeeds_after_call,
        args=(Definition(TraceModel),),
        context=context,
    )
    call = baseline.facts_of_kind("dynamic_call")[0]
    max_result_bytes = (
        len(json.dumps(call.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        + dynamic_trace._reserved_summary_size(
            target_kind=call.source["target_kind"],
            calls_recorded=1,
            max_calls=10_000,
        )
    )
    monkeypatch.setattr(dynamic_trace, "MAX_RESULT_BYTES", max_result_bytes)

    def fails_after_call(model):
        model.train()
        raise RuntimeError("trace target failure")

    result = code.trace(
        fails_after_call,
        args=(Definition(TraceModel),),
        context=context,
    )
    assert result.diagnostics_of_code("dryml.code.dynamic_trace_target_failed")
    assert len(result.facts_of_kind("dynamic_call")) == 1
    assert _summary(result).data["outcome"] == "target_failed"
    assert sum(
        len(json.dumps(fact.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        for fact in result.facts
    ) <= max_result_bytes
