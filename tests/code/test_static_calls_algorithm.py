"""Contract tests for opt-in conservative static call resolution."""

from __future__ import annotations

import json
import operator
from typing import Protocol

import dryml.code as code
import pytest
from dryml.code.algorithms import static_calls
from dryml.code.algorithms import static_analysis
from dryml.code.algorithms.source import SourceInfo


HELPER_EXECUTED = False
METHOD_EXECUTED = False
TARGET_EXECUTED = False


def helper():
    global HELPER_EXECUTED
    HELPER_EXECUTED = True


def target_body_must_not_run():
    global TARGET_EXECUTED
    TARGET_EXECUTED = True


class CallableWithoutExecution:
    def __call__(self):
        raise AssertionError("static analysis must not invoke callable instances")


CALLABLE_INSTANCE = CallableWithoutExecution()


class ImportableGlobalClass:
    pass


class MainModuleModel:
    def train(self):
        return None


class HostileCallable:
    def __getattribute__(self, name):
        raise AssertionError("static analysis must not inspect callable instance attributes")

    def __call__(self):
        raise AssertionError("static analysis must not invoke callable instances")


HOSTILE_CALLABLE = HostileCallable()


class HostileWrapped:
    def __getattribute__(self, name):
        raise AssertionError("static analysis must not inspect __wrapped__ metadata")


class Model:
    def train(self):
        global METHOD_EXECUTED
        METHOD_EXECUTED = True


class StaticModel:
    @staticmethod
    def train():
        raise AssertionError("static analysis must not invoke static methods")


class ClassModel:
    @classmethod
    def train(cls):
        raise AssertionError("static analysis must not invoke class methods")


class PropertyModel:
    @property
    def train(self):
        raise AssertionError("static analysis must not invoke properties")


class ProtocolModel(Protocol):
    def train(self): ...


class HostileMeta(type):
    def __getattribute__(cls, name):
        raise AssertionError("static analysis must not invoke metaclass attribute hooks")


class HostileModel(metaclass=HostileMeta):
    def train(self):
        return None


def direct_global_target():
    helper()


def importable_global_class_target():
    ImportableGlobalClass()


BUILTIN_HELPER = operator.add
BOUND_BUILTIN_HELPER = [].append


def builtin_global_target():
    BUILTIN_HELPER(1, 2)


def bound_builtin_global_target():
    BOUND_BUILTIN_HELPER(1)


def callable_instance_target():
    CALLABLE_INSTANCE()


def hostile_callable_target():
    HOSTILE_CALLABLE()


def annotated_method_target(model):
    model.train()


annotated_method_target.__annotations__["model"] = Model


def static_method_target(model):
    model.train()


static_method_target.__annotations__["model"] = StaticModel


def class_method_target(model):
    model.train()


class_method_target.__annotations__["model"] = ClassModel


def property_target(model):
    model.train()


property_target.__annotations__["model"] = PropertyModel


def string_annotation_target(model: "Model"):
    model.train()


def missing_annotation_target(model):
    model.train()


def union_annotation_target(model):
    model.train()


union_annotation_target.__annotations__["model"] = Model | StaticModel


def protocol_annotation_target(model):
    model.train()


protocol_annotation_target.__annotations__["model"] = ProtocolModel


def hostile_meta_target(model):
    model.train()


hostile_meta_target.__annotations__["model"] = HostileModel


def alias_target():
    alias = helper
    alias()


def reassigned_receiver_target(model):
    model = object()
    model.train()


reassigned_receiver_target.__annotations__["model"] = Model


def attribute_reassigned_receiver_target(model):
    model.train = helper
    model.train()


attribute_reassigned_receiver_target.__annotations__["model"] = Model


def dynamic_attribute_reassigned_receiver_target(model):
    setattr(model, "train", helper)
    model.train()


dynamic_attribute_reassigned_receiver_target.__annotations__["model"] = Model


def local_annotation_target():
    model = object()
    model.train()


local_annotation_target.__annotations__["model"] = Model


def unknown_global_target():
    unknown_static_global()


def spoofed_helper():
    return None


def spoofed_reference_target():
    spoofed_helper()


LOCAL_GLOBAL_HELPER = lambda: None


def local_global_reference_target():
    LOCAL_GLOBAL_HELPER()


def main_module_annotation_target(model):
    model.train()


main_module_annotation_target.__annotations__["model"] = MainModuleModel


class ClassSourceTarget:
    pass


def parameter_shadows_global(helper):
    helper()


def local_shadows_global():
    helper = lambda: None
    helper()


def loop_rebinds_receiver(model, models):
    for model in models:
        model.train()


loop_rebinds_receiver.__annotations__["model"] = Model


def nested_scope_target(model):
    def nested():
        model.train()
    return nested


nested_scope_target.__annotations__["model"] = Model


def nested_callable_expression_target():
    def nested():
        return (lambda: None)()
    return nested


def closure_factory():
    def helper():
        return None

    def closure_target():
        helper()

    return closure_target


CLOSURE_TARGET = closure_factory()


def comprehension_shadow_target(values):
    return [helper() for helper in values]


def match_shadow_target(value):
    match value:
        case helper:
            return helper()


def deleted_shadow_target():
    del helper
    helper()


def nested_class_shadow_target():
    class Inner:
        def run(self):
            helper()

    return Inner


def decorator_factory():
    return lambda function: function


@decorator_factory()
def decorated_default_target(value=helper()):
    helper()


def attribute_chain_target(model):
    model.first.second()


def nested_receiver_target(model):
    model.child().train()


nested_receiver_target.__annotations__["model"] = Model


def dynamic_getattr_target(model):
    getattr(model, "train")()


dynamic_getattr_target.__annotations__["model"] = Model


def no_call_target():
    return None


def _facts(target):
    result = code.analyze(target, algorithms=("static_calls",))
    return result, result.facts_of_kind("static_call")


def test_static_calls_is_registered_opt_in_and_non_invoking():
    global HELPER_EXECUTED, TARGET_EXECUTED
    HELPER_EXECUTED = False
    TARGET_EXECUTED = False

    default_result = code.analyze(direct_global_target)
    result, facts = _facts(direct_global_target)
    code.analyze(target_body_must_not_run, algorithms=("static_calls",))

    assert "static_calls" in code.available_analyzers()
    assert not default_result.facts_of_kind("static_call")
    assert HELPER_EXECUTED is False
    assert TARGET_EXECUTED is False
    assert facts[0].data["status"] == "resolved"
    assert facts[0].data["confidence"] == "exact_static"
    assert facts[0].data["syntax"] == "direct_name"
    assert set(facts[0].data["target"]) == {"kind", "import_path", "method_name", "subject_ref"}
    assert result.facts_of_kind("static_call_summary")[0].data["complete"] is True


def test_static_calls_resolves_safely_importable_global_classes():
    _, facts = _facts(importable_global_class_target)

    assert facts[0].data["status"] == "resolved"
    assert facts[0].data["target"]["kind"] == "class"
    assert facts[0].data["target"]["import_path"].endswith(":ImportableGlobalClass")


def test_static_calls_resolves_safe_builtin_globals():
    _, facts = _facts(builtin_global_target)

    assert facts[0].data["status"] == "resolved"
    assert facts[0].data["target"]["import_path"] == "_operator:add"


def test_static_calls_rejects_bound_builtin_globals_without_receiver_identity():
    _, facts = _facts(bound_builtin_global_target)

    assert facts[0].data["status"] == "unsupported"
    assert facts[0].data["reason"] == "global_value_not_safe_function"


def test_static_calls_resolves_only_safe_annotated_methods_without_invocation():
    global METHOD_EXECUTED
    METHOD_EXECUTED = False

    _, method_facts = _facts(annotated_method_target)
    _, static_facts = _facts(static_method_target)
    _, class_facts = _facts(class_method_target)
    _, property_facts = _facts(property_target)

    assert method_facts[0].data["status"] == "resolved"
    assert method_facts[0].data["syntax"] == "annotated_receiver_method"
    assert static_facts[0].data["status"] == "resolved"
    assert class_facts[0].data["status"] == "resolved"
    assert property_facts[0].data == {
        **property_facts[0].data,
        "status": "unsupported",
        "reason": "property_descriptor",
    }
    assert METHOD_EXECUTED is False


def test_static_calls_reports_conservative_non_resolution_cases():
    cases = (
        (string_annotation_target, "string_annotation"),
        (missing_annotation_target, "missing_annotation"),
        (union_annotation_target, "non_concrete_annotation"),
        (protocol_annotation_target, "non_standard_annotation_class"),
        (alias_target, "local_name_unsupported"),
        (reassigned_receiver_target, "receiver_reassigned"),
        (attribute_reassigned_receiver_target, "receiver_reassigned"),
        (dynamic_attribute_reassigned_receiver_target, "receiver_reassigned"),
        (local_annotation_target, "attribute_chain_unsupported"),
        (nested_receiver_target, "call_result_receiver"),
        (parameter_shadows_global, "parameter_name_unsupported"),
        (local_shadows_global, "local_name_unsupported"),
        (loop_rebinds_receiver, "receiver_reassigned"),
        (nested_scope_target, "nested_scope_unsupported"),
        (CLOSURE_TARGET, "closure_name_unsupported"),
        (comprehension_shadow_target, "nested_scope_unsupported"),
        (match_shadow_target, "local_name_unsupported"),
        (deleted_shadow_target, "local_name_unsupported"),
        (nested_class_shadow_target, "nested_scope_unsupported"),
    )

    for target, reason in cases:
        _, facts = _facts(target)
        assert all(fact.data["status"] != "resolved" for fact in facts)
        assert any(fact.data["reason"] == reason for fact in facts)


def test_static_calls_does_not_invoke_callable_instances_or_dynamic_getattr():
    _, callable_facts = _facts(callable_instance_target)
    _, hostile_facts = _facts(hostile_callable_target)
    _, dynamic_facts = _facts(dynamic_getattr_target)

    assert callable_facts[0].data["status"] == "unsupported"
    assert hostile_facts[0].data["status"] == "unsupported"
    assert all(fact.data["status"] != "resolved" for fact in dynamic_facts)


def test_static_calls_reports_unknown_globals_and_class_source_availability():
    unknown_result, unknown_facts = _facts(unknown_global_target)
    class_result = code.analyze(ClassSourceTarget, algorithms=("static_calls",))

    assert unknown_result.ok
    assert unknown_facts[0].data["status"] == "unresolved"
    assert unknown_facts[0].data["reason"] == "global_name_unavailable"
    assert class_result.facts_of_kind("static_call_summary")
    assert not class_result.diagnostics_of_code("dryml.code.source_unavailable")


def test_static_calls_does_not_publish_unverified_or_oversized_import_references():
    original_module = spoofed_helper.__module__
    original_qualname = spoofed_helper.__qualname__
    try:
        spoofed_helper.__module__ = "builtins"
        spoofed_helper.__qualname__ = "len"
        _, facts = _facts(spoofed_reference_target)
        assert facts[0].data["status"] == "unsupported"
        assert facts[0].data["reason"] == "target_reference_unavailable"

        spoofed_helper.__module__ = "x" * (static_analysis.MAX_STATIC_SCALAR_CHARS + 1)
        result, oversized_facts = _facts(spoofed_reference_target)
        assert oversized_facts[0].data["reason"] == "target_reference_limit_exceeded"
        diagnostic = result.diagnostics_of_code("dryml.code.static_target_reference_limit_exceeded")[0]
        assert diagnostic.data == {
            "limit_name": "scalar_chars",
            "limit": static_analysis.MAX_STATIC_SCALAR_CHARS,
            "observed_lower_bound": static_analysis.MAX_STATIC_SCALAR_CHARS + 1,
        }
        assert result.facts_of_kind("static_call_summary")[0].data["complete"] is False
    finally:
        spoofed_helper.__module__ = original_module
        spoofed_helper.__qualname__ = original_qualname


def test_static_calls_rejects_globals_without_stable_import_identity():
    _, facts = _facts(local_global_reference_target)

    assert facts[0].data["status"] == "unsupported"
    assert facts[0].data["reason"] == "target_reference_unavailable"


def test_static_calls_rejects_main_module_annotated_methods_without_import_path():
    original_module = MainModuleModel.train.__module__
    try:
        MainModuleModel.train.__module__ = "__main__"
        _, facts = _facts(main_module_annotation_target)
    finally:
        MainModuleModel.train.__module__ = original_module

    assert facts[0].data["status"] == "unsupported"
    assert facts[0].data["reason"] == "target_reference_unavailable"


def test_static_calls_handles_nested_callable_expressions_conservatively():
    result, facts = _facts(nested_callable_expression_target)

    assert not result.diagnostics_of_code("dryml.code.algorithm_failed")
    assert facts[0].data["status"] == "unsupported"
    assert facts[0].data["display"] == "lambda"
    assert facts[0].data["reason"] == "nested_scope_unsupported"
    assert result.facts_of_kind("static_call_summary")[0].data["complete"] is True


def test_static_calls_does_not_inspect_hostile_targets_or_wrapped_metadata():
    def wrapped_target():
        helper()

    wrapped_target.__wrapped__ = HostileWrapped()

    hostile_target_result = code.analyze(HostileCallable(), algorithms=("static_calls",))
    hostile_class_result = code.analyze(HostileModel, algorithms=("static_calls",))
    wrapped_result, wrapped_facts = _facts(wrapped_target)

    assert hostile_target_result.diagnostics_of_code("dryml.code.source_unavailable")
    assert hostile_class_result.diagnostics_of_code("dryml.code.source_unavailable")
    assert wrapped_result.ok
    assert wrapped_facts[0].data["status"] == "resolved"


def test_static_calls_inspects_hostile_metaclasses_without_dynamic_lookup():
    result, facts = _facts(hostile_meta_target)

    assert result.ok
    assert facts[0].data["status"] == "ambiguous"
    assert facts[0].data["reason"] == "non_standard_annotation_class"


def test_static_call_facts_round_trip_with_each_status_and_are_json_compatible():
    targets = (direct_global_target, missing_annotation_target, union_annotation_target, property_target)
    statuses = set()
    for target in targets:
        _, facts = _facts(target)
        for fact in facts:
            restored = code.CodeFact.from_data(fact.to_data())
            assert isinstance(restored, code.StaticCallFact)
            json.dumps(restored.to_data())
            statuses.add(restored.data["status"])

    assert statuses == {"resolved", "unresolved", "ambiguous", "unsupported"}


def test_static_calls_emits_complete_zero_count_summary():
    result, facts = _facts(no_call_target)
    summary = result.facts_of_kind("static_call_summary")[0]

    assert not facts
    assert summary.data["complete"] is True
    assert summary.data["call_sites_seen"] == 0
    assert summary.data["facts_emitted"] == 0


def test_static_calls_emits_all_definition_calls_in_textual_source_order():
    _, facts = _facts(decorated_default_target)

    positions = [(fact.data["relative_line"], fact.data["col_offset"]) for fact in facts]
    assert positions == sorted(positions)
    assert len(facts) == 3


def test_static_calls_enforces_call_site_limit(monkeypatch):
    monkeypatch.setattr(static_calls, "MAX_CALL_SITES", 1)

    result, facts = _facts(direct_global_target)
    assert len(facts) == 1
    assert result.facts_of_kind("static_call_summary")[0].data["complete"] is True
    assert not result.diagnostics_of_code("dryml.code.static_call_sites_limit_exceeded")
    # The fixture has one site; use a second AST call by analyzing this local function.
    def two_calls():
        helper()
        helper()

    result, facts = _facts(two_calls)
    summary = result.facts_of_kind("static_call_summary")[0]

    assert len(facts) == 1
    assert summary.data["complete"] is False
    diagnostic = result.diagnostics_of_code("dryml.code.static_call_sites_limit_exceeded")[0]
    assert diagnostic.data == {"limit_name": "call_sites", "limit": 1, "observed_lower_bound": 2}


def test_static_calls_enforces_chain_and_scalar_bounds(monkeypatch):
    monkeypatch.setattr(static_calls, "MAX_CHAIN_COMPONENTS", 1)
    _, chain_facts = _facts(attribute_chain_target)

    assert chain_facts[0].data["status"] == "unsupported"
    assert chain_facts[0].data["reason"] == "chain_limit_exceeded"

    oversized_name = "x" * (static_analysis.MAX_STATIC_SCALAR_CHARS + 1)
    source = f"def synthetic():\n    {oversized_name}()\n"
    monkeypatch.setattr(static_calls, "get_source_info", lambda obj: SourceInfo(source, "synthetic.py", 1))
    result, scalar_facts = _facts(direct_global_target)

    assert scalar_facts[0].data["reason"] == "scalar_limit_exceeded"
    assert result.facts_of_kind("static_call_summary")[0].data["complete"] is True


def test_static_calls_accepts_exact_chain_and_scalar_limits(monkeypatch):
    monkeypatch.setattr(static_calls, "MAX_CHAIN_COMPONENTS", 2)
    _, chain_facts = _facts(attribute_chain_target)
    assert chain_facts[0].data["reason"] == "attribute_chain_unsupported"

    exact_name = "x" * static_analysis.MAX_STATIC_SCALAR_CHARS
    source = f"def synthetic():\n    {exact_name}()\n"
    monkeypatch.setattr(static_calls, "get_source_info", lambda obj: SourceInfo(source, "synthetic.py", 1))
    _, scalar_facts = _facts(direct_global_target)
    assert scalar_facts[0].data["display"] == exact_name
    assert scalar_facts[0].data["reason"] == "global_name_unavailable"


def test_static_calls_bounds_oversized_source_filenames_without_losing_summary(monkeypatch):
    filename = "x" * (static_analysis.MAX_STATIC_SCALAR_CHARS + 1)
    monkeypatch.setattr(
        static_calls,
        "get_source_info",
        lambda obj: SourceInfo("def synthetic():\n    helper()\n", filename, 1),
    )

    result, facts = _facts(direct_global_target)

    assert facts[0].source["filename"] is None
    assert result.facts_of_kind("static_call_summary")[0].source["filename"] is None

    monkeypatch.setattr(
        static_calls,
        "get_source_info",
        lambda obj: SourceInfo("def synthetic():\n    pass\n", filename, 1),
    )
    result, facts = _facts(no_call_target)

    assert not facts
    assert result.facts_of_kind("static_call_summary")[0].source["filename"] is None


def test_static_calls_enforces_source_and_ast_bounds(monkeypatch):
    monkeypatch.setattr(static_analysis, "MAX_SOURCE_BYTES", 1)
    source_result, _ = _facts(direct_global_target)
    assert source_result.diagnostics_of_code("dryml.code.static_source_bytes_limit_exceeded")

    monkeypatch.setattr(static_analysis, "MAX_SOURCE_BYTES", 1_048_576)
    monkeypatch.setattr(static_analysis, "MAX_AST_NODES", 1)
    node_result, _ = _facts(direct_global_target)
    assert node_result.diagnostics_of_code("dryml.code.static_ast_nodes_limit_exceeded")
    node_summary = node_result.facts_of_kind("static_call_summary")[0]
    assert node_summary.data["complete"] is False
    assert node_summary.data["call_sites_seen"] == 0
    assert node_summary.data["facts_emitted"] == 0


def test_static_calls_reports_unavailable_when_imports_are_disabled():
    result = code.analyze(
        "probe_targets:plain_function",
        algorithms=("static_calls",),
        context=code.CodeAnalysisContext(allow_import=False),
    )

    assert result.diagnostics_of_code("dryml.code.source_unavailable")


def test_static_call_fact_rejects_unbounded_or_invalid_serialized_data():
    data = _facts(direct_global_target)[1][0].to_data()
    data["data"]["display"] = "x" * 4_097
    with pytest.raises(ValueError, match="bounded string"):
        code.CodeFact.from_data(data)

    data = _facts(direct_global_target)[1][0].to_data()
    data["data"]["status"] = "guessed"
    with pytest.raises(ValueError, match="unsupported StaticCallFact status"):
        code.CodeFact.from_data(data)

    data = _facts(direct_global_target)[1][0].to_data()
    data["data"]["confidence"] = "conservative_hint"
    with pytest.raises(ValueError, match="must be exact_static"):
        code.CodeFact.from_data(data)

    data = _facts(direct_global_target)[1][0].to_data()
    data["source"]["analyzer"] = "wrong"
    with pytest.raises(ValueError, match="static_calls analyzer"):
        code.CodeFact.from_data(data)

    data = _facts(direct_global_target)[1][0].to_data()
    data["data"]["unexpected"] = "x" * 10_000
    with pytest.raises(ValueError, match="fixed static-call schema"):
        code.CodeFact.from_data(data)

    data = _facts(direct_global_target)[1][0].to_data()
    data["source"]["filename"] = "x" * 4_097
    with pytest.raises(ValueError, match="source filename"):
        code.CodeFact.from_data(data)

    data = _facts(direct_global_target)[1][0].to_data()
    data["data"]["display"] = None
    with pytest.raises(ValueError, match="display must"):
        code.CodeFact.from_data(data)

    data = _facts(direct_global_target)[1][0].to_data()
    data["data"]["status"] = "unsupported"
    data["data"]["confidence"] = "conservative_hint"
    data["data"]["target"] = None
    data["data"]["reason"] = ""
    with pytest.raises(ValueError, match="non-empty bounded reason"):
        code.CodeFact.from_data(data)

    data = _facts(direct_global_target)[1][0].to_data()
    data["data"]["unexpected"] = [["x"] * 10_000]
    with pytest.raises(ValueError, match="fixed static-call schema"):
        code.CodeFact.from_data(data)

    for field_name, value, message in (
        ("relative_line", 0, "must be positive"),
        ("absolute_line", -1, "must be positive"),
        ("col_offset", -1, "must be non-negative"),
    ):
        data = _facts(direct_global_target)[1][0].to_data()
        data["data"][field_name] = value
        with pytest.raises(ValueError, match=message):
            code.CodeFact.from_data(data)
