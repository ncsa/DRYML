"""Contract tests for opt-in conservative static call resolution."""

from __future__ import annotations

import json

import dryml.code as code
import pytest
from dryml.code.algorithms import static_calls


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


def direct_global_target():
    helper()


def callable_instance_target():
    CALLABLE_INSTANCE()


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


def alias_target():
    alias = helper
    alias()


def reassigned_receiver_target(model):
    model = object()
    model.train()


reassigned_receiver_target.__annotations__["model"] = Model


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
        (alias_target, "global_name_unavailable"),
        (reassigned_receiver_target, "receiver_reassigned"),
        (nested_receiver_target, "call_result_receiver"),
    )

    for target, reason in cases:
        _, facts = _facts(target)
        assert all(fact.data["status"] != "resolved" for fact in facts)
        assert any(fact.data["reason"] == reason for fact in facts)


def test_static_calls_does_not_invoke_callable_instances_or_dynamic_getattr():
    _, callable_facts = _facts(callable_instance_target)
    _, dynamic_facts = _facts(dynamic_getattr_target)

    assert callable_facts[0].data["status"] == "resolved"
    assert all(fact.data["status"] != "resolved" for fact in dynamic_facts)


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


def test_static_calls_enforces_call_site_limit(monkeypatch):
    monkeypatch.setattr(static_calls, "MAX_CALL_SITES", 1)

    result, facts = _facts(direct_global_target)
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


def test_static_call_fact_rejects_unbounded_or_invalid_serialized_data():
    data = _facts(direct_global_target)[1][0].to_data()
    data["data"]["display"] = "x" * 4_097
    with pytest.raises(ValueError, match="bounded string"):
        code.CodeFact.from_data(data)

    data = _facts(direct_global_target)[1][0].to_data()
    data["data"]["status"] = "guessed"
    with pytest.raises(ValueError, match="unsupported StaticCallFact status"):
        code.CodeFact.from_data(data)
