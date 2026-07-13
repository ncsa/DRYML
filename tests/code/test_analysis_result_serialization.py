from __future__ import annotations

import json

import dryml.code as code
import pytest


def test_fact_and_diagnostic_serialization_round_trip():
    fact = code.CodeFact("custom", source={"nested": [1]}, data={"object": object()})
    diagnostic = code.DiagnosticFact(severity="warning", code="dryml.code.test", message="test")

    fact_data = fact.to_data()
    diagnostic_data = diagnostic.to_data()

    assert code.CodeFact.from_data(fact_data).kind == "custom"
    assert code.DiagnosticFact.from_data(diagnostic_data).code == "dryml.code.test"
    json.dumps(fact_data)
    json.dumps(diagnostic_data)


def test_requirement_fact_serialization_preserves_fields():
    fact = code.RequirementFact(
        namespace="environment",
        requirement_kind="default",
        fragment={"requirements": ["numpy"]},
        priority=5,
        merge_policy="append",
    )

    data = fact.to_data()
    restored = code.CodeFact.from_data(data)

    assert data["requirement_kind"] == "default"
    assert restored.to_data()["merge_policy"] == "append"


def test_analysis_result_serializes_and_round_trips(requirement_targets):
    result = code.analyze(requirement_targets.run_training)
    data = result.to_data()
    restored = code.CodeAnalysisResult.from_data(data)

    assert result.facts_of_kind("callable")
    assert result.facts_of_kind("source")
    assert result.ok is True
    assert restored.target.import_path == result.target.import_path
    assert restored.facts_of_kind("requirement")
    json.dumps(data)


def test_ok_property_false_for_errors_and_true_for_warnings():
    target = code.CodeTargetSpec("unknown")
    warning_result = code.CodeAnalysisResult(target, diagnostics=(code.DiagnosticFact(severity="warning"),))
    error_result = code.CodeAnalysisResult(target, diagnostics=(code.DiagnosticFact(severity="error"),))

    assert warning_result.ok is True
    assert error_result.ok is False
    assert error_result.diagnostics_of_code("dryml.code.diagnostic")


def test_dynamic_call_fact_round_trips_typed_with_fixed_schema():
    fact = code.DynamicCallFact(
        source={"analyzer": "dynamic_trace", "target_kind": "local_function"},
        data={
            "sequence": 0,
            "receiver_kind": "definition",
            "receiver_ref": "0123456789abcdef",
            "receiver_class": None,
            "method_name": "train",
            "args": [{"definition_kind": "concrete_definition", "definition_ref": "cdef-v4-0123456789abcdef"}],
            "kwargs": {"epochs": 2},
            "method_facts": [],
        },
    )
    restored = code.CodeFact.from_data(json.loads(json.dumps(fact.to_data())))
    assert isinstance(restored, code.DynamicCallFact)
    assert restored == fact


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(extra=True),
        lambda data: data["source"].update(extra=True),
        lambda data: data["data"].update(extra=True),
        lambda data: data["source"].update(analyzer="static_calls"),
        lambda data: data["source"].update(target_kind=""),
        lambda data: data["data"].pop("sequence"),
        lambda data: data["data"].update(sequence=True),
        lambda data: data["data"].update(sequence=-1),
        lambda data: data["data"].update(receiver_kind="unknown"),
        lambda data: data["data"].update(receiver_ref="cdef-v4-0123456789abcdef"),
        lambda data: data["data"].update(receiver_class="not-an-import-path"),
        lambda data: data["data"].update(method_name="__class__"),
        lambda data: data["data"].update(method_name="m" * 513),
        lambda data: data["data"].update(args=object()),
        lambda data: data["data"].update(args=()),
        lambda data: data["data"].update(kwargs=[]),
        lambda data: data["data"].update(kwargs={1: None}),
        lambda data: data["data"].update(method_facts={}),
        lambda data: data["data"].update(method_facts=[{"kind": "unknown"}]),
    ],
)
def test_dynamic_call_fact_rejects_malformed_or_extra_data(mutation):
    data = {
        "kind": "dynamic_call",
        "source": {"analyzer": "dynamic_trace", "target_kind": "function"},
        "data": {
            "sequence": 0,
            "receiver_kind": "definition",
            "receiver_ref": "0123456789abcdef",
            "receiver_class": None,
            "method_name": "train",
            "args": [],
            "kwargs": {},
            "method_facts": [],
        },
    }
    mutation(data)
    with pytest.raises((TypeError, ValueError)):
        code.CodeFact.from_data(data)


def test_dynamic_call_fact_rejects_wrong_kind_directly():
    with pytest.raises(ValueError, match="kind"):
        code.DynamicCallFact(
            kind="not_dynamic_call",
            source={"analyzer": "dynamic_trace", "target_kind": "function"},
            data={
                "sequence": 0,
                "receiver_kind": "definition",
                "receiver_ref": "0123456789abcdef",
                "receiver_class": None,
                "method_name": "train",
                "args": [],
                "kwargs": {},
                "method_facts": [],
            },
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda fact: fact.pop("source"),
        lambda fact: fact.update(extra=True),
        lambda fact: fact.update(source=[]),
        lambda fact: fact.update(data=[]),
        lambda fact: fact.pop("namespace"),
        lambda fact: fact.update(namespace="environment", extra=True),
        lambda fact: fact.update(fragment=[]),
        lambda fact: fact.update(priority=True),
    ],
)
def test_dynamic_call_fact_rejects_malformed_nested_method_fact_wire(mutate):
    method_fact = {
        "kind": "requirement",
        "source": {"analyzer": "direct_annotations"},
        "data": {"annotation": {}},
        "namespace": "environment",
        "requirement_kind": "requirement",
        "fragment": {"requirements": ["numpy"]},
        "priority": 0,
        "merge_policy": None,
    }
    mutate(method_fact)
    data = {
        "kind": "dynamic_call",
        "source": {"analyzer": "dynamic_trace", "target_kind": "function"},
        "data": {
            "sequence": 0,
            "receiver_kind": "definition",
            "receiver_ref": "0123456789abcdef",
            "receiver_class": None,
            "method_name": "train",
            "args": [],
            "kwargs": {},
            "method_facts": [method_fact],
        },
    }

    with pytest.raises((TypeError, ValueError)):
        code.CodeFact.from_data(data)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda fact: fact.update(namespace="not a namespace"),
        lambda fact: fact.update(requirement_kind="other"),
        lambda fact: fact["source"].update(analyzer="other"),
        lambda fact: fact["source"].update(target_kind=""),
        lambda fact: fact["data"].pop("resolution"),
        lambda fact: fact["data"]["annotation"].update(schema_version=2),
    ],
)
def test_dynamic_call_fact_rejects_semantically_invalid_nested_requirement_fact(mutate):
    import dryml

    @dryml.env.req(requirements=("numpy",))
    def annotated_target():
        pass

    requirement = next(
        fact.to_data()
        for fact in code.analyze(annotated_target, algorithms=("direct_annotations",)).facts
        if fact.kind == "requirement"
    )
    mutate(requirement)
    data = {
        "kind": "dynamic_call",
        "source": {"analyzer": "dynamic_trace", "target_kind": "function"},
        "data": {
            "sequence": 0,
            "receiver_kind": "definition",
            "receiver_ref": "0123456789abcdef",
            "receiver_class": None,
            "method_name": "train",
            "args": [],
            "kwargs": {},
            "method_facts": [requirement],
        },
    }

    with pytest.raises((TypeError, ValueError)):
        code.CodeFact.from_data(data)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda fact: fact["data"].update(source_trace=[]),
        lambda fact: fact["data"]["source_trace"].update(namespace="world"),
        lambda fact: fact["data"]["source_trace"].update(fragment_index=1),
        lambda fact: fact["data"]["resolution"].pop("source_traces"),
        lambda fact: fact["data"]["resolution"]["source_traces"][0]["data"].update(source={"kind": "synthetic", "target": None, "label": None, "namespace": None, "path": None, "metadata": {}}),
        lambda fact: fact["data"]["resolution"]["fragments"][0].update(priority=1),
    ],
)
def test_dynamic_call_fact_rejects_inconsistent_requirement_trace_or_resolution(mutate):
    import dryml

    @dryml.env.req(requirements=("numpy",))
    def annotated_target():
        pass

    requirement = next(
        fact.to_data()
        for fact in code.analyze(annotated_target, algorithms=("direct_annotations",)).facts
        if fact.kind == "requirement"
    )
    mutate(requirement)
    data = {
        "kind": "dynamic_call",
        "source": {"analyzer": "dynamic_trace", "target_kind": "function"},
        "data": {
            "sequence": 0,
            "receiver_kind": "definition",
            "receiver_ref": "0123456789abcdef",
            "receiver_class": None,
            "method_name": "train",
            "args": [],
            "kwargs": {},
            "method_facts": [requirement],
        },
    }

    with pytest.raises((TypeError, ValueError)):
        code.CodeFact.from_data(data)


@pytest.mark.parametrize(
    "method_fact",
    [
        {
            "kind": "method_contract",
            "source": {"analyzer": "method_contracts", "target_kind": "class"},
            "data": {
                "method_contract_detected": True,
                "class_module": None,
                "class_qualname": None,
                "trait_impls": [{"name": "run", "traits": {"backend": "invalid", "batch_mode": None}}],
                "has_user_call": False,
            },
        },
        {
            "kind": "shape",
            "source": {"analyzer": "method_contracts", "target_kind": "class"},
            "data": {"input_handles": [], "output_handles": []},
        },
        {
            "kind": "shape",
            "source": {"analyzer": "method_contracts", "target_kind": "class"},
            "data": {"input_handles": [], "output_handles": {}},
        },
    ],
)
def test_dynamic_call_fact_validates_method_contract_domains_and_shape_wire(method_fact):
    data = {
        "kind": "dynamic_call",
        "source": {"analyzer": "dynamic_trace", "target_kind": "function"},
        "data": {
            "sequence": 0,
            "receiver_kind": "definition",
            "receiver_ref": "0123456789abcdef",
            "receiver_class": None,
            "method_name": "train",
            "args": [],
            "kwargs": {},
            "method_facts": [method_fact],
        },
    }

    if method_fact["kind"] == "shape" and isinstance(method_fact["data"]["output_handles"], list):
        restored = code.CodeFact.from_data(data)
        assert isinstance(restored, code.DynamicCallFact)
    else:
        with pytest.raises((TypeError, ValueError)):
            code.CodeFact.from_data(data)


@pytest.mark.parametrize(
    ("receiver_kind", "prefix"),
    [("definition", ""), ("concrete_definition", "cdef-v4-")],
)
def test_dynamic_call_receiver_reference_exact_boundary_and_n_plus_one(
    receiver_kind,
    prefix,
):
    exact_ref = prefix + "a" * (4_096 - len(prefix))

    def make_fact(reference):
        return code.DynamicCallFact(
            source={"analyzer": "dynamic_trace", "target_kind": "function"},
            data={
                "sequence": 0,
                "receiver_kind": receiver_kind,
                "receiver_ref": reference,
                "receiver_class": None,
                "method_name": "train",
                "args": [],
                "kwargs": {},
                "method_facts": [],
            },
        )

    assert make_fact(exact_ref).data["receiver_ref"] == exact_ref
    with pytest.raises(ValueError, match="bounded reference"):
        make_fact(exact_ref + "a")


def test_dynamic_call_fact_serialized_byte_exact_boundary_and_n_plus_one():
    limit = 1_048_576
    source = {"analyzer": "dynamic_trace", "target_kind": "function"}

    def wire(args):
        return {
            "kind": "dynamic_call",
            "source": source,
            "data": {
                "sequence": 0,
                "receiver_kind": "definition",
                "receiver_ref": "0123456789abcdef",
                "receiver_class": None,
                "method_name": "train",
                "args": args,
                "kwargs": {},
                "method_facts": [],
            },
        }

    def encoded_size(value):
        return len(json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"))

    exact_args = None
    for count in range(1, 300):
        remaining = limit - encoded_size(wire([""] * count))
        if 0 <= remaining <= count * 4_096:
            exact_args = []
            for _ in range(count):
                size = min(4_096, remaining)
                exact_args.append("x" * size)
                remaining -= size
            break
    assert exact_args is not None
    exact_wire = wire(exact_args)
    assert encoded_size(exact_wire) == limit
    fact = code.CodeFact.from_data(exact_wire)
    assert isinstance(fact, code.DynamicCallFact)

    over_args = list(exact_args)
    index = next(index for index, value in enumerate(over_args) if len(value) < 4_096)
    over_args[index] += "x"
    assert encoded_size(wire(over_args)) == limit + 1
    with pytest.raises(ValueError, match="serialized byte limit"):
        code.CodeFact.from_data(wire(over_args))
