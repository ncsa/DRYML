from __future__ import annotations

import json

import dryml.code as code


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
