from __future__ import annotations

import json

import dryml.code as code


def test_symbol_capture_importable_function_and_class(requirement_targets):
    func_result = code.analyze(requirement_targets.plain_importable_function, algorithms=("symbol_capture",))
    class_result = code.analyze(requirement_targets.LightningModel, algorithms=("symbol_capture",))

    assert func_result.facts_of_kind("symbol")[0].data["import_path"] == "dryml_requirement_targets:plain_importable_function"
    assert class_result.facts_of_kind("symbol")[0].data["symbol_kind"] == "import_ref"


def test_symbol_capture_local_lambda_and_closure_diagnostics(requirement_targets):
    local = requirement_targets.make_local_training_function()
    value = 1

    def closure():
        return value

    local_result = code.analyze(local, algorithms=("symbol_capture",))
    lambda_result = code.analyze(requirement_targets.local_lambda_with_annotation, algorithms=("symbol_capture",))
    closure_result = code.analyze(closure, algorithms=("symbol_capture",))

    assert local_result.facts_of_kind("symbol") or local_result.diagnostics
    assert lambda_result.facts_of_kind("symbol") or lambda_result.diagnostics
    assert closure_result.diagnostics_of_code("dryml.code.closure_unsupported")


def test_symbol_fact_serializes_to_json(requirement_targets):
    result = code.analyze(requirement_targets.plain_importable_function, algorithms=("symbol_capture",))

    json.dumps(result.to_data())
