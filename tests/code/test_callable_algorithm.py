from __future__ import annotations

import dryml.code as code
from dryml.code.callable_info import analyze_callable


class CallableInstance:
    def __call__(self, value, scale=1):
        return value * scale


def test_callable_algorithm_module_function(requirement_targets):
    result = code.analyze(requirement_targets.plain_importable_function, algorithms=("callables",))
    fact = result.facts_of_kind("callable")[0]

    assert fact.data["qualname"] == "plain_importable_function"
    assert fact.data["importable"] is True
    assert fact.data["signature"] == "(value=1)"


def test_callable_algorithm_lambda_and_local_not_importable(requirement_targets):
    local = requirement_targets.make_local_training_function()

    lambda_result = code.analyze(requirement_targets.local_lambda_with_annotation, algorithms=("callables",))
    local_result = code.analyze(local, algorithms=("callables",))

    assert lambda_result.facts_of_kind("callable")[0].data["is_lambda"] is True
    assert lambda_result.diagnostics_of_code("dryml.code.not_importable")
    assert local_result.diagnostics_of_code("dryml.code.not_importable")


def test_callable_algorithm_methods_and_callable_instance(requirement_targets):
    bound = code.analyze(requirement_targets.LightningModel().train, algorithms=("callables",))
    instance = code.analyze(CallableInstance(), algorithms=("callables",))

    assert bound.facts_of_kind("callable")[0].data["is_bound_method"] is True
    assert instance.facts_of_kind("callable")[0].data["is_callable_instance"] is True


def test_old_analyze_callable_compatibility(requirement_targets):
    info = analyze_callable(requirement_targets.plain_importable_function)

    assert info.is_function is True
    assert info.qualname == "plain_importable_function"
