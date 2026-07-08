from __future__ import annotations

import dryml.code as code
from dryml.core2.methods import Method, traits


class ExampleMethod(Method):
    @traits(backend=None)
    def call_default(self, value):
        return value


def test_non_method_target_skips_cleanly(requirement_targets):
    result = code.analyze(requirement_targets.plain_importable_function, algorithms=("method_contracts",))

    assert result.facts == ()
    assert result.diagnostics == ()


def test_method_target_emits_basic_contract_fact():
    result = code.analyze(ExampleMethod, algorithms=("method_contracts",))
    fact = result.facts_of_kind("method_contract")[0]

    assert fact.data["method_contract_detected"] is True
    assert fact.data["trait_impls"][0]["name"] == "call_default"


def test_general_analyze_does_not_crash_on_method_target():
    result = code.analyze(ExampleMethod)

    assert result.facts_of_kind("method_contract")
    assert result.ok is True
