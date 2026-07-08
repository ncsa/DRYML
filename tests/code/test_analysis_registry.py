from __future__ import annotations

import pytest

import dryml.code as code


def test_default_analyzers_are_registered():
    names = code.available_analyzers()

    assert {"callables", "source", "ast_access", "symbol_capture", "direct_annotations", "method_contracts"}.issubset(names)
    assert code.get_analyzer("callables").name == "callables"


def test_register_duplicate_and_replace(requirement_targets):
    analyzer = code.FunctionAnalyzer("test_duplicate_analyzer", lambda target, context: code.CodeAnalysisResult(target.spec))
    replacement = code.FunctionAnalyzer("test_duplicate_analyzer", lambda target, context: code.CodeAnalysisResult(target.spec))

    code.register_analyzer(analyzer, replace=True)
    with pytest.raises(ValueError):
        code.register_analyzer(replacement)
    code.register_analyzer(replacement, replace=True)
    assert code.get_analyzer("test_duplicate_analyzer") is replacement


def test_explicit_algorithm_order_is_preserved(requirement_targets):
    seen = []

    def first(target, context):
        seen.append("first")
        return code.CodeAnalysisResult(target.spec, facts=(code.CodeFact("order", data={"name": "first"}),))

    def second(target, context):
        seen.append("second")
        return code.CodeAnalysisResult(target.spec, facts=(code.CodeFact("order", data={"name": "second"}),))

    code.register_analyzer(code.FunctionAnalyzer("test_order_first", first), replace=True)
    code.register_analyzer(code.FunctionAnalyzer("test_order_second", second), replace=True)

    result = code.analyze(requirement_targets.plain_importable_function, algorithms=("test_order_first", "test_order_second"))

    assert seen == ["first", "second"]
    assert [fact.data["name"] for fact in result.facts] == ["first", "second"]


def test_analyzer_exception_collect_and_raise_policies(requirement_targets):
    def broken(target, context):
        raise RuntimeError("boom")

    code.register_analyzer(code.FunctionAnalyzer("test_broken_analyzer", broken), replace=True)

    collected = code.analyze(requirement_targets.plain_importable_function, algorithms=("test_broken_analyzer",))
    assert collected.diagnostics_of_code("dryml.code.algorithm_failed")

    context = code.CodeAnalysisContext(algorithms=("test_broken_analyzer",), diagnostics_policy="raise")
    with pytest.raises(code.CodeAnalysisError):
        code.analyze(requirement_targets.plain_importable_function, context=context)


def test_unknown_analyzer_becomes_diagnostic(requirement_targets):
    result = code.analyze(requirement_targets.plain_importable_function, algorithms=("does_not_exist",))

    assert result.diagnostics_of_code("dryml.code.unknown_analyzer")
