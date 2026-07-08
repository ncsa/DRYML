from __future__ import annotations

import ast

import pytest

import dryml.code as code
from dryml.code.algorithms import ast_access
from dryml.code.ast_tools import collect_accesses_from_source


def test_ast_access_attribute_and_method_hints(requirement_targets):
    result = code.analyze(requirement_targets.run_training, algorithms=("ast_access",))
    fact = result.facts_of_kind("ast_access")[0]

    assert "experiment.train" in fact.data["attribute_accesses"]
    assert any(call["access"] == "experiment.train" for call in fact.data["method_calls"])
    assert result.facts_of_kind("call_site")


def test_ast_access_nested_current_behavior():
    collector = collect_accesses_from_source("def f(obj):\n    return obj.child().train()\n")

    assert any(call.chain == ("child",) for call in collector.method_calls)
    assert all(call.chain != ("child", "train") for call in collector.method_calls)


def test_ast_access_parse_failure_and_no_source(monkeypatch, requirement_targets):
    def broken_parse(source):
        raise SyntaxError("bad")

    monkeypatch.setattr(ast_access, "collect_accesses_from_source", broken_parse)
    parse_failed = code.analyze(requirement_targets.run_training, algorithms=("ast_access",))
    no_source = code.analyze(len, algorithms=("ast_access",))

    assert parse_failed.diagnostics_of_code("dryml.code.ast_parse_failed")
    assert no_source.diagnostics_of_code("dryml.code.source_unavailable")


def test_old_ast_helper_compatibility():
    collector = collect_accesses_from_source("def f(obj):\n    return obj.value\n")

    assert any(access.root == "obj" and access.chain == ("value",) for access in collector.attr_accesses)
