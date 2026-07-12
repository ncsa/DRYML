from __future__ import annotations

import ast

import pytest

import dryml.code as code
from dryml.code.algorithms import ast_access
from dryml.code.algorithms import static_analysis
from dryml.code.algorithms.source import SourceInfo
from dryml.code.ast_tools import collect_accesses_from_source


def test_ast_access_attribute_and_method_hints(requirement_targets):
    result = code.analyze(requirement_targets.run_training, algorithms=("ast_access",))
    fact = result.facts_of_kind("ast_access")[0]

    assert "experiment.train" in fact.data["attribute_accesses"]
    assert any(call["access"] == "experiment.train" for call in fact.data["method_calls"])
    assert result.facts_of_kind("call_site")
    call = result.facts_of_kind("call_site")[0]
    assert call.data["semantic_resolution"] == "not_attempted"
    assert call.data["relative_line"] is not None
    assert call.data["absolute_line"] is not None


def test_ast_access_nested_current_behavior():
    collector = collect_accesses_from_source("def f(obj):\n    return obj.child().train()\n")

    assert any(call.chain == ("child",) for call in collector.method_calls)
    assert all(call.chain != ("child", "train") for call in collector.method_calls)


def test_ast_access_parse_failure_and_no_source(monkeypatch, requirement_targets):
    def broken_parse(source):
        raise SyntaxError("bad")

    monkeypatch.setattr(static_analysis.ast, "parse", broken_parse)
    parse_failed = code.analyze(requirement_targets.run_training, algorithms=("ast_access",))
    no_source = code.analyze(len, algorithms=("ast_access",))

    assert parse_failed.diagnostics_of_code("dryml.code.ast_parse_failed")
    assert no_source.diagnostics_of_code("dryml.code.source_unavailable")


def test_ast_access_reports_source_disabled(requirement_targets):
    result = code.analyze(
        requirement_targets.run_training,
        algorithms=("ast_access",),
        context=code.CodeAnalysisContext(allow_source=False),
    )

    assert result.diagnostics_of_code("dryml.code.source_disabled")


def test_old_ast_helper_compatibility():
    collector = collect_accesses_from_source("def f(obj):\n    return obj.value\n")

    assert any(access.root == "obj" and access.chain == ("value",) for access in collector.attr_accesses)


def test_public_ast_helper_enforces_source_and_node_bounds(monkeypatch):
    monkeypatch.setattr(ast_access, "MAX_SOURCE_BYTES", 1)
    with pytest.raises(ValueError, match="source_bytes"):
        collect_accesses_from_source("def f():\n    return None\n")

    monkeypatch.setattr(ast_access, "MAX_SOURCE_BYTES", 1_048_576)
    monkeypatch.setattr(ast_access, "MAX_AST_NODES", 1)
    with pytest.raises(ValueError, match="ast_nodes"):
        collect_accesses_from_source("def f():\n    return None\n")


def test_shared_static_parser_enforces_source_and_ast_bounds(requirement_targets, monkeypatch):
    target = code.normalize_target(requirement_targets.run_training)
    monkeypatch.setattr(static_analysis, "MAX_SOURCE_BYTES", 1)
    parsed, source_diagnostic = static_analysis.parse_static_source(
        target,
        analyzer="ast_access",
        source="def f():\n    return None\n",
        filename=None,
        start_line=None,
    )

    assert parsed is None
    assert source_diagnostic is not None
    assert source_diagnostic.data["limit_name"] == "source_bytes"

    monkeypatch.setattr(static_analysis, "MAX_SOURCE_BYTES", 1_048_576)
    monkeypatch.setattr(static_analysis, "MAX_AST_NODES", 1)
    parsed, node_diagnostic = static_analysis.parse_static_source(
        target,
        analyzer="ast_access",
        source="def f():\n    return None\n",
        filename=None,
        start_line=None,
    )

    assert parsed is None
    assert node_diagnostic is not None
    assert node_diagnostic.data["limit_name"] == "ast_nodes"


def test_ast_access_enforces_call_chain_and_scalar_bounds(monkeypatch):
    monkeypatch.setattr(ast_access, "MAX_CALL_SITES", 1)
    collector = collect_accesses_from_source("def f(obj):\n    obj.one()\n    obj.two()\n")
    assert collector.call_limit_exhausted

    monkeypatch.setattr(ast_access, "MAX_CHAIN_COMPONENTS", 1)
    collector = collect_accesses_from_source("def f(obj):\n    obj.one.two()\n")
    assert collector.method_calls[0].chain_limit_exceeded

    oversized_name = "x" * (static_analysis.MAX_STATIC_SCALAR_CHARS + 1)
    source = f"def synthetic(obj):\n    obj.{oversized_name}\n"
    monkeypatch.setattr(ast_access, "get_source_info", lambda obj: SourceInfo(source, "synthetic.py", 1))
    result = code.analyze(lambda: None, algorithms=("ast_access",))
    fact = result.facts_of_kind("ast_access")[0]
    assert fact.data["attribute_details"][0]["scalar_limit_exceeded"] is True
    assert fact.data["attribute_details"][0]["access"] == "obj.<bounded>"


def test_ast_access_enforces_source_node_and_call_bounds_end_to_end(monkeypatch):
    source = "def synthetic(obj):\n    obj.one()\n    obj.two()\n"
    monkeypatch.setattr(ast_access, "get_source_info", lambda obj: SourceInfo(source, "synthetic.py", 1))

    monkeypatch.setattr(static_analysis, "MAX_SOURCE_BYTES", 1)
    source_result = code.analyze(lambda: None, algorithms=("ast_access",))
    assert source_result.diagnostics_of_code("dryml.code.static_source_bytes_limit_exceeded")

    monkeypatch.setattr(static_analysis, "MAX_SOURCE_BYTES", 1_048_576)
    monkeypatch.setattr(static_analysis, "MAX_AST_NODES", 1)
    node_result = code.analyze(lambda: None, algorithms=("ast_access",))
    assert node_result.diagnostics_of_code("dryml.code.static_ast_nodes_limit_exceeded")

    monkeypatch.setattr(static_analysis, "MAX_AST_NODES", 100_000)
    monkeypatch.setattr(ast_access, "MAX_CALL_SITES", 1)
    call_result = code.analyze(lambda: None, algorithms=("ast_access",))
    fact = call_result.facts_of_kind("ast_access")[0]
    assert fact.data["complete"] is False
    assert call_result.diagnostics_of_code("dryml.code.static_call_sites_limit_exceeded")
