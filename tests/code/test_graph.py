"""Tests for deterministic immutable foundational program graphs."""

from __future__ import annotations

from dataclasses import replace
import subprocess
import sys

import pytest

from dryml.code import CodeAnalysisError, ImportTarget, ProgramGraph, SourceTarget, SourceUnavailableError
from dryml.code.graph import ProgramEdge, build_program_graph


_SOURCE = """def subject(value):
    result = client.worker(value)
    return result
"""


def test_build_program_graph_is_deterministic_and_queryable() -> None:
    """Equivalent source produces canonical relationships and query tuples."""

    first = build_program_graph(SourceTarget(_SOURCE, name="subject", filename="subject.py"))
    second = build_program_graph(SourceTarget(_SOURCE, name="subject", filename="subject.py"))

    assert first == second
    assert first.digest == second.digest
    assert first.nodes[0].kind == "target"
    assert first.nodes_of_kind("attribute_access") == tuple(
        node for node in first.nodes if node.kind == "attribute_access"
    )
    assert first.edges_of_kind("call") == tuple(edge for edge in first.edges if edge.kind == "call")

    call = first.edges_of_kind("call")[0]
    assert first.successors(call.source, kind="call") == (next(node for node in first.nodes if node.id == call.target),)
    assert first.predecessors(call.target, kind="call") == (next(node for node in first.nodes if node.id == call.source),)


def test_graph_uses_absolute_lines_and_utf8_byte_columns() -> None:
    """Syntax provenance retains absolute lines and CPython byte columns."""

    graph = build_program_graph(
        SourceTarget("def subject():\n    é = obj.value\n", name="subject", filename="input.py", start_line=40)
    )

    access = graph.nodes_of_kind("attribute_access")[0]

    assert access.source is not None
    assert access.source.line == 41
    assert access.source.column == 9


def test_unavailable_nonrequired_source_returns_root_and_diagnostic() -> None:
    """An admitted import module has target evidence and no guessed source facts."""

    graph = build_program_graph(ImportTarget("math"))

    assert tuple(node.kind for node in graph.nodes) == ("target",)
    assert graph.edges == ()
    assert graph.diagnostics[0].code == "source.unavailable"


def test_explicit_malformed_source_fails_before_graph_construction() -> None:
    """Malformed explicit text remains a typed pre-graph source failure."""

    with pytest.raises(SourceUnavailableError) as error:
        build_program_graph(SourceTarget("def subject(:\n", name="subject"))

    assert error.value.code == "source.invalid"


def test_program_graph_rejects_invalid_parts_and_queries() -> None:
    """Closed vocabulary, endpoints, duplicates, and queries fail as graph errors."""

    graph = build_program_graph(SourceTarget(_SOURCE, name="subject"))
    target = graph.nodes[0]
    syntax = graph.nodes_of_kind("syntax")[0]
    containment = next(edge for edge in graph.edges if edge.kind == "containment")

    invalid_graphs = (
        lambda: ProgramGraph(graph.target, (replace(target, kind="unknown"),) + graph.nodes[1:], graph.edges, graph.diagnostics),
        lambda: ProgramGraph(graph.target, (replace(target, value=(("unknown", "value"),)),) + graph.nodes[1:], graph.edges, graph.diagnostics),
        lambda: ProgramGraph(graph.target, (replace(target, id="0" * 64),) + graph.nodes[1:], (), graph.diagnostics),
        lambda: ProgramGraph(graph.target, (target, target), (), graph.diagnostics),
        lambda: ProgramGraph(graph.target, graph.nodes, (containment, containment), graph.diagnostics),
        lambda: ProgramGraph(graph.target, graph.nodes, (ProgramEdge(target.id, "missing", "containment"),), graph.diagnostics),
        lambda: ProgramGraph(graph.target, graph.nodes, (ProgramEdge(syntax.id, target.id, "call"),), graph.diagnostics),
    )
    for invalid in invalid_graphs:
        with pytest.raises(CodeAnalysisError) as error:
            invalid()
        assert error.value.code == "graph.invalid"

    for operation in (
        lambda: graph.nodes_of_kind("unknown"),
        lambda: graph.edges_of_kind("unknown"),
        lambda: graph.successors("missing"),
        lambda: graph.predecessors("missing"),
        lambda: graph.successors(target.id, kind="unknown"),
    ):
        with pytest.raises(CodeAnalysisError) as error:
            operation()
        assert error.value.code == "graph.invalid"


def test_graph_identity_avoids_addresses_repr_and_hash_randomization() -> None:
    """The digest is stable across fresh interpreters and contains no live identity."""

    graph = build_program_graph(SourceTarget(_SOURCE, name="subject"))
    program = (
        "from dryml.code import SourceTarget; from dryml.code.graph import build_program_graph; "
        f"print(build_program_graph(SourceTarget({_SOURCE!r}, name='subject')).digest)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == graph.digest
    assert "0x" not in graph.digest
    assert "object at" not in repr(graph.nodes)


def test_program_graph_canonicalizes_incidental_node_and_edge_order() -> None:
    """Construction canonicalizes valid tuple and closed-mapping order."""

    graph = build_program_graph(SourceTarget(_SOURCE, name="subject"))
    call = graph.nodes_of_kind("static_call")[0]
    reordered_call = replace(call, value=tuple(reversed(call.value)))
    reordered_nodes = tuple(reordered_call if node.id == call.id else node for node in reversed(graph.nodes))

    reordered = ProgramGraph(graph.target, reordered_nodes, tuple(reversed(graph.edges)), graph.diagnostics)

    assert reordered == graph
    assert reordered.digest == graph.digest
