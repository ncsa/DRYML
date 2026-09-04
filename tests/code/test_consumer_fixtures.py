"""Neutral consumer-owned probe fixtures for the generic code kernel boundary."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import dryml.code as code


@dataclass(frozen=True, slots=True)
class EnvironmentStyleResult:
    """Consumer-owned compatibility result, intentionally outside dryml.code."""

    target_kind: str
    accepted: bool


@dataclass(frozen=True, slots=True)
class HostResourceStyleResult:
    """Consumer-owned resource result, intentionally outside dryml.code."""

    syntax_nodes: int
    requested_label: str


class EnvironmentStyleKernel(code.AnalysisKernel[bool, EnvironmentStyleResult]):
    """Fixture consumer that validates an environment-like policy itself."""

    input_type = bool
    output_type = EnvironmentStyleResult

    def run(self, graph, value, context):
        return EnvironmentStyleResult(graph.target.kind, value and graph.target.kind == "function")


class HostResourceStyleKernel(code.AnalysisKernel[str, HostResourceStyleResult]):
    """Fixture consumer that projects neutral graph evidence into its own type."""

    input_type = str
    output_type = HostResourceStyleResult

    def run(self, graph, value, context):
        return HostResourceStyleResult(sum(node.kind == "syntax" for node in graph.nodes), value)


def fixture_target(value):
    """Provide a file-backed target whose body must not execute during probing."""

    raise AssertionError("probe must not invoke target bodies")


def test_probe_returns_consumer_owned_environment_and_resource_results(monkeypatch):
    """Probe remains in-process and leaves consumer semantics outside dryml.code."""

    def forbidden(*args, **kwargs):
        raise AssertionError("probe must not create processes or inspect host resources")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "cpu_count", forbidden)
    result = code.probe(
        fixture_target,
        (
            code.KernelCall(EnvironmentStyleKernel(), True),
            code.KernelCall(HostResourceStyleKernel(), "fixture-resource"),
        ),
    )

    environment = result.require(EnvironmentStyleKernel)
    resources = result.require(HostResourceStyleKernel)
    assert environment == EnvironmentStyleResult("function", True)
    assert resources.syntax_nodes > 0
    assert resources.requested_label == "fixture-resource"
    assert result.facts == ()
    assert not hasattr(code, "EnvironmentStyleResult")
    assert not hasattr(code, "HostResourceStyleResult")
