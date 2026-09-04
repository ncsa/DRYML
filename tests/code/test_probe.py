"""Tests for the in-process static probe entry point."""

from __future__ import annotations

import pytest

from dryml.code import AnalysisKernel, KernelCall, SourceTarget, analyze, probe
from dryml.code.errors import InvalidKernelError


class ValueKernel(AnalysisKernel[int, int]):
    """Return its typed input through the static analysis API."""

    input_type = int
    output_type = int

    def run(self, graph: object, value: int, context: object) -> int:
        """Return the consumer-owned input value."""

        return value


class TraceKernel(AnalysisKernel[None, None]):
    """Declare a trace-mode call rejected by static entry points."""

    input_type = type(None)
    output_type = type(None)
    mode = "trace"

    def run(self, graph: object, value: None, context: object) -> None:
        """Provide an implementation that static APIs must not invoke."""

        raise AssertionError("trace kernel executed")


def test_probe_is_identical_to_static_analyze() -> None:
    """Probe delegates exactly to the static analysis behavior and result shape."""

    target = SourceTarget("def subject():\n    return 1\n", name="subject")
    calls = (KernelCall(ValueKernel(), 4),)

    assert probe(target, calls) == analyze(target, calls)


def test_static_entry_points_reject_trace_without_target_execution() -> None:
    """Trace declarations fail before target normalization or kernel execution."""

    for entry in (analyze, probe):
        with pytest.raises(InvalidKernelError) as error:
            entry(object(), (KernelCall(TraceKernel(), None),))
        assert error.value.code == "kernel.invalid"
