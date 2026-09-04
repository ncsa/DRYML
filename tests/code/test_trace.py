"""Tests for bounded current-thread dynamic code tracing."""

from __future__ import annotations

import subprocess
import sys

import pytest

from dryml.code import AnalysisKernel, CodeFact, KernelCall, SourceTarget, analyze, probe, trace
from dryml.code.errors import CodeAnalysisError
from dryml.code.targets import DescriptorTarget, ImportTarget


_CALLS: list[str] = []
_FINISHED: list[bool] = []
_FIRST_GLOBAL = 1
_SECOND_GLOBAL = 2


class StaticFact(AnalysisKernel[None, CodeFact]):
    """Record that static work completed before a traced invocation."""

    input_type = type(None)
    output_type = CodeFact

    def run(self, graph: object, value: None, context: object) -> CodeFact:
        """Return static evidence without invoking the target."""

        _CALLS.append("static")
        return CodeFact("static", 1)


class TraceFact(AnalysisKernel[None, CodeFact]):
    """Record the derived graph received by a trace-mode consumer."""

    input_type = type(None)
    output_type = CodeFact
    mode = "trace"
    requires = (StaticFact,)

    def run(self, graph: object, value: None, context: object) -> CodeFact:
        """Return a trace-owned fact after inspecting dynamic evidence."""

        _CALLS.append("trace")
        return CodeFact("events", sum(node.kind == "trace_event" for node in graph.nodes))


class TraceDependent(AnalysisKernel[None, int]):
    """Require trace evidence to exercise invocation failure skips."""

    input_type = type(None)
    output_type = int
    mode = "trace"
    requires = (TraceFact,)

    def run(self, graph: object, value: None, context: object) -> int:
        """Return a value only when the trace producer succeeds."""

        _CALLS.append("dependent")
        return 1


def _target(value: int = 1) -> int:
    """Provide a file-backed target with a Python descendant call."""

    return _child(value)


def _child(value: int) -> int:
    """Provide a descendant frame for trace event capture."""

    return value + 1


def _raising_target() -> None:
    """Raise an ordinary target exception containing private text."""

    raise RuntimeError("/private/path trace-secret")


def _interrupting_target() -> None:
    """Raise an interruption-style exception that must propagate."""

    raise KeyboardInterrupt()


def _busy_target() -> None:
    """Finish a multi-line body after trace capture overflows."""

    for _ in range(10):
        _FINISHED.append(True)


def _overflow_raising_target() -> None:
    """Overflow trace capture before raising an ordinary target exception."""

    for _ in range(10):
        pass
    raise RuntimeError("private overflow failure")


def _same_line_lambda_target() -> None:
    """Invoke same-line lambdas whose constants must distinguish code IDs."""

    first = lambda: 1 + 2j; second = lambda: 3 + 4j
    first()
    second()


def _heterogeneous_frozenset_target(value: object = None) -> bool:
    """Exercise a compiler-folded constant with incomparable scalar values."""

    return value in {None, 1}


def _same_line_name_target() -> None:
    """Invoke same-line lambdas distinguished only by referenced names."""

    first = lambda: _FIRST_GLOBAL; second = lambda: _SECOND_GLOBAL
    first()
    second()


def _displacing_target() -> None:
    """Remove DRYML's active hook to simulate incomplete trace capture."""

    sys.settrace(None)


async def _coroutine_target() -> None:
    """Provide an unsupported coroutine target."""


def _generator_target() -> object:
    """Provide an unsupported generator target."""

    yield 1


async def _async_generator_target() -> object:
    """Provide an unsupported async-generator target."""

    yield 1


def _run_child(program: str) -> None:
    """Run a lifecycle assertion without the parent pytest coverage trace hook."""

    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_trace_executes_once_after_static_work_and_binds_distinct_graphs() -> None:
    """Trace invokes once, preserves static provenance, and derives a graph."""

    _run_child(
        "from dryml.code import KernelCall, trace; "
        "from tests.code.test_trace import StaticFact, TraceFact, _CALLS, _target; "
        "_CALLS.clear(); result = trace(_target, (KernelCall(TraceFact(), None), KernelCall(StaticFact(), None))); "
        "assert _CALLS == ['static', 'trace']; "
        "assert result.invocation.status == 'succeeded'; "
        "assert result.base_graph is not result.graph and result.base_graph.digest != result.graph.digest; "
        "assert result.outcomes[0].graph_digest == result.graph.digest; "
        "assert result.outcomes[1].graph_digest == result.base_graph.digest; "
        "assert tuple(record.origin for record in result.facts) == ('trace', 'static'); "
        "events = {node.id: dict(node.value) for node in result.graph.nodes_of_kind('trace_event')}; "
        "assert events; "
        "assert all(events[edge.target]['depth'] == events[edge.source]['depth'] + 1 for edge in result.graph.edges_of_kind('frame_descent'))"
    )


def test_static_analysis_and_probe_never_invoke_trace_target() -> None:
    """Static entry points retain their no-invocation contract."""

    def forbidden() -> None:
        raise AssertionError("target was invoked")

    assert analyze(forbidden, ()).invocation is None
    assert probe(forbidden, ()).invocation is None


@pytest.mark.parametrize(
    "target",
    (
        SourceTarget("def subject():\n    return 1\n", name="subject"),
        ImportTarget("math"),
        DescriptorTarget(dict, "get"),
        int,
        len,
        (value for value in (1,)),
        _coroutine_target,
        _generator_target,
        _async_generator_target,
    ),
)
def test_trace_rejects_non_live_synchronous_function_forms_before_kernels(target: object) -> None:
    """Unsupported target wrappers and callables fail without consumer execution."""

    class MustNotRun(AnalysisKernel[None, int]):
        input_type = type(None)
        output_type = int

        def run(self, graph: object, value: None, context: object) -> int:
            raise AssertionError("kernel executed")

    with pytest.raises(CodeAnalysisError) as error:
        trace(target, (KernelCall(MustNotRun(), None),))  # type: ignore[arg-type]
    assert error.value.code == "trace.unsupported"


@pytest.mark.parametrize("maximum", (0, -1, True, 100_001))
def test_trace_validates_event_bound_first(maximum: object) -> None:
    """Invalid bounds fail before declaration validation or target execution."""

    with pytest.raises(CodeAnalysisError) as error:
        trace(_target, (), max_events=maximum)  # type: ignore[arg-type]
    assert error.value.code == "trace.limit"


def test_trace_rejects_existing_hook_without_changing_it() -> None:
    """A current-thread hook blocks trace admission before consumers run."""

    previous = sys.gettrace()

    def existing(frame: object, event: str, argument: object) -> object:
        return existing

    sys.settrace(existing)
    try:
        with pytest.raises(CodeAnalysisError) as error:
            trace(_target, ())
        assert error.value.code == "trace.hook_active"
        assert sys.gettrace() is existing
    finally:
        sys.settrace(previous)


def test_target_exception_is_redacted_and_skips_trace_dependents() -> None:
    """An ordinary target error preserves static output and fails closed."""

    _run_child(
        "from dryml.code import KernelCall, trace; "
        "from tests.code.test_trace import StaticFact, TraceDependent, TraceFact, _CALLS, _raising_target; "
        "_CALLS.clear(); result = trace(_raising_target, (KernelCall(TraceDependent(), None), KernelCall(StaticFact(), None), KernelCall(TraceFact(), None))); "
        "assert _CALLS == ['static']; "
        "assert result.invocation.status == 'failed'; "
        "assert result.invocation.diagnostic.code == 'trace.invocation'; "
        "assert tuple(outcome.status for outcome in result.outcomes) == ('skipped', 'succeeded', 'failed'); "
        "assert result.require(StaticFact).kind == 'static'"
    )


def test_trace_reraises_interruption_after_restoring_hook() -> None:
    """Interruption-style target failures return no partial result."""

    _run_child(
        "import sys; from dryml.code import trace; "
        "from tests.code.test_trace import _interrupting_target; "
        "exec(\"try:\\n    trace(_interrupting_target, ())\\nexcept KeyboardInterrupt:\\n    pass\\nelse:\\n    raise AssertionError('interruption was swallowed')\"); "
        "assert sys.gettrace() is None"
    )


def test_trace_callback_reraises_interruption_after_cleanup() -> None:
    """Interruption during event projection is not converted to a result."""

    _run_child(
        "import importlib, sys; from dryml.code import trace; "
        "from tests.code.test_trace import _target; "
        "module = importlib.import_module('dryml.code.trace'); "
        "exec(\"def interrupt(frame, cache):\\n    raise KeyboardInterrupt()\"); "
        "module._code_id = interrupt; "
        "exec(\"try:\\n    trace(_target, ())\\nexcept KeyboardInterrupt:\\n    pass\\nelse:\\n    raise AssertionError('callback interruption was swallowed')\"); "
        "assert sys.gettrace() is None"
    )


def test_trace_overflow_stops_trace_outputs_but_finishes_target() -> None:
    """Overflow preserves completed static work and rejects all trace outputs."""

    _run_child(
        "from dryml.code import KernelCall, trace; "
        "from tests.code.test_trace import StaticFact, TraceFact, _FINISHED, _busy_target; "
        "_FINISHED.clear(); result = trace(_busy_target, (KernelCall(StaticFact(), None), KernelCall(TraceFact(), None)), max_events=1); "
        "assert len(_FINISHED) == 10; "
        "assert tuple(outcome.status for outcome in result.outcomes) == ('succeeded', 'failed'); "
        "assert result.outcomes[1].diagnostics[0].code == 'trace.limit'; "
        "assert tuple(record.producer for record in result.facts) == (StaticFact,)"
    )


def test_trace_overflow_preserves_target_failure_outcome() -> None:
    """A target exception remains visible when trace capture also overflows."""

    _run_child(
        "from dryml.code import KernelCall, trace; "
        "from tests.code.test_trace import StaticFact, TraceFact, _overflow_raising_target; "
        "result = trace(_overflow_raising_target, (KernelCall(StaticFact(), None), KernelCall(TraceFact(), None)), max_events=1); "
        "assert result.invocation.status == 'failed'; "
        "assert result.invocation.diagnostic.code == 'trace.invocation'; "
        "assert result.outcomes[1].diagnostics[0].code == 'trace.limit'"
    )


def test_trace_overflow_without_trace_kernels_is_incomplete() -> None:
    """Graph-only callers still receive explicit overflow diagnostics."""

    _run_child(
        "from dryml.code import trace; "
        "from tests.code.test_trace import _busy_target; "
        "result = trace(_busy_target, (), max_events=1); "
        "assert not result.complete; "
        "assert any(diagnostic.code == 'trace.limit' for diagnostic in result.diagnostics)"
    )


def test_same_line_lambda_constants_distinguish_trace_code_ids() -> None:
    """Constant fingerprints distinguish otherwise identical descendant code."""

    _run_child(
        "from dryml.code import trace; "
        "from tests.code.test_trace import _same_line_lambda_target; "
        "result = trace(_same_line_lambda_target, ()); "
        "calls = [dict(node.value) for node in result.graph.nodes_of_kind('trace_event') if dict(node.value)['event'] == 'call' and dict(node.value)['depth'] == 1]; "
        "assert len(calls) == 2 and len({event['code_id'] for event in calls}) == 2"
    )


def test_heterogeneous_frozenset_constants_trace_successfully() -> None:
    """Canonical constant ordering never compares heterogeneous scalars."""

    _run_child(
        "from dryml.code import trace; "
        "from tests.code.test_trace import _heterogeneous_frozenset_target; "
        "result = trace(_heterogeneous_frozenset_target, ()); "
        "assert result.complete and result.invocation.status == 'succeeded'"
    )


def test_same_line_lambda_names_distinguish_trace_code_ids() -> None:
    """Code identities include referenced-name operand tables."""

    _run_child(
        "from dryml.code import trace; "
        "from tests.code.test_trace import _same_line_name_target; "
        "result = trace(_same_line_name_target, ()); "
        "calls = [dict(node.value) for node in result.graph.nodes_of_kind('trace_event') if dict(node.value)['event'] == 'call' and dict(node.value)['depth'] == 1]; "
        "assert len(calls) == 2 and len({event['code_id'] for event in calls}) == 2"
    )


def test_displaced_trace_hook_fails_closed() -> None:
    """A target cannot disable capture and still receive complete evidence."""

    _run_child(
        "import sys; from dryml.code import trace; "
        "from tests.code.test_trace import _displacing_target; "
        "result = trace(_displacing_target, ()); "
        "assert not result.complete and result.invocation.status == 'failed'; "
        "assert result.invocation.diagnostic.code == 'trace.invocation'; "
        "assert sys.gettrace() is None"
    )


def test_trace_success_runs_in_fresh_child_without_pytest_trace_hook() -> None:
    """A clean interpreter verifies successful hook installation and restoration."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dryml.code import trace; "
            "from tests.code.test_trace import _target; "
            "result = trace(_target, ()); "
            "assert result.complete and result.invocation.status == 'succeeded' and result.graph.nodes_of_kind('trace_event'); "
            "assert sys.gettrace() is None",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""


def test_trace_graph_and_code_identity_are_stable_in_fresh_children() -> None:
    """Repeated traces preserve immutable event and code-ID ordering."""

    _run_child(
        "from dryml.code import trace; "
        "from tests.code.test_trace import _target; "
        "first = trace(_target, ()); second = trace(_target, ()); "
        "assert first.graph == second.graph; "
        "assert first.graph.digest == second.graph.digest; "
        "assert [dict(node.value)['code_id'] for node in first.graph.nodes_of_kind('trace_event')] == [dict(node.value)['code_id'] for node in second.graph.nodes_of_kind('trace_event')]"
    )


def test_trace_cleanup_failure_takes_precedence_in_fresh_child() -> None:
    """Failed restoration disables tracing best-effort and returns no result."""

    _run_child(
        "import importlib; "
        "from dryml.code import CodeAnalysisError, trace; "
        "from tests.code.test_trace import _target; "
        "module = importlib.import_module('dryml.code.trace'); original = module.sys.settrace; calls = []; "
        "exec(\"def broken(hook):\\n    calls.append(hook)\\n    if len(calls) == 2: raise RuntimeError('private')\\n    return original(hook)\"); "
        "module.sys.settrace = broken; "
        "exec(\"try:\\n    trace(_target, ())\\nexcept CodeAnalysisError as error:\\n    assert error.code == 'trace.cleanup' and error.__suppress_context__\\nelse:\\n    raise AssertionError('cleanup failure was swallowed')\"); "
        "assert len(calls) == 3; module.sys.settrace = original"
    )


def test_trace_cleanup_failure_takes_precedence_over_interruption() -> None:
    """Restoration failure masks and drops an in-flight interruption."""

    _run_child(
        "import importlib; "
        "from dryml.code import CodeAnalysisError, trace; "
        "from tests.code.test_trace import _interrupting_target; "
        "module = importlib.import_module('dryml.code.trace'); original = module.sys.settrace; calls = []; "
        "exec(\"def broken(hook):\\n    calls.append(hook)\\n    if len(calls) == 2: raise RuntimeError('private')\\n    return original(hook)\"); "
        "module.sys.settrace = broken; "
        "exec(\"try:\\n    trace(_interrupting_target, ())\\nexcept CodeAnalysisError as error:\\n    assert error.code == 'trace.cleanup' and error.__suppress_context__\\nelse:\\n    raise AssertionError('cleanup failure was swallowed')\"); "
        "assert len(calls) == 3; module.sys.settrace = original"
    )
