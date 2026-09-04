"""Bounded current-thread tracing for live synchronous Python callables.

The module has no registry or cross-thread coordination. It projects only
immutable event evidence and always attempts to restore the previous hook.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import types
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .analysis import (
    AnalysisResult,
    InvocationOutcome,
    _Snapshot,
    _dependency_closure,
    _execution_order,
    _failure_outcome,
    _outcome_for_value,
    _snapshot_calls,
    _validate_admission,
)
from .errors import CodeAnalysisError
from .facts import Diagnostic, SourceLocation
from .graph import ProgramEdge, ProgramGraph, ProgramNode, _encode, _node_id, _pack, _source_value, build_program_graph
from .kernels import AnalysisKernel, KernelCall, KernelContext, KernelOutcome, _records_for_outcomes
from .targets import CodeTargetInput, normalize_target


@dataclass(frozen=True, slots=True)
class _TraceEvent:
    """Private immutable projection of one accepted interpreter event."""

    event: str
    code_id: str
    qualname: str
    first_line: int
    location: SourceLocation
    sequence: int
    depth: int
    root: bool
    parent_sequence: int | None


def _trace_error(code: str, message: str) -> CodeAnalysisError:
    """Create one fixed redacted trace failure without runtime evidence."""

    return CodeAnalysisError(message, code=code)  # type: ignore[arg-type]


def _validate_max_events(max_events: int) -> None:
    """Validate the bounded event-memory policy before reading any request input."""

    if type(max_events) is not int or not 1 <= max_events <= 100_000:
        raise _trace_error("trace.limit", "trace event limit is invalid")


def _live_target(target: CodeTargetInput) -> types.FunctionType | types.MethodType:
    """Admit only a direct synchronous Python function or bound Python method."""

    if type(target) is types.FunctionType:
        function = target
    elif type(target) is types.MethodType and type(target.__func__) is types.FunctionType:
        function = target.__func__
    else:
        raise _trace_error("trace.unsupported", "trace target is unsupported")
    flags = function.__code__.co_flags
    if flags & (inspect.CO_COROUTINE | inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR):
        raise _trace_error("trace.unsupported", "trace target is unsupported")
    return target


def _constant_fingerprint(value: object) -> tuple[object, ...]:
    """Return a recursive code-constant fingerprint without retaining constants."""

    value_type = type(value)
    if value is None or value_type in (bool, int):
        return ("scalar", value)
    if value_type is float:
        return ("float", repr(value))
    if value_type is complex:
        return ("complex", repr(value.real), repr(value.imag))
    if value_type is str:
        return ("str", hashlib.sha256(value.encode("utf-8")).hexdigest())
    if value_type is bytes:
        return ("bytes", hashlib.sha256(value).hexdigest())
    if value_type is tuple:
        return ("tuple", tuple(("item", index, _constant_fingerprint(item)) for index, item in enumerate(value)))
    if value_type is frozenset:
        fingerprints = tuple(_constant_fingerprint(item) for item in value)
        return ("frozenset", tuple(
            ("item", index, item)
            for index, item in enumerate(sorted(fingerprints, key=_encode))
        ))
    if value_type is types.CodeType:
        return ("code", _code_fingerprint(value))
    return ("other", value_type.__module__, value_type.__qualname__)


def _code_fingerprint(code: types.CodeType) -> tuple[object, ...]:
    """Return stable nested metadata for a code object without preserving it."""

    filename = SourceLocation(code.co_filename, code.co_firstlineno, None).filename
    code_qualname = getattr(code, "co_qualname", None)
    qualname = code_qualname if type(code_qualname) is str else code.co_name
    return (
        "metadata",
        qualname,
        filename,
        code.co_firstlineno,
        code.co_argcount,
        getattr(code, "co_posonlyargcount", 0),
        code.co_kwonlyargcount,
        code.co_flags,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        hashlib.sha256(code.co_code).hexdigest(),
        tuple(("constant", index, _constant_fingerprint(item)) for index, item in enumerate(code.co_consts)),
    )


def _code_id(
    frame: types.FrameType,
    cache: dict[tuple[types.CodeType, str | None], tuple[str, str, int]],
) -> tuple[str, str, int]:
    """Hash current code metadata into a domain-separated opaque code identity."""

    code = frame.f_code
    module = frame.f_globals.get("__name__")
    module_name = module if type(module) is str else None
    cache_key = (code, module_name)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    code_qualname = getattr(code, "co_qualname", None)
    qualname = code_qualname if type(code_qualname) is str else code.co_name
    filename = SourceLocation(code.co_filename, code.co_firstlineno, None).filename
    payload = (
        ("bytecode", hashlib.sha256(code.co_code).hexdigest()),
        ("constants", _code_fingerprint(code)),
        ("filename", filename),
        ("first_line", code.co_firstlineno),
        ("module", module_name),
        ("qualname", qualname),
    )
    result = (
        hashlib.sha256(_pack(b"dryml.code.trace-code.v1", _encode(payload))).hexdigest(),
        qualname,
        code.co_firstlineno,
    )
    cache[cache_key] = result
    return result


def _event_value(event: _TraceEvent) -> tuple[tuple[str, object], ...]:
    """Project one transient event into the closed graph payload grammar."""

    return (
        ("code_id", event.code_id),
        ("current_thread_only", True),
        ("depth", event.depth),
        ("event", event.event),
        ("location", _source_value(event.location)),
        ("python_only", True),
        ("root", event.root),
        ("sequence", event.sequence),
    )


def _derived_graph(base: ProgramGraph, events: tuple[_TraceEvent, ...]) -> ProgramGraph:
    """Add deterministic structural trace evidence to one immutable base graph."""

    nodes = list(base.nodes)
    event_ids: list[str] = []
    syntax_index: dict[tuple[str | None, int, str], list[ProgramNode]] = {}
    for node in base.nodes_of_kind("syntax"):
        if node.source is None or node.source.line is None:
            continue
        payload = dict(node.value)
        if payload["type"] not in ("AsyncFunctionDef", "FunctionDef"):
            continue
        names = [
            field[2]
            for field in payload["fields"]
            if field[1] == "name" and type(field[2]) is str
        ]
        if len(names) == 1:
            syntax_index.setdefault(
                (node.source.filename, node.source.line, names[0]), []
            ).append(node)
    for event in events:
        value = _event_value(event)
        identifier = _node_id("trace_event", value, event.location, 0)
        event_ids.append(identifier)
        nodes.append(ProgramNode(identifier, "trace_event", value, event.location))
    edges = list(base.edges)
    for index in range(1, len(event_ids)):
        edges.append(ProgramEdge(event_ids[index - 1], event_ids[index], "trace_sequence"))
    for index, event in enumerate(events):
        if event.parent_sequence is not None:
            edges.append(ProgramEdge(event_ids[event.parent_sequence], event_ids[index], "frame_descent"))
        expected_name = event.qualname.rsplit(".", 1)[-1]
        if expected_name != "<lambda>":
            candidates = syntax_index.get(
                (event.location.filename, event.first_line, expected_name), []
            )
            if len(candidates) == 1:
                edges.append(ProgramEdge(event_ids[index], candidates[0].id, "observed_code"))
    return ProgramGraph(base.target, tuple(nodes), tuple(edges), base.diagnostics)


def _binding(snapshot: _Snapshot, base_digest: str, trace_digest: str) -> str:
    """Return the graph identity selected by one snapshotted execution mode."""

    return base_digest if snapshot.mode == "static" else trace_digest


def _context_for_trace(
    snapshot: _Snapshot,
    snapshots: tuple[_Snapshot, ...],
    artifacts: dict[tuple[type[AnalysisKernel[Any, Any]], str], KernelOutcome[Any]],
    trace_graph: ProgramGraph,
    base_digest: str,
    trace_digest: str,
    declarations: dict[type[AnalysisKernel[Any, Any]], tuple[str, type[Any]]],
) -> KernelContext:
    """Create a trace context whose static prerequisites retain base provenance."""

    by_type = {item.kernel_type: item for item in snapshots}
    closure = set(_dependency_closure(snapshot, snapshots))
    outcomes = tuple(
        artifacts[(item.kernel_type, _binding(item, base_digest, trace_digest))]
        for item in snapshots
        if item.kernel_type in closure
    )
    dependencies = {
        required: artifacts[(required, _binding(by_type[required], base_digest, trace_digest))]
        for required in snapshot.requires
    }
    return KernelContext(trace_graph, dependencies, _records_for_outcomes(outcomes, declarations))


def _run_static(
    snapshots: tuple[_Snapshot, ...],
    base_graph: ProgramGraph,
    base_digest: str,
    artifacts: dict[tuple[type[AnalysisKernel[Any, Any]], str], KernelOutcome[Any]],
    declarations: dict[type[AnalysisKernel[Any, Any]], tuple[str, type[Any]]],
) -> None:
    """Execute static kernels once in ordinary stable order without trace fusion."""

    from .analysis import _run_unfused

    static = tuple(snapshot for snapshot in snapshots if snapshot.mode == "static")
    ordered = _execution_order(static)
    pending = list(ordered)
    while pending:
        snapshot = next(item for item in pending if all((required, base_digest) in artifacts for required in item.requires))
        unavailable = tuple(required for required in snapshot.requires if artifacts[(required, base_digest)].status != "succeeded")
        if unavailable:
            artifacts[(snapshot.kernel_type, base_digest)] = KernelOutcome(snapshot.kernel_type, base_digest, "skipped", None, skipped_for=unavailable)
        else:
            artifacts[(snapshot.kernel_type, base_digest)] = _run_unfused(
                snapshot, base_graph, base_digest, static, ordered, artifacts, declarations,
            )
        pending.remove(snapshot)


def _trace_failure(snapshot: _Snapshot, digest: str, code: str) -> KernelOutcome[Any]:
    """Return a redacted failed trace artifact with no output or facts."""

    message = "trace event limit exceeded" if code == "trace.limit" else "target invocation failed"
    return KernelOutcome(snapshot.kernel_type, digest, "failed", None, (Diagnostic(code, message, kernel=snapshot.kernel_type),))


def _run_trace_kernels(
    snapshots: tuple[_Snapshot, ...],
    trace_graph: ProgramGraph,
    base_digest: str,
    trace_digest: str,
    artifacts: dict[tuple[type[AnalysisKernel[Any, Any]], str], KernelOutcome[Any]],
    declarations: dict[type[AnalysisKernel[Any, Any]], tuple[str, type[Any]]],
    failure: str | None,
) -> None:
    """Run trace-mode consumers or fail them closed after trace acquisition failure."""

    trace_snapshots = tuple(snapshot for snapshot in snapshots if snapshot.mode == "trace")
    digest = trace_digest
    pending = list(trace_snapshots)
    while pending:
        snapshot = next(
            item for item in pending
            if all(
                requirement not in {candidate.kernel_type for candidate in pending}
                for requirement in item.requires
            )
        )
        required = tuple(
            requirement for requirement in snapshot.requires
            if artifacts[(requirement, _binding(next(item for item in snapshots if item.kernel_type is requirement), base_digest, digest))].status != "succeeded"
        )
        if required and failure != "trace.limit":
            artifacts[(snapshot.kernel_type, digest)] = KernelOutcome(snapshot.kernel_type, digest, "skipped", None, skipped_for=required)
        elif failure is not None:
            artifacts[(snapshot.kernel_type, digest)] = _trace_failure(snapshot, digest, failure)
        else:
            try:
                value = snapshot.kernel.run(
                    trace_graph,
                    snapshot.input,
                    _context_for_trace(
                        snapshot,
                        snapshots,
                        artifacts,
                        trace_graph,
                        base_digest,
                        trace_digest,
                        declarations,
                    ),
                )
            except Exception:
                artifacts[(snapshot.kernel_type, digest)] = _failure_outcome(snapshot, digest)
            else:
                artifacts[(snapshot.kernel_type, digest)] = _outcome_for_value(snapshot, digest, value)
        pending.remove(snapshot)


def trace(
    target: CodeTargetInput,
    calls: Iterable[KernelCall[Any, Any]],
    *,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    max_events: int = 100_000,
) -> AnalysisResult:
    """Invoke a live Python callable once while collecting bounded immutable evidence.

    Args:
        target: A direct synchronous Python function or bound Python method.
        calls: Per-request static and trace kernel declarations.
        args: Trusted opaque positional values passed directly to ``target``.
        kwargs: Trusted opaque keyword values copied for the one target call.
        max_events: Exact non-boolean integer event bound from 1 through 100,000.

    Returns:
        Static artifacts on the base graph and trace artifacts on a distinct
        derived graph. Ordinary target errors become a redacted failed
        invocation outcome.

    Raises:
        CodeAnalysisError: For invalid bounds, unsupported targets, active hooks,
            or hook-restoration failure. Cleanup failure has precedence.
        BaseException: Interruption-style target failures after hook cleanup.

    Side Effects:
        Runs static kernels, installs a temporary current-thread hook, and calls
        the target exactly once. It has no process, transport, registry, lock,
        or cross-thread coordination behavior.
    """

    _validate_max_events(max_events)
    snapshots = _snapshot_calls(calls)
    live = _live_target(target)
    normalized = normalize_target(live)
    _validate_admission(snapshots, normalized.info.kind)
    base_graph = build_program_graph(normalized)
    if sys.gettrace() is not None:
        raise _trace_error("trace.hook_active", "current thread trace hook is active")
    call_args = tuple(args)
    call_kwargs = dict(kwargs) if kwargs is not None else {}
    declarations = {snapshot.kernel_type: (snapshot.mode, snapshot.output_type) for snapshot in snapshots}
    artifacts: dict[tuple[type[AnalysisKernel[Any, Any]], str], KernelOutcome[Any]] = {}
    base_digest = base_graph.digest
    _run_static(snapshots, base_graph, base_digest, artifacts, declarations)

    events: list[_TraceEvent] = []
    code_ids: dict[tuple[types.CodeType, str | None], tuple[str, str, int]] = {}
    active: dict[int, tuple[int, int | None]] = {}
    root_identifier: int | None = None
    overflow = False
    callback_failed = False

    def hook(frame: types.FrameType, event: str, argument: object) -> Any:
        """Project root-descendant interpreter events without retaining frames."""

        nonlocal root_identifier, overflow, callback_failed
        if overflow or callback_failed:
            return None
        try:
            identifier = id(frame)
            if event == "call":
                if root_identifier is None:
                    root_identifier = identifier
                    active[identifier] = (0, None)
                else:
                    parent = active.get(id(frame.f_back))
                    if parent is None:
                        return None
                    active[identifier] = (parent[0] + 1, parent[1])
                frame.f_trace_opcodes = False
            current = active.get(identifier)
            if current is None or event not in ("call", "line", "return", "exception"):
                return hook
            if len(events) >= max_events:
                overflow = True
                active.clear()
                return None
            code_id, qualname, first_line = _code_id(frame, code_ids)
            location = SourceLocation(frame.f_code.co_filename, frame.f_lineno, None)
            depth, latest_sequence = current
            sequence = len(events)
            parent_sequence = latest_sequence if event == "call" and identifier != root_identifier else None
            events.append(_TraceEvent(event, code_id, qualname, first_line, location, sequence, depth, identifier == root_identifier, parent_sequence))
            active[identifier] = (depth, sequence)
            if event == "return":
                active.pop(identifier, None)
            return hook
        except Exception:
            callback_failed = True
            active.clear()
            return None

    prior = sys.gettrace()
    invocation_failed = False
    interrupted: BaseException | None = None
    cleanup_failed = False
    try:
        sys.settrace(hook)
        try:
            live(*call_args, **call_kwargs)
        except Exception:
            invocation_failed = True
        except BaseException as error:
            interrupted = error
    finally:
        if sys.gettrace() is not hook:
            callback_failed = True
        try:
            sys.settrace(prior)
        except BaseException:
            cleanup_failed = True
            try:
                sys.settrace(None)
            except BaseException:
                pass
        active.clear()
        code_ids.clear()
    if cleanup_failed:
        interrupted = None
        raise _trace_error("trace.cleanup", "trace hook cleanup failed") from None
    if interrupted is not None:
        raise interrupted.with_traceback(interrupted.__traceback__)

    trace_graph = _derived_graph(base_graph, tuple(events))
    events.clear()
    trace_digest = trace_graph.digest
    trace_failure = "trace.limit" if overflow else "trace.invocation" if invocation_failed or callback_failed else None
    _run_trace_kernels(
        snapshots,
        trace_graph,
        base_digest,
        trace_digest,
        artifacts,
        declarations,
        trace_failure,
    )
    outcomes = tuple(artifacts[(snapshot.kernel_type, _binding(snapshot, base_digest, trace_digest))] for snapshot in snapshots)
    facts = _records_for_outcomes(outcomes, declarations)
    invocation = InvocationOutcome(
        "failed" if invocation_failed or callback_failed else "succeeded",
        Diagnostic("trace.invocation", "target invocation failed") if invocation_failed or callback_failed else None,
    )
    diagnostics = base_graph.diagnostics
    if overflow:
        diagnostics += (Diagnostic("trace.limit", "trace event limit exceeded"),)
    diagnostics += tuple(diagnostic for outcome in outcomes for diagnostic in outcome.diagnostics)
    if invocation.diagnostic is not None:
        diagnostics += (invocation.diagnostic,)
    return AnalysisResult(normalized.info, base_graph, trace_graph, outcomes, facts, diagnostics, invocation)


__all__ = ["trace"]
