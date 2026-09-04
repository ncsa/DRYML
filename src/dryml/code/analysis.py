"""Static per-request kernel DAG validation, scheduling, and result assembly."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Generic, Iterable, Literal, TypeVar

from .errors import InvalidKernelError, KernelDependencyError, MissingOutputError
from .facts import CodeFact, CodeFacts, Diagnostic, FactRecord
from .graph import ProgramGraph, build_program_graph
from .kernels import AnalysisKernel, KernelCall, KernelContext, KernelMode, KernelOutcome, OutputU, _has_instance_state, _inherits_traversal_template, _records_for_outcomes
from .targets import CodeTargetInput, TargetInfo, TargetKind, normalize_target


@dataclass(frozen=True, slots=True)
class InvocationOutcome:
    """Reserved structured status for invocation-bearing analysis paths.

    Args:
        status: Whether a future trace invocation succeeded or failed.
        diagnostic: Optional redacted invocation diagnostic.

    Raises:
        ValueError: If the framework-owned fields are invalid.

    Side Effects:
        None. U3 static analysis always returns ``None`` for this field.
    """

    status: Literal["succeeded", "failed"]
    diagnostic: Diagnostic | None = None

    def __post_init__(self) -> None:
        """Validate the immutable invocation status carrier."""

        if self.status not in ("succeeded", "failed") or (self.diagnostic is not None and type(self.diagnostic) is not Diagnostic):
            raise ValueError("invocation outcome is invalid")


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Ephemeral structured output of one graph-bound static analysis request.

    Args:
        target: Immutable metadata-only target provenance.
        base_graph: Immutable graph from static target normalization.
        graph: Graph used by returned outcomes; identical to ``base_graph`` here.
        outcomes: Submitted kernel outcomes in original submission order.
        facts: Exact successful fact-wrapper records in producer submission order.
        diagnostics: Graph and outcome diagnostics in deterministic order.
        invocation: Optional invocation status for later trace analysis.

    Raises:
        ValueError: If framework-created result containers are invalid.

    Side Effects:
        None. Caller kernels, inputs, and outputs remain opaque references.
    """

    target: TargetInfo
    base_graph: ProgramGraph
    graph: ProgramGraph
    outcomes: tuple[KernelOutcome[Any], ...]
    facts: tuple[FactRecord, ...]
    diagnostics: tuple[Diagnostic, ...]
    invocation: InvocationOutcome | None = None

    def __post_init__(self) -> None:
        """Validate framework-owned immutable result fields and static binding."""

        if type(self.target) is not TargetInfo or type(self.base_graph) is not ProgramGraph or type(self.graph) is not ProgramGraph:
            raise ValueError("analysis result is invalid")
        if self.base_graph is not self.graph or self.graph.target != self.target:
            raise ValueError("static analysis result graph binding is invalid")
        if type(self.outcomes) is not tuple or type(self.facts) is not tuple or type(self.diagnostics) is not tuple:
            raise ValueError("analysis result containers are invalid")
        if any(type(outcome) is not KernelOutcome for outcome in self.outcomes):
            raise ValueError("analysis result outcomes are invalid")
        if any(type(record) is not FactRecord for record in self.facts) or any(type(diagnostic) is not Diagnostic for diagnostic in self.diagnostics):
            raise ValueError("analysis result evidence is invalid")
        if self.invocation is not None and type(self.invocation) is not InvocationOutcome:
            raise ValueError("analysis result invocation is invalid")

    @property
    def complete(self) -> bool:
        """Return whether every submitted artifact and invocation succeeded.

        Returns:
            ``True`` only if all outcomes and any invocation succeeded and no
            returned diagnostic has error severity.

        Side Effects:
            None.
        """

        return (
            all(outcome.status == "succeeded" for outcome in self.outcomes)
            and (self.invocation is None or self.invocation.status == "succeeded")
            and not any(diagnostic.severity == "error" for diagnostic in self.diagnostics)
        )

    def output(self, kernel_type: type[AnalysisKernel[Any, OutputU]]) -> OutputU | None:
        """Return one successful output or ``None`` for absence and valid None.

        Args:
            kernel_type: Exact submitted concrete producer class.

        Returns:
            The successful output, which may be ``None``, or ``None`` when no
            successful matching producer exists.

        Side Effects:
            None. Use :meth:`require` to distinguish absence from valid ``None``.
        """

        for outcome in self.outcomes:
            if outcome.kernel is kernel_type and outcome.status == "succeeded":
                return outcome.value  # type: ignore[return-value]
        return None

    def require(self, kernel_type: type[AnalysisKernel[Any, OutputU]]) -> OutputU:
        """Return one successful submitted output, including valid ``None``.

        Args:
            kernel_type: Exact submitted concrete producer class.

        Returns:
            The successful output value.

        Raises:
            MissingOutputError: If that producer is absent, failed, or skipped.

        Side Effects:
            None.
        """

        for outcome in self.outcomes:
            if outcome.kernel is kernel_type and outcome.status == "succeeded":
                return outcome.value  # type: ignore[return-value]
        raise MissingOutputError()


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """Private frozen declaration and input projection for one submitted call."""

    index: int
    kernel: AnalysisKernel[Any, Any]
    kernel_type: type[AnalysisKernel[Any, Any]]
    input: Any
    input_type: type[Any]
    output_type: type[Any]
    target_kinds: frozenset[TargetKind] | None
    requires: tuple[type[AnalysisKernel[Any, Any]], ...]
    mode: KernelMode
    fusion_safe: bool


def _invalid(message: str = "invalid kernel declaration") -> InvalidKernelError:
    """Return a fixed redacted declaration failure without caller values."""

    return InvalidKernelError(message)


def _snapshot_calls(calls: Iterable[KernelCall[Any, Any]]) -> tuple[_Snapshot, ...]:
    """Materialize calls once and validate target-independent declarations."""

    try:
        materialized = tuple(calls)
    except Exception:
        raise _invalid("kernel calls are invalid") from None
    snapshots: list[_Snapshot] = []
    seen: set[type[AnalysisKernel[Any, Any]]] = set()
    valid_kinds = frozenset({"function", "bound_method", "callable_instance", "descriptor", "class", "import", "source"})
    for index, call in enumerate(materialized):
        if type(call) is not KernelCall or not isinstance(call.kernel, AnalysisKernel):
            raise _invalid()
        kernel_type = type(call.kernel)
        if inspect.isabstract(kernel_type) or kernel_type in seen:
            raise _invalid()
        seen.add(kernel_type)
        try:
            input_type = kernel_type.input_type
            output_type = kernel_type.output_type
            target_kinds = kernel_type.target_kinds
            requires = kernel_type.requires
            mode = kernel_type.mode
            fusion_safe = kernel_type.fusion_safe
        except Exception:
            raise _invalid() from None
        if not isinstance(input_type, type) or not isinstance(output_type, type):
            raise _invalid()
        if target_kinds is not None:
            if type(target_kinds) is not frozenset or not target_kinds or not target_kinds <= valid_kinds:
                raise _invalid()
        if type(requires) is not tuple:
            raise _invalid()
        try:
            if len(set(requires)) != len(requires):
                raise _invalid()
        except TypeError:
            raise _invalid() from None
        for required in requires:
            if not isinstance(required, type) or not issubclass(required, AnalysisKernel) or inspect.isabstract(required):
                raise _invalid()
        if type(mode) is not str or mode not in ("static", "trace") or type(fusion_safe) is not bool:
            raise _invalid()
        snapshots.append(_Snapshot(index, call.kernel, kernel_type, call.input, input_type, output_type, target_kinds, requires, mode, fusion_safe))
    by_type = {snapshot.kernel_type: snapshot for snapshot in snapshots}
    for snapshot in snapshots:
        for required in snapshot.requires:
            producer = by_type.get(required)
            if producer is None:
                raise KernelDependencyError("required kernel producer is missing")
            if snapshot.mode == "static" and producer.mode == "trace":
                raise KernelDependencyError("static kernel cannot require trace output")
    remaining = {snapshot.kernel_type: set(snapshot.requires) for snapshot in snapshots}
    resolved: set[type[AnalysisKernel[Any, Any]]] = set()
    while remaining:
        ready = [snapshot.kernel_type for snapshot in snapshots if snapshot.kernel_type in remaining and remaining[snapshot.kernel_type] <= resolved]
        if not ready:
            raise KernelDependencyError("kernel dependency cycle")
        resolved.add(ready[0])
        del remaining[ready[0]]
    return tuple(snapshots)


def _validate_admission(snapshots: tuple[_Snapshot, ...], target_kind: TargetKind) -> None:
    """Validate target-kind and nominal input admission after normalization."""

    for snapshot in snapshots:
        if snapshot.target_kinds is not None and target_kind not in snapshot.target_kinds:
            raise _invalid("kernel does not accept target kind")
        if not isinstance(snapshot.input, snapshot.input_type):
            raise _invalid("kernel input has wrong type")


def _execution_order(snapshots: tuple[_Snapshot, ...]) -> tuple[_Snapshot, ...]:
    """Return a stable topological ordering with submission-index ready ties."""

    pending = {snapshot.kernel_type: snapshot for snapshot in snapshots}
    complete: set[type[AnalysisKernel[Any, Any]]] = set()
    ordered: list[_Snapshot] = []
    while pending:
        ready = [snapshot for snapshot in snapshots if snapshot.kernel_type in pending and set(snapshot.requires) <= complete]
        snapshot = ready[0]
        ordered.append(snapshot)
        complete.add(snapshot.kernel_type)
        del pending[snapshot.kernel_type]
    return tuple(ordered)


def _dependency_closure(
    snapshot: _Snapshot,
    snapshots: tuple[_Snapshot, ...],
) -> tuple[type[AnalysisKernel[Any, Any]], ...]:
    """Return the transitive declared producer closure in execution order."""

    by_type = {item.kernel_type: item for item in snapshots}
    closure: set[type[AnalysisKernel[Any, Any]]] = set()

    def collect(kernel_type: type[AnalysisKernel[Any, Any]]) -> None:
        """Collect declared producers without reading consumer-owned outputs."""

        for required in by_type[kernel_type].requires:
            if required not in closure:
                closure.add(required)
                collect(required)

    collect(snapshot.kernel_type)
    return tuple(item.kernel_type for item in snapshots if item.kernel_type in closure)


def _context_for(
    snapshot: _Snapshot,
    graph: ProgramGraph,
    graph_digest: str,
    snapshots: tuple[_Snapshot, ...],
    ordered: tuple[_Snapshot, ...],
    artifacts: dict[tuple[type[AnalysisKernel[Any, Any]], str], KernelOutcome[Any]],
    declarations: dict[type[AnalysisKernel[Any, Any]], tuple[KernelMode, type[Any]]],
) -> KernelContext:
    """Create one private dependency snapshot for a scheduled kernel."""

    closure = set(_dependency_closure(snapshot, snapshots))
    dependency_outcomes = tuple(
        artifacts[(item.kernel_type, graph_digest)]
        for item in ordered
        if item.kernel_type in closure
    )
    return KernelContext(
        graph,
        {required: artifacts[(required, graph_digest)] for required in snapshot.requires},
        _records_for_outcomes(dependency_outcomes, declarations),
    )


def _failure_outcome(snapshot: _Snapshot, graph_digest: str) -> KernelOutcome[Any]:
    """Return the redacted execution failure attributed to one kernel only."""

    return KernelOutcome(
        snapshot.kernel_type,
        graph_digest,
        "failed",
        None,
        (Diagnostic("kernel.execution", "kernel execution failed", kernel=snapshot.kernel_type),),
    )


def _outcome_for_value(snapshot: _Snapshot, graph_digest: str, value: Any) -> KernelOutcome[Any]:
    """Validate one callback result using ordinary scheduler output semantics."""

    if (value is None and snapshot.output_type is not type(None)) or not isinstance(value, snapshot.output_type):
        return KernelOutcome(
            snapshot.kernel_type,
            graph_digest,
            "failed",
            None,
            (Diagnostic("kernel.output_type", "kernel output has wrong type", kernel=snapshot.kernel_type),),
        )
    return KernelOutcome(snapshot.kernel_type, graph_digest, "succeeded", value)


def _run_unfused(
    snapshot: _Snapshot,
    graph: ProgramGraph,
    graph_digest: str,
    snapshots: tuple[_Snapshot, ...],
    ordered: tuple[_Snapshot, ...],
    artifacts: dict[tuple[type[AnalysisKernel[Any, Any]], str], KernelOutcome[Any]],
    declarations: dict[type[AnalysisKernel[Any, Any]], tuple[KernelMode, type[Any]]],
) -> KernelOutcome[Any]:
    """Execute one kernel through the established independent scheduler path."""

    try:
        value = snapshot.kernel.run(
            graph,
            snapshot.input,
            _context_for(snapshot, graph, graph_digest, snapshots, ordered, artifacts, declarations),
        )
    except Exception:
        return _failure_outcome(snapshot, graph_digest)
    return _outcome_for_value(snapshot, graph_digest, value)


def _has_dependency_path(
    source: _Snapshot,
    target: _Snapshot,
    by_type: dict[type[AnalysisKernel[Any, Any]], _Snapshot],
) -> bool:
    """Return whether one declared producer path connects two candidate peers."""

    pending = list(source.requires)
    seen: set[type[AnalysisKernel[Any, Any]]] = set()
    while pending:
        kernel_type = pending.pop()
        if kernel_type is target.kernel_type:
            return True
        if kernel_type not in seen:
            seen.add(kernel_type)
            pending.extend(by_type[kernel_type].requires)
    return False


def _has_read_only_dependencies(
    snapshot: _Snapshot,
    snapshots: tuple[_Snapshot, ...],
    graph_digest: str,
    artifacts: dict[tuple[type[AnalysisKernel[Any, Any]], str], KernelOutcome[Any]],
) -> bool:
    """Require successful framework-immutable artifacts throughout dependencies.

    Consumer outputs are opaque and only read-only by cooperative contract. The
    fusion path therefore accepts dependent traversals only when their complete
    closure consists of successful immutable framework fact carriers or None.
    """

    by_type = {item.kernel_type: item for item in snapshots}
    for kernel_type in _dependency_closure(snapshot, snapshots):
        outcome = artifacts.get((kernel_type, graph_digest))
        if outcome is None or outcome.status != "succeeded":
            return False
        if by_type[kernel_type].output_type not in (CodeFact, CodeFacts, type(None)):
            return False
    return True


def _fused_batch(
    ready: tuple[_Snapshot, ...],
    graph: ProgramGraph,
    graph_digest: str,
    snapshots: tuple[_Snapshot, ...],
    artifacts: dict[tuple[type[AnalysisKernel[Any, Any]], str], KernelOutcome[Any]],
) -> tuple[_Snapshot, ...]:
    """Return the first contiguous ready batch whose fusion proof is explicit."""

    by_type = {item.kernel_type: item for item in snapshots}
    batch: list[_Snapshot] = []
    for snapshot in ready:
        if (
            snapshot.mode != "static"
            or not snapshot.fusion_safe
            or not _inherits_traversal_template(snapshot.kernel_type)
            or _has_instance_state(snapshot.kernel)
            or not _has_read_only_dependencies(snapshot, snapshots, graph_digest, artifacts)
            or any(
                _has_dependency_path(snapshot, peer, by_type)
                or _has_dependency_path(peer, snapshot, by_type)
                for peer in batch
            )
        ):
            break
        batch.append(snapshot)
    return tuple(batch) if len(batch) > 1 else ()


def _run_fused(
    batch: tuple[_Snapshot, ...],
    graph: ProgramGraph,
    graph_digest: str,
    snapshots: tuple[_Snapshot, ...],
    ordered: tuple[_Snapshot, ...],
    artifacts: dict[tuple[type[AnalysisKernel[Any, Any]], str], KernelOutcome[Any]],
    declarations: dict[type[AnalysisKernel[Any, Any]], tuple[KernelMode, type[Any]]],
) -> dict[type[AnalysisKernel[Any, Any]], KernelOutcome[Any]]:
    """Run compatible traversal callbacks once per node with private state.

    Outcomes remain buffered until every active peer has finished, so a callback
    failure is attributed only to its own kernel and cannot affect another
    peer's private context, state, callback sequence, or eventual artifact.
    """

    contexts = {
        snapshot.kernel_type: _context_for(snapshot, graph, graph_digest, snapshots, ordered, artifacts, declarations)
        for snapshot in batch
    }
    states: dict[type[AnalysisKernel[Any, Any]], Any] = {}
    outcomes: dict[type[AnalysisKernel[Any, Any]], KernelOutcome[Any]] = {}
    active: list[_Snapshot] = []
    for snapshot in batch:
        try:
            states[snapshot.kernel_type] = snapshot.kernel.begin(snapshot.input, contexts[snapshot.kernel_type])  # type: ignore[attr-defined]
        except Exception:
            outcomes[snapshot.kernel_type] = _failure_outcome(snapshot, graph_digest)
        else:
            active.append(snapshot)
    for node in graph.nodes:
        for snapshot in tuple(active):
            try:
                states[snapshot.kernel_type] = snapshot.kernel.visit(  # type: ignore[attr-defined]
                    node,
                    states[snapshot.kernel_type],
                    contexts[snapshot.kernel_type],
                )
            except Exception:
                outcomes[snapshot.kernel_type] = _failure_outcome(snapshot, graph_digest)
                active.remove(snapshot)
    for snapshot in active:
        try:
            value = snapshot.kernel.finish(states[snapshot.kernel_type], contexts[snapshot.kernel_type])  # type: ignore[attr-defined]
        except Exception:
            outcomes[snapshot.kernel_type] = _failure_outcome(snapshot, graph_digest)
        else:
            outcomes[snapshot.kernel_type] = _outcome_for_value(snapshot, graph_digest, value)
    return outcomes


def _analyze(
    target: CodeTargetInput,
    calls: Iterable[KernelCall[Any, Any]],
    *,
    fuse_traversals: bool,
) -> AnalysisResult:
    """Run static analysis with an internal switch for differential fusion tests."""

    snapshots = _snapshot_calls(calls)
    if any(snapshot.mode == "trace" for snapshot in snapshots):
        raise _invalid("static analysis does not accept trace kernels")
    normalized = normalize_target(target)
    _validate_admission(snapshots, normalized.info.kind)
    graph = build_program_graph(normalized)
    ordered = _execution_order(snapshots)
    graph_digest = graph.digest
    declarations = {
        snapshot.kernel_type: (snapshot.mode, snapshot.output_type)
        for snapshot in snapshots
    }
    artifacts: dict[tuple[type[AnalysisKernel[Any, Any]], str], KernelOutcome[Any]] = {}
    pending = list(ordered)
    while pending:
        ready = tuple(
            snapshot
            for snapshot in pending
            if all((required, graph_digest) in artifacts for required in snapshot.requires)
        )
        snapshot = ready[0]
        unavailable = tuple(
            required
            for required in snapshot.requires
            if artifacts[(required, graph_digest)].status != "succeeded"
        )
        if unavailable:
            artifacts[(snapshot.kernel_type, graph_digest)] = KernelOutcome(
                snapshot.kernel_type, graph_digest, "skipped", None, skipped_for=unavailable,
            )
            pending.remove(snapshot)
            continue
        batch = _fused_batch(ready, graph, graph_digest, snapshots, artifacts) if fuse_traversals else ()
        if batch:
            for kernel_type, outcome in _run_fused(
                batch, graph, graph_digest, snapshots, ordered, artifacts, declarations,
            ).items():
                artifacts[(kernel_type, graph_digest)] = outcome
            for item in batch:
                pending.remove(item)
            continue
        artifacts[(snapshot.kernel_type, graph_digest)] = _run_unfused(
            snapshot, graph, graph_digest, snapshots, ordered, artifacts, declarations,
        )
        pending.remove(snapshot)
    submission_outcomes = tuple(artifacts[(snapshot.kernel_type, graph_digest)] for snapshot in snapshots)
    facts = _records_for_outcomes(submission_outcomes, declarations)
    diagnostics = graph.diagnostics + tuple(diagnostic for outcome in submission_outcomes for diagnostic in outcome.diagnostics)
    return AnalysisResult(graph.target, graph, graph, submission_outcomes, facts, diagnostics)


def analyze(target: CodeTargetInput, calls: Iterable[KernelCall[Any, Any]]) -> AnalysisResult:
    """Run static consumer kernels over one deterministic immutable graph.

    Args:
        target: Supported static code target or target wrapper.
        calls: One-shot or reusable iterable of per-request kernel calls.

    Returns:
        A structured static result with submission-ordered outcomes and exact
        successful fact records.

    Raises:
        InvalidKernelError: If declarations, static-mode admission, target kind,
            or caller input are invalid before kernel execution.
        KernelDependencyError: If producer dependencies are missing, cyclic, or
            require an illegal trace-to-static edge.
        CodeAnalysisError: If target normalization or graph construction fails.

    Side Effects:
        May read source or explicitly import an ``ImportTarget`` module. It does
        not invoke target bodies, trace, launch workers, or use registries.
        Compatible static traversals may share one canonical node walk; consumer
        kernel side effects remain consumer-owned.
    """

    return _analyze(target, calls, fuse_traversals=True)


__all__ = ["AnalysisResult", "InvocationOutcome", "analyze"]
