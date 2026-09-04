"""Kernel declarations, artifacts, and dependency-scoped execution context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from .errors import MissingOutputError
from .facts import CodeFact, CodeFacts, Diagnostic, FactRecord
from .graph import ProgramGraph, ProgramNode
from .targets import TargetKind


InputT = TypeVar("InputT")
StateT = TypeVar("StateT")
OutputT = TypeVar("OutputT")
OutputU = TypeVar("OutputU")

KernelMode: TypeAlias = Literal["static", "trace"]


class AnalysisKernel(ABC, Generic[InputT, OutputT]):
    """Consumer-defined analysis that produces one graph-bound opaque artifact.

    Subclasses declare nominal runtime input and output classes, optional target
    admission, direct producer dependencies, and an execution mode. The
    framework snapshots these declarations per request; callers own instances,
    inputs, and output values and must not mutate artifacts visible to siblings.
    """

    input_type: type[InputT]
    output_type: type[OutputT]
    target_kinds: frozenset[TargetKind] | None = None
    requires: tuple[type["AnalysisKernel[Any, Any]"], ...] = ()
    mode: KernelMode = "static"
    fusion_safe: bool = False

    @abstractmethod
    def run(self, graph: ProgramGraph, value: InputT, context: "KernelContext") -> OutputT:
        """Produce one artifact for ``graph``.

        Args:
            graph: Immutable graph snapshot bound to this kernel execution.
            value: Caller input validated against :attr:`input_type`.
            context: Read-only access to declared successful dependencies.

        Returns:
            A value satisfying :attr:`output_type`; ``None`` is valid only when
            that declaration is ``type(None)``.

        Raises:
            Exception: Consumer exceptions become a redacted structured failed
            outcome; interruption-style ``BaseException`` values propagate.

        Side Effects:
            Consumer-defined. Conforming kernels do not mutate graph data,
            dependency artifacts, sibling-visible inputs, or shared state.
        """


class TraversalKernel(AnalysisKernel[InputT, OutputT], Generic[InputT, StateT, OutputT]):
    """An analysis kernel using the standard canonical program-node traversal.

    The executor normally invokes this template independently for each kernel.
    A static kernel that truthfully sets :attr:`fusion_safe` may share a node
    walk only when it has no instance state or ``run`` override and the
    scheduler can prove the remaining conservative eligibility conditions.
    """

    def run(self, graph: ProgramGraph, value: InputT, context: "KernelContext") -> OutputT:
        """Traverse canonical graph nodes and return :meth:`finish` output.

        Args:
            graph: Immutable graph whose canonical node order is visited.
            value: Caller input validated by the scheduler.
            context: Read-only declared dependency access.

        Returns:
            The value returned by :meth:`finish` after every graph node.

        Raises:
            Exception: Callback failures are handled as kernel execution
            failures by the scheduler.

        Side Effects:
            Invokes consumer callbacks; the framework itself mutates only local
            traversal state.
        """

        state = self.begin(value, context)
        for node in graph.nodes:
            state = self.visit(node, state, context)
        return self.finish(state, context)

    @abstractmethod
    def begin(self, value: InputT, context: "KernelContext") -> StateT:
        """Create private traversal state before visiting graph nodes.

        Args:
            value: Validated caller input.
            context: Read-only declared dependency access.

        Returns:
            Private state passed to :meth:`visit` and :meth:`finish`.

        Raises:
            Exception: Scheduler-contained as a kernel execution failure.

        Side Effects:
            Consumer-defined; state must remain private to this traversal.
        """

    @abstractmethod
    def visit(self, node: ProgramNode, state: StateT, context: "KernelContext") -> StateT:
        """Advance private state for one canonical graph node.

        Args:
            node: Current immutable graph node.
            state: Current private traversal state.
            context: Read-only declared dependency access.

        Returns:
            State for the next node or :meth:`finish`.

        Raises:
            Exception: Scheduler-contained as a kernel execution failure.

        Side Effects:
            Consumer-defined; must not mutate framework graph or artifacts.
        """

    @abstractmethod
    def finish(self, state: StateT, context: "KernelContext") -> OutputT:
        """Convert final private traversal state to the output artifact.

        Args:
            state: State after all canonical graph nodes.
            context: Read-only declared dependency access.

        Returns:
            The kernel output to validate against :attr:`output_type`.

        Raises:
            Exception: Scheduler-contained as a kernel execution failure.

        Side Effects:
            Consumer-defined.
        """


def _inherits_traversal_template(kernel_type: type[AnalysisKernel[Any, Any]]) -> bool:
    """Return whether a concrete type inherits the unmodified traversal template.

    Fusion may only replay :class:`TraversalKernel`'s exact callback sequence.
    Checking each class before the base rejects direct and intermediate ``run``
    overrides, including a subclass that merely reassigns the template method.
    """

    if not issubclass(kernel_type, TraversalKernel):
        return False
    for base in kernel_type.__mro__:
        if base is TraversalKernel:
            return True
        if "run" in base.__dict__:
            return False
    return False


def _has_instance_state(kernel: AnalysisKernel[Any, Any]) -> bool:
    """Return whether a traversal instance has state fusion cannot verify.

    The fusion contract admits callback-local state only. An instance dictionary
    with values, or slot declarations on the concrete hierarchy, may affect
    callback interleaving outside that contract and therefore declines fusion.
    """

    try:
        attributes = object.__getattribute__(kernel, "__dict__")
    except AttributeError:
        return True
    if type(attributes) is not dict or attributes:
        return True
    for base in type(kernel).__mro__:
        if base is TraversalKernel:
            return False
        if "__slots__" in base.__dict__:
            return True
    return True


@dataclass(frozen=True, slots=True)
class KernelCall(Generic[InputT, OutputT]):
    """One immutable per-request kernel instance and its opaque caller input.

    Args:
        kernel: Consumer-owned kernel instance.
        input: Caller-owned input checked against the snapshotted declaration.

    Side Effects:
        None. The framework reads this carrier once while establishing a request.
    """

    kernel: AnalysisKernel[InputT, OutputT]
    input: InputT


@dataclass(frozen=True, slots=True)
class KernelOutcome(Generic[OutputT]):
    """One graph-bound structured result for a submitted kernel.

    Args:
        kernel: Exact concrete producer class used as the artifact key.
        graph_digest: Digest of the graph snapshot used to produce the outcome.
        status: Explicit success, failure, or dependency-skip state.
        value: Successful opaque output, including valid ``None`` when declared.
        diagnostics: Immutable redacted framework diagnostics.
        skipped_for: Direct failed or skipped producer classes causing a skip.

    Raises:
        ValueError: If framework-owned status or provenance fields are invalid.

    Side Effects:
        None. Values and classes remain caller-owned opaque references.
    """

    kernel: type[AnalysisKernel[Any, OutputT]]
    graph_digest: str
    status: Literal["succeeded", "failed", "skipped"]
    value: OutputT | None
    diagnostics: tuple[Diagnostic, ...] = ()
    skipped_for: tuple[type[AnalysisKernel[Any, Any]], ...] = ()

    def __post_init__(self) -> None:
        """Validate immutable framework-owned outcome metadata."""

        if not isinstance(self.kernel, type) or type(self.graph_digest) is not str or not self.graph_digest:
            raise ValueError("kernel outcome is invalid")
        if self.status not in ("succeeded", "failed", "skipped"):
            raise ValueError("kernel outcome status is invalid")
        if type(self.diagnostics) is not tuple or type(self.skipped_for) is not tuple:
            raise ValueError("kernel outcome details are invalid")
        if any(type(diagnostic) is not Diagnostic for diagnostic in self.diagnostics):
            raise ValueError("kernel outcome diagnostics are invalid")
        if any(not isinstance(kernel, type) for kernel in self.skipped_for):
            raise ValueError("kernel outcome dependencies are invalid")


class KernelContext:
    """Read-only graph and declared-successful-dependency view for one kernel.

    Instances are created only by the scheduler. They expose no target handles,
    registries, sibling inputs, or undeclared artifacts.
    """

    def __init__(
        self,
        graph: ProgramGraph,
        dependencies: dict[type[AnalysisKernel[Any, Any]], KernelOutcome[Any]],
        dependency_facts: tuple[FactRecord, ...],
    ) -> None:
        """Create an internal dependency-scoped context for one execution.

        Args:
            graph: Immutable graph bound to the current execution.
            dependencies: Direct successful producer outcomes keyed by class.
            dependency_facts: Transitive dependency facts in producer order.

        Side Effects:
            Retains opaque dependency outputs only during consumer execution.
        """

        self._graph = graph
        self._dependencies = dependencies
        self._dependency_facts = dependency_facts

    @property
    def graph(self) -> ProgramGraph:
        """Return the immutable graph bound to the current kernel execution.

        Returns:
            The same graph snapshot passed to the kernel's :meth:`run` method.

        Side Effects:
            None.
        """

        return self._graph

    def require(self, kernel_type: type[AnalysisKernel[Any, OutputU]]) -> OutputU:
        """Return one declared successful direct dependency artifact.

        Args:
            kernel_type: Exact producer class declared in this kernel's
                ``requires`` tuple.

        Returns:
            The successful producer value, including a declared ``None`` value.

        Raises:
            MissingOutputError: If the producer was not declared or is absent.

        Side Effects:
            None.
        """

        outcome = self._dependencies.get(kernel_type)
        if outcome is None or outcome.status != "succeeded":
            raise MissingOutputError()
        return outcome.value  # type: ignore[return-value]

    def facts(self, *, kind: str | None = None) -> tuple[FactRecord, ...]:
        """Return exact fact wrappers from the transitive declared closure.

        Args:
            kind: Optional exact fact-kind filter.

        Returns:
            Fact records in deterministic dependency producer order.

        Raises:
            ValueError: If ``kind`` is neither a string nor ``None``.

        Side Effects:
            None.
        """

        if kind is not None and type(kind) is not str:
            raise ValueError("fact kind is invalid")
        if kind is None:
            return self._dependency_facts
        return tuple(record for record in self._dependency_facts if record.fact.kind == kind)


def _records_for_outcomes(
    outcomes: tuple[KernelOutcome[Any], ...],
    declarations: dict[type[AnalysisKernel[Any, Any]], tuple[KernelMode, type[Any]]],
) -> tuple[FactRecord, ...]:
    """Project successful exact fact wrappers from already ordered outcomes."""

    records: list[FactRecord] = []
    for outcome in outcomes:
        if outcome.status != "succeeded":
            continue
        mode, output_type = declarations[outcome.kernel]
        if output_type is CodeFact and type(outcome.value) is CodeFact:
            records.append(FactRecord(outcome.value, outcome.kernel, outcome.graph_digest, mode))
        elif output_type is CodeFacts and type(outcome.value) is CodeFacts:
            records.extend(
                FactRecord(fact, outcome.kernel, outcome.graph_digest, mode)
                for fact in outcome.value.values
            )
    return tuple(records)


__all__ = [
    "AnalysisKernel",
    "KernelCall",
    "KernelContext",
    "KernelMode",
    "KernelOutcome",
    "TraversalKernel",
]
