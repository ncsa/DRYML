"""Inspectable immutable facts for locally prepared Method graphs.

The IR records logical Method occurrences and their known element-port facts. It
does not open sources, execute stream nodes, convert values, or compile code;
those responsibilities remain with later stream and conversion layers.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Literal

from dryml.core.tensor_spec import SpecTree

from .implementation import PreparedMethodInvoker
from .signature import MethodCallNode, spec_from_node
from .traits import Traits

MethodGraphNodeKind = Literal["source", "method"]
MethodPortKind = Literal["element", "iterator"]


@dataclass(frozen=True, slots=True)
class MethodPort:
    """One immutable graph port and its currently known element specification.

    Args:
        kind: ``"element"`` for ordinary call ports or ``"iterator"`` for an
            explicit ordered stream iterator port.
        spec: Immutable normalized element facts, or ``None`` when unknown.

    U4 creates element ports only. U5 adds iterator ports and graph-owned cursor
    execution; conversion edges remain deliberately deferred.
    """

    kind: MethodPortKind
    spec: MethodCallNode | None = None


@dataclass(frozen=True, slots=True)
class MethodGraphNode:
    """One occurrence-indexed source or Method node in a prepared graph.

    Args:
        occurrence: Stable zero-based occurrence index within this graph view.
        kind: Whether this is a virtual source fact or a Method occurrence.
        method_type: The Method class for Method nodes, otherwise ``None``.
        inputs: Ordered input-port facts.
        outputs: Ordered output-port facts.
        selected: Retained local invoker when this occurrence was prepared.
        traits: Traits of the selected implementation, when known.

    The virtual source node records only supplied specification facts. It never
    opens a Dataset source. A selected invoker retains a weak receiver and can be
    inspected for future compilation, but no compiler or executor is provided.
    """

    occurrence: int
    kind: MethodGraphNodeKind
    method_type: type | None
    inputs: tuple[MethodPort, ...] = ()
    outputs: tuple[MethodPort, ...] = ()
    selected: PreparedMethodInvoker | None = None
    traits: Traits | None = None
    stream_node: object | None = None


class MethodGraph:
    """A read-only Method-family or Dataset-stream graph view.

    Args:
        owner: The weakly held Method or Dataset owner of this graph's facts.
        stream_plan: Optional Dataset-owned local stream preparation plan.

    Constructing a graph is inert. Method graphs delegate to the owner's normal
    preparation state; Dataset graphs retain qualified iterator-port facts only
    after :meth:`learn`. Neither form opens a source during construction.
    """

    __slots__ = ("_owner_ref", "_stream_plan", "__weakref__")

    def __init__(self, owner: object, *, stream_plan: object | None = None) -> None:
        """Create an inert view of one weakly held Method or Dataset owner.

        Raises:
            TypeError: If ``owner`` cannot support a weak reference.
        """

        object.__setattr__(self, "_owner_ref", weakref.ref(owner))
        object.__setattr__(self, "_stream_plan", stream_plan)

    def __setattr__(self, name: str, value: object) -> None:
        """Reject mutation after construction so graph structure remains stable."""

        raise AttributeError("MethodGraph is immutable.")

    @property
    def nodes(self) -> tuple[MethodGraphNode, ...]:
        """Return immutable source and Method occurrence facts known to this view."""

        if self._stream_plan is not None:
            return self._stream_plan.nodes
        owner = self._owner()
        return owner._graph_nodes_for(self)

    @property
    def source_nodes(self) -> tuple[MethodGraphNode, ...]:
        """Return virtual source occurrences without opening any source."""

        return tuple(node for node in self.nodes if node.kind == "source")

    @property
    def method_nodes(self) -> tuple[MethodGraphNode, ...]:
        """Return Method occurrences, including their selected local invokers."""

        return tuple(node for node in self.nodes if node.kind == "method")

    @property
    def conversion_edges(self) -> tuple[object, ...]:
        """Return immutable local dense handoff facts selected during preparation."""

        if self._stream_plan is not None:
            return self._stream_plan.conversion_edges
        return self._owner()._graph_conversion_edges_for(self)

    @property
    def input_specs(self) -> tuple[SpecTree, ...]:
        """Return fresh public copies of the graph's known root input specs."""

        return tuple(
            spec_from_node(port.spec)
            for source in self.source_nodes
            for port in source.outputs
            if port.spec is not None
        )

    def learn(
        self,
        input_spec: SpecTree | None = None,
        *additional_input_specs: SpecTree,
        strategy: str = "local",
        output_spec: SpecTree | None = None,
    ) -> None:
        """Prepare this graph through its Method or Dataset local state.

        Args:
            input_spec: Optional first known input specification or Dataset source
                assertion.
            *additional_input_specs: Later positional specs or Dataset source
                assertions.
            strategy: Preparation strategy; only ``"local"`` is supported.
            output_spec: Optional Method raw-result validation or Dataset output
                assertion.

        Raises:
            ValueError: If a strategy other than ``"local"`` is requested.
            MethodError: If a Method owner is no longer live.
            NotImplementedError: If a Dataset pipeline has an unqualified stream
                operator.

        Side Effects:
            Stores immutable occurrence selections in local preparation state. It
            invokes no candidate body or Dataset source.
        """

        if self._stream_plan is not None:
            self._stream_plan.learn(input_spec, additional_input_specs, strategy=strategy, output_spec=output_spec)
            return
        self._owner()._learn_graph(self, input_spec, additional_input_specs, strategy, output_spec)

    def eager(self) -> None:
        """Invalidate this graph's retained local preparation facts.

        Side Effects:
            Delegates to the owner Method's normal eager reset or clears Dataset
            graph selections. The graph remains inspectable but no longer exposes
            retained selected invokers.
        """

        if self._stream_plan is not None:
            self._stream_plan.eager()
            return
        self._owner().eager()

    def iterator(self):
        """Open one isolated cursor for a prepared Dataset iterator-port graph.

        Raises:
            NotImplementedError: If this is an element-only Method graph.
            RuntimeError: If a Dataset graph was not prepared with :meth:`learn`.

        Side Effects:
            The returned cursor opens sources lazily and owns every acquired
            source/output iterator for exactly one traversal.
        """

        if self._stream_plan is None:
            raise NotImplementedError("MethodGraph iterator execution requires Dataset iterator ports.")
        return self._stream_plan.iterator()

    def _owner(self):
        owner = self._owner_ref()
        if owner is None:
            from .errors import MethodError

            raise MethodError("The MethodGraph owner is no longer live.")
        return owner


__all__ = ["MethodGraph", "MethodGraphNode", "MethodGraphNodeKind", "MethodPort", "MethodPortKind"]
