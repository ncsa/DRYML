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
MethodPortKind = Literal["element", "stream"]


@dataclass(frozen=True, slots=True)
class MethodPort:
    """One immutable graph port and its currently known element specification.

    Args:
        kind: ``"element"`` for U4's ordinary call ports or ``"stream"`` for
            a future iterator port declaration.
        spec: Immutable normalized element facts, or ``None`` when unknown.

    U4 creates element ports only. Stream-port execution, cursor ownership, and
    conversion edges are deliberately deferred to later units.
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


class MethodGraph:
    """A read-only Method-family view backed by its owner's preparation state.

    Args:
        owner: The Method whose local preparation state owns this graph's facts.

    Constructing a graph is inert. :meth:`learn` delegates to the owner's normal
    Method preparation machinery, retaining occurrence-local selections without
    changing child Method first-call caches. U4 supports element ports only;
    :meth:`iterator` is intentionally unavailable until stream execution exists.
    """

    __slots__ = ("_owner_ref", "__weakref__")

    def __init__(self, owner: object) -> None:
        """Create an inert view of one weakly held Method owner.

        Raises:
            TypeError: If ``owner`` cannot support a weak reference.
        """

        object.__setattr__(self, "_owner_ref", weakref.ref(owner))

    def __setattr__(self, name: str, value: object) -> None:
        """Reject mutation after construction so graph structure remains stable."""

        raise AttributeError("MethodGraph is immutable.")

    @property
    def nodes(self) -> tuple[MethodGraphNode, ...]:
        """Return immutable source and Method occurrence facts known to this view."""

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
        """Return retained conversion edges, which are empty until U6.

        The empty immutable surface reserves graph capacity for explicit local
        conversion facts without letting U4 select or execute a conversion.
        """

        return ()

    @property
    def input_specs(self) -> tuple[SpecTree, ...]:
        """Return fresh public copies of the graph's known root input specs."""

        return tuple(spec_from_node(port.spec) for port in self.source_nodes[0].outputs if port.spec is not None)

    def learn(
        self,
        input_spec: SpecTree | None = None,
        *additional_input_specs: SpecTree,
        strategy: str = "local",
        output_spec: SpecTree | None = None,
    ) -> None:
        """Prepare this graph through the owner's existing Method state.

        Args:
            input_spec: Optional first known input specification.
            *additional_input_specs: Known later positional specifications.
            strategy: Preparation strategy; U4 supports only ``"local"``.
            output_spec: Optional raw-result validation specification.

        Raises:
            ValueError: If a strategy other than ``"local"`` is requested.
            MethodError: If the weak owner is no longer live.

        Side Effects:
            Stores immutable occurrence selections in the owner's process-local
            preparation state. It invokes no candidate or source.
        """

        self._owner()._learn_graph(self, input_spec, additional_input_specs, strategy, output_spec)

    def eager(self) -> None:
        """Invalidate this graph's retained local preparation facts.

        Side Effects:
            Delegates to the owner Method's normal eager reset. The graph remains
            inspectable but no longer exposes selected invokers.
        """

        self._owner().eager()

    def iterator(self):
        """Reject stream execution until U5 owns cursor lifecycle behavior.

        Raises:
            NotImplementedError: U4 supplies graph facts only, not stream
                execution or Dataset cursor ownership.
        """

        raise NotImplementedError("MethodGraph iterator execution is provided by U5.")

    def _owner(self):
        owner = self._owner_ref()
        if owner is None:
            from .errors import MethodError

            raise MethodError("The MethodGraph owner is no longer live.")
        return owner


__all__ = ["MethodGraph", "MethodGraphNode", "MethodGraphNodeKind", "MethodPort", "MethodPortKind"]
