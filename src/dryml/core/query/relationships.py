"""Pure direct retained-relationship discovery for Query V3.

The helpers in this module inspect only immutable CDefs and reference values.
They never read Store authority, materialize Objects, allocate ObjectIds, or
capture state. U4 builds closure and occurrence traversal on this direct-edge
seam rather than treating transitive descendants as adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator

from ..cdef_graph import EdgeKind
from ..definition import ConcreteDefinition
from ..links import DefLink
from ..reference_values import ObjectRef, StateRef
from ..utils.graph.path import GraphPath
from ..utils.graph.value import iter_value_edges


class RelationshipKind(str, Enum):
    """Kinds of directed retained identity association used by Query V3."""

    STATE_OBJECT = "state_object"
    OBJECT_DEFINITION = "object_definition"
    MATERIALIZE = "materialize"
    REFERENCE = "reference"


@dataclass(frozen=True, slots=True)
class DirectRelationship:
    """One direct retained identity edge with its literal graph field path."""

    kind: RelationshipKind
    path: GraphPath
    target: ConcreteDefinition | ObjectRef | StateRef


def iter_direct_relationships(
    value: ConcreteDefinition | ObjectRef | StateRef,
) -> Iterator[DirectRelationship]:
    """Yield direct identity relationships carried by one immutable value.

    Association edges have an empty path. CDef edges preserve literal
    materializing versus Ref roles. Reference values expose their rooted CDef
    association; recursive traversal is intentionally left to consumers.
    """

    if isinstance(value, StateRef):
        yield DirectRelationship(RelationshipKind.STATE_OBJECT, GraphPath(), value.object)
        yield from _reference_edges(value, state=True)
        return
    if isinstance(value, ObjectRef):
        yield DirectRelationship(RelationshipKind.OBJECT_DEFINITION, GraphPath(), value.definition)
        yield from _reference_edges(value, state=False)
        return
    if isinstance(value, ConcreteDefinition):
        yield from _definition_edges(value)
        return
    raise TypeError("Query V3 relationships require a ConcreteDefinition, ObjectRef, or StateRef.")


def _reference_edges(value: ObjectRef | StateRef, *, state: bool) -> Iterator[DirectRelationship]:
    reference = value.object if isinstance(value, StateRef) else value
    for edge in _definition_edges(reference.definition):
        if edge.kind is RelationshipKind.REFERENCE:
            yield edge
            continue
        try:
            target = value.at(edge.path) if state else reference.at(edge.path)
        except (TypeError, ValueError):
            # A direct stateless CDef is represented by the CDef association and
            # does not have an independently projectable exact reference value.
            continue
        yield DirectRelationship(edge.kind, edge.path, target)


def _definition_edges(value: ConcreteDefinition) -> Iterator[DirectRelationship]:
    for value_edge in iter_value_edges(value):
        yield from _value_edges(value_edge.value, GraphPath((value_edge.segment,)))


def _value_edges(value: Any, path: GraphPath) -> Iterator[DirectRelationship]:
    if isinstance(value, DefLink):
        if isinstance(value.target, (ConcreteDefinition, ObjectRef, StateRef)):
            kind = (
                RelationshipKind.MATERIALIZE
                if value.kind is EdgeKind.MATERIALIZE
                else RelationshipKind.REFERENCE
            )
            yield DirectRelationship(kind, path, value.target)
        return
    if isinstance(value, (ConcreteDefinition, ObjectRef, StateRef)):
        yield DirectRelationship(RelationshipKind.MATERIALIZE, path, value)
        return
    for value_edge in iter_value_edges(value):
        yield from _value_edges(value_edge.value, path.child(value_edge.segment))
