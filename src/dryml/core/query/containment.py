"""Detached, root-local traversal for retained CDef containment facts.

This module inspects already captured concrete definitions only.  It chooses no
Store authority, resolves no references, and imports none of the query builders
or result surfaces that consume its witnesses.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from ..cdef_graph import EdgeKind
from ..cdef_identity import cdef_node_key, same_cdef
from ..definition import ConcreteDefinition
from ..links import DefLink
from ..reference_values import ObjectRef, StateRef
from ..utils.graph.path import GraphPath
from ..utils.graph.value import iter_value_edges
from .model import (
    ContainmentHop,
    DefinitionOccurrence,
    ReferenceOccurrence,
    validate_containment_policy,
    validate_containment_target,
)


ContainmentTarget = ConcreteDefinition | ObjectRef | StateRef
ContainmentOccurrence = DefinitionOccurrence | ReferenceOccurrence


@dataclass(frozen=True, slots=True)
class _DirectContainmentEdge:
    """One retained direct CDef or exact-reference boundary."""

    path: GraphPath
    target: ContainmentTarget
    kind: EdgeKind


def iter_containment_occurrences(
    roots: Iterable[ConcreteDefinition],
    target: ContainmentTarget,
    *,
    edges: str = "materialize",
    contains_ref: bool = False,
) -> Iterator[ContainmentOccurrence]:
    """Yield every qualified non-empty root-to-target containment witness.

    Args:
        roots: Detached authoritative CDef roots selected by the caller.
        target: Concrete CDef, ObjectRef, or StateRef to match exactly.
        edges: Edge kinds permitted at every hop: ``"materialize"``,
            ``"ref"``, or ``"all"``.
        contains_ref: Whether a witness must contain at least one ``REF`` hop.

    Yields:
        DefinitionOccurrence or ReferenceOccurrence values retaining the full
        typed path and ordered, literal hop evidence.

    Raises:
        TypeError: If roots or target values are unsupported, or the filter is
            not an exact bool.
        ValueError: If ``edges`` is unsupported.

    Side Effects:
        None. The walker does not load, materialize, resolve, or dereference
        retained values.  Cycle protection is path-local, so shared nodes retain
        one occurrence for each distinct root-to-target path.
    """

    target = validate_containment_target(target)
    edges, contains_ref = validate_containment_policy(edges, contains_ref)
    for root in roots:
        if not isinstance(root, ConcreteDefinition):
            raise TypeError(
                f"Containment roots must be ConcreteDefinition values, got {type(root).__name__}."
            )
        yield from _iter_root_occurrences(root, target, edges, contains_ref)


def iter_containment_owners(
    roots: Iterable[ConcreteDefinition],
    target: ContainmentTarget,
    *,
    edges: str = "materialize",
    contains_ref: bool = False,
) -> Iterator[ConcreteDefinition]:
    """Yield roots having at least one qualified non-empty path to ``target``.

    Args:
        roots: Detached authoritative CDef roots selected by the caller.
        target: Concrete CDef, ObjectRef, or StateRef to match exactly.
        edges: Edge kinds permitted at every hop.
        contains_ref: Whether a qualifying path must contain a ``REF`` hop.

    Yields:
        Each supplied root with existential qualification, in supplied order.

    Raises:
        TypeError: If roots or target values are unsupported, or the filter is
            not an exact bool.
        ValueError: If ``edges`` is unsupported.

    Side Effects:
        None. This projection traversal uses private CDef identities and a
        ``(node, has_ref)`` memo without resolving retained values.
    """

    target = validate_containment_target(target)
    edges, contains_ref = validate_containment_policy(edges, contains_ref)
    for root in roots:
        if not isinstance(root, ConcreteDefinition):
            raise TypeError(
                f"Containment roots must be ConcreteDefinition values, got {type(root).__name__}."
            )
        if _root_contains_target(root, target, edges, contains_ref):
            yield root


def visit_containment_occurrences(
    roots: Iterable[ConcreteDefinition],
    target: ContainmentTarget,
    visit: Callable[[ContainmentOccurrence], None],
    *,
    edges: str = "materialize",
    contains_ref: bool = False,
) -> None:
    """Call ``visit`` once for each root-local containment occurrence.

    Args:
        roots: Detached authoritative CDef roots selected by the caller.
        target: Concrete CDef, ObjectRef, or StateRef to match exactly.
        visit: Consumer invoked in iterator order for each qualified witness.
        edges: Edge kinds permitted at every hop.
        contains_ref: Whether a qualifying path must contain a ``REF`` hop.

    Raises:
        TypeError: Propagated input validation errors or when ``visit`` is not
            callable.
        ValueError: If ``edges`` is unsupported.

    Side Effects:
        Invokes ``visit`` only; traversal itself does not resolve or materialize
        retained values.
    """

    if not callable(visit):
        raise TypeError("visit must be callable.")
    for occurrence in iter_containment_occurrences(
        roots, target, edges=edges, contains_ref=contains_ref,
    ):
        visit(occurrence)


def _iter_root_occurrences(
    root: ConcreteDefinition,
    target: ContainmentTarget,
    edges: str,
    contains_ref: bool,
) -> Iterator[ContainmentOccurrence]:
    def visit(
        node: ConcreteDefinition,
        path: GraphPath,
        hops: tuple[ContainmentHop, ...],
        has_ref: bool,
        active_nodes: frozenset[object],
    ) -> Iterator[ContainmentOccurrence]:
        for edge in _iter_direct_containment_edges(node):
            if not _permits(edges, edge.kind):
                continue
            next_has_ref = has_ref or edge.kind is EdgeKind.REF
            next_path = path.join(edge.path)
            next_hops = hops + (ContainmentHop(edge.path, edge.kind),)
            if isinstance(edge.target, ConcreteDefinition):
                if _matches(target, edge.target) and (
                    not contains_ref or next_has_ref
                ):
                    yield DefinitionOccurrence(root, next_path, edge.target, next_hops)
                child_key = cdef_node_key(edge.target)
                if child_key not in active_nodes:
                    yield from visit(
                        edge.target,
                        next_path,
                        next_hops,
                        next_has_ref,
                        active_nodes | frozenset((child_key,)),
                    )
            elif _matches(target, edge.target) and (
                not contains_ref or next_has_ref
            ):
                yield ReferenceOccurrence(root, next_path, edge.target, next_hops)

    root_key = cdef_node_key(root)
    yield from visit(root, GraphPath(), (), False, frozenset((root_key,)))


def _root_contains_target(
    root: ConcreteDefinition,
    target: ContainmentTarget,
    edges: str,
    contains_ref: bool,
) -> bool:
    memo: dict[tuple[object, bool], bool] = {}

    def visit(
        node: ConcreteDefinition,
        has_ref: bool,
        active_nodes: frozenset[object],
    ) -> bool:
        node_key = cdef_node_key(node)
        memo_key = (node_key, has_ref)
        if memo_key in memo:
            return memo[memo_key]
        for edge in _iter_direct_containment_edges(node):
            if not _permits(edges, edge.kind):
                continue
            next_has_ref = has_ref or edge.kind is EdgeKind.REF
            if _matches(target, edge.target) and (
                not contains_ref or next_has_ref
            ):
                memo[memo_key] = True
                return True
            if isinstance(edge.target, ConcreteDefinition):
                child_key = cdef_node_key(edge.target)
                if child_key not in active_nodes and visit(
                    edge.target,
                    next_has_ref,
                    active_nodes | frozenset((child_key,)),
                ):
                    memo[memo_key] = True
                    return True
        memo[memo_key] = False
        return False

    return visit(root, False, frozenset((cdef_node_key(root),)))


def _iter_direct_containment_edges(
    cdef: ConcreteDefinition,
) -> Iterator[_DirectContainmentEdge]:
    for value_edge in iter_value_edges(cdef):
        yield from _iter_edges_from_value(
            value_edge.value, GraphPath((value_edge.segment,)),
        )


def _iter_edges_from_value(
    value: Any,
    path: GraphPath,
) -> Iterator[_DirectContainmentEdge]:
    if isinstance(value, DefLink):
        target = value.target
        if isinstance(target, (ConcreteDefinition, ObjectRef, StateRef)):
            yield _DirectContainmentEdge(path, target, value.kind)
        return
    if isinstance(value, (ConcreteDefinition, ObjectRef, StateRef)):
        yield _DirectContainmentEdge(path, value, EdgeKind.MATERIALIZE)
        return
    for value_edge in iter_value_edges(value):
        yield from _iter_edges_from_value(
            value_edge.value, path.child(value_edge.segment),
        )


def _matches(target: ContainmentTarget, value: ContainmentTarget) -> bool:
    if isinstance(target, ConcreteDefinition):
        return isinstance(value, ConcreteDefinition) and same_cdef(target, value)
    return type(value) is type(target) and value == target


def _permits(policy: str, kind: EdgeKind) -> bool:
    return policy == "all" or policy == kind.value
