"""Pure direct retained-relationship discovery for Query V3.

The helpers in this module inspect only immutable CDefs and reference values.
They never read Store authority, materialize Objects, allocate ObjectIds, or
capture state. U4 builds closure and occurrence traversal on this direct-edge
seam rather than treating transitive descendants as adjacency.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Iterator

from ..cdef_identity import cdef_node_key
from ..cdef_graph import EdgeKind
from ..definition import ConcreteDefinition
from ..links import DefLink
from ..reference_values import ObjectRef, StateRef
from ..node import Node
from ..state_values import LocalState
from ..utils.graph.path import GraphPath, graph_path_sort_key
from ..utils.graph.value import iter_value_edges


class RelationshipKind(str, Enum):
    """Kinds of directed retained identity association used by Query V3."""

    STATE_OBJECT = "state_object"
    OBJECT_DEFINITION = "object_definition"
    MATERIALIZE = "materialize"
    REFERENCE = "reference"
    STATE_CREATION = "state_creation"


@dataclass(frozen=True, slots=True)
class RelationshipHop:
    """One typed direct relationship hop and its local graph-field segment.

    Args:
        kind: Literal retained association or graph-edge role.
        path: Local canonical GraphPath segment. Association paths are empty.

    Raises:
        TypeError: If ``kind`` or ``path`` is not a supported typed value.
    """

    kind: RelationshipKind
    path: GraphPath

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RelationshipKind):
            raise TypeError("Relationship hop kind must be a RelationshipKind.")
        if not isinstance(self.path, GraphPath):
            raise TypeError("Relationship hop path must be a GraphPath.")


@dataclass(frozen=True, slots=True)
class RelationshipPath:
    """Immutable ordered typed hops for one retained relationship witness.

    Association hops retain their empty field segments, so a StateRef-to-
    ObjectRef relationship is distinguishable from an empty root path. This
    query-owned representation never changes the authoritative GraphPath codec.
    """

    hops: tuple[RelationshipHop, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.hops, tuple) or not all(
                isinstance(hop, RelationshipHop) for hop in self.hops
        ):
            raise TypeError("RelationshipPath hops must be a tuple of RelationshipHop values.")

    @classmethod
    def from_hops(
        cls, hops: Iterable[RelationshipHop | tuple[RelationshipKind, GraphPath]],
    ) -> "RelationshipPath":
        """Construct a path from typed hops without normalizing GraphPath data."""

        return cls(tuple(
            hop if isinstance(hop, RelationshipHop) else RelationshipHop(*hop)
            for hop in hops
        ))

    @property
    def graph_path(self) -> GraphPath:
        """Return the concatenated legacy graph-field path for path matching."""

        path = GraphPath()
        for hop in self.hops:
            path = path.join(hop.path)
        return path

    @property
    def sort_key(self) -> tuple[tuple[str, bytes], ...]:
        """Return a deterministic typed-path ordering key."""

        return tuple((hop.kind.value, graph_path_sort_key(hop.path)) for hop in self.hops)

    def child(self, kind: RelationshipKind, path: GraphPath) -> "RelationshipPath":
        """Return this path extended by one direct typed hop."""

        return RelationshipPath((*self.hops, RelationshipHop(kind, path)))


@dataclass(frozen=True, slots=True)
class EdgePolicy:
    """Immutable permitted relationship kinds for V3 closure and nesting.

    ``ALL`` follows every retained relationship. ``OWNED`` includes both
    associations and materializing ownership links; ``ASSOCIATIONS`` includes
    only StateRef/ObjectRef and ObjectRef/CDef associations.
    """

    kinds: frozenset[RelationshipKind]

    ALL: ClassVar["EdgePolicy"]
    OWNED: ClassVar["EdgePolicy"]
    ASSOCIATIONS: ClassVar["EdgePolicy"]
    STATE_OBJECT: ClassVar["EdgePolicy"]

    def __post_init__(self) -> None:
        if not isinstance(self.kinds, frozenset) or not all(
                isinstance(kind, RelationshipKind) for kind in self.kinds
        ):
            raise TypeError("EdgePolicy kinds must be a frozenset of RelationshipKind values.")

    def permits(self, kind: RelationshipKind) -> bool:
        """Return whether this literal policy permits one direct relationship."""

        return kind in self.kinds


EdgePolicy.ALL = EdgePolicy(frozenset(RelationshipKind))
EdgePolicy.OWNED = EdgePolicy(frozenset((
    RelationshipKind.STATE_OBJECT,
    RelationshipKind.OBJECT_DEFINITION,
    RelationshipKind.MATERIALIZE,
)))
EdgePolicy.ASSOCIATIONS = EdgePolicy(frozenset((
    RelationshipKind.STATE_OBJECT,
    RelationshipKind.OBJECT_DEFINITION,
)))
EdgePolicy.STATE_OBJECT = EdgePolicy(frozenset((RelationshipKind.STATE_OBJECT,)))


def normalize_edge_policy(value: EdgePolicy | None) -> EdgePolicy:
    """Validate one V3 policy, defaulting only an omitted policy to ``ALL``."""

    if value is None:
        return EdgePolicy.ALL
    if not isinstance(value, EdgePolicy):
        raise TypeError("Relationship traversal edges must be an EdgePolicy.")
    return value


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

    if isinstance(value, LocalState):
        yield DirectRelationship(RelationshipKind.STATE_CREATION, GraphPath(), value.creation_cdef)
        return
    if isinstance(value, Node):
        if value.state_spec():
            yield DirectRelationship(RelationshipKind.STATE_OBJECT, GraphPath(), value.object_view())
        elif value.object_id is not None:
            yield DirectRelationship(RelationshipKind.OBJECT_DEFINITION, GraphPath(), value.creation_view())
        yield from _definition_edges(value)
        return
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


def _local_discovery_fields(node):
    """Read available field names without resolving an inert authored call.

    Unresolved soft calls retain positional paths and explicit keyword names.
    Their positional values cannot supply named scalar postings, but real Node
    children remain discoverable when the soft recipe is an independent root.
    """
    from .node_pattern import _parameters
    from ..utils.graph.path import Arg, Parameter

    if not isinstance(node, ConcreteDefinition) and not isinstance(node.cls, type):
        yield from ((Arg(index), value) for index, value in enumerate(node.args or ()))
        yield from ((Parameter(name), value) for name, value in node.kwargs.items())
    else:
        yield from ((Parameter(name), value) for name, value in _parameters(node).items())


def _local_discovery_edges(node):
    """Yield direct Node edges only; paths are local presentation, never identity."""
    from ..quoted import QuotedDef, SelectorSpec

    def walk(value, path, role="materialize"):
        if isinstance(value, DefLink):
            yield from walk(value.target, path, value.kind.value)
        elif isinstance(value, Node):
            yield path, role, value
        elif isinstance(value, (ObjectRef, StateRef, QuotedDef, SelectorSpec)):
            return
        else:
            for edge in iter_value_edges(value):
                yield from walk(edge.value, path.child(edge.segment), role)

    for segment, value in _local_discovery_fields(node):
        yield from walk(value, GraphPath((segment,)))


def _local_discovery_tokens(node):
    """Return exact necessary local anchors, excluding subclass and absent facets."""
    from ..utils.graph.path import Parameter
    if node.object_id is not None:
        yield ["identity", node.object_id.to_data()]
    if node.state is not None:
        yield ["state", node.state.key.cdef_digest, node.state.key.codec_hash]
    for segment, value in _local_discovery_fields(node):
        if isinstance(segment, Parameter) and type(value) in (str, int, bool, type(None)):
            yield ["scalar", segment.name, type(value).__name__, value]


class _DiscoveryPreparation:
    """Operation-local shared semantic tables, without external authority proofs."""

    def __init__(self, decoded_owner, block_digest):
        from ..node_prepare import NodePreparation
        from ..node_views import _project, OBJECT_PROJECTION_VIEW
        from ..definition import ConcreteNode
        self.decoded, self.block_digest = decoded_owner, block_digest
        sources = [node for _, node in sorted(decoded_owner._transport_nodes.items())]
        memo, adapted = {}, {}
        self.creation = {id(n): n.creation_view() if isinstance(n, ConcreteDefinition) else n for n in sources}
        self.objects = {id(n): _project(n, compatibility=False, _memo=memo, _adapted=adapted) for n in sources}
        self.owner = NodePreparation((*sources, *self.creation.values(), *self.objects.values()), previous=decoded_owner,
                                     max_cached_keys=decoded_owner.max_cached_keys,
                                     max_cached_bytes=max(1, decoded_owner.max_cached_bytes // 8))
        self.fact_keys = {}
        # Preparation slots can change when tied unordered values are traversed
        # in a different order. A later membership delta must still join facts
        # written earlier for the same immutable block, without rewriting it.
        for node in sources:
            locator = decoded_owner._transport_locators[id(node)]
            self.fact_keys[node._view] = block_digest + ":" + locator["table"] + ":" + str(locator["slot"])
        for projection, selections in (("creation", self.creation), ("object", self.objects)):
            for node in sources:
                selected = selections[id(node)]
                self.fact_keys.setdefault(selected._view, self.fact_keys[node._view] + ":" + projection)
        self.edges = {id(n): tuple(_local_discovery_edges(n)) for n in self.owner.nodes}
        self.summaries = {}
        identities, occurrences = {}, {}

        def summary(node):
            marker = id(node)
            if marker in self.summaries:
                return self.summaries[marker]
            occurrence = occurrences.setdefault(node._occurrence, len(occurrences))
            stateful = isinstance(node, ConcreteNode) and node._stateful_role
            oid = node.object_id
            identity = 0 if not stateful or oid is None else 1 << identities.setdefault(oid, len(identities))
            owned = (1 << occurrence) if stateful else 0
            has_id, has_state = oid is not None, node.state is not None
            complete = isinstance(node, ConcreteNode) and (not stateful or has_id and has_state)
            # Legacy references are not admitted by require_saved's owned grammar.
            def legacy(value):
                if isinstance(value, (ObjectRef, StateRef)):
                    return True
                if isinstance(value, DefLink):
                    return value.kind.value == "materialize" and legacy(value.target)
                if isinstance(value, Node):
                    return False
                return any(legacy(edge.value) for edge in iter_value_edges(value))
            complete &= not any(legacy(v) for _, v in _local_discovery_fields(node))
            for _, role, child in self.edges[marker]:
                if role != "materialize":
                    continue
                ci, cs, cc, co, ids = summary(child)
                has_id |= ci
                has_state |= cs
                complete &= cc
                owned |= co
                identity |= ids
            result = has_id, has_state, complete and owned.bit_count() == identity.bit_count(), owned, identity
            self.summaries[marker] = result
            return result
        for node in sources:
            summary(node)
        self.object_domain = OBJECT_PROJECTION_VIEW.name

    def fact_key(self, node):
        return self.fact_keys[node._view]

    def object_digest(self, node):
        import hashlib
        key = self.owner.semantic_key(node, policy=self.object_domain)
        return hashlib.sha256(self.object_domain.encode("ascii") + b"\0" + key).hexdigest()

    def facts(self):
        """Yield each local token/edge once per authenticated block, not per root."""
        from ..store.records import _json_bytes
        for node in self.owner.nodes:
            parent = self.fact_key(node)
            for token in _local_discovery_tokens(node):
                encoded = _json_bytes(token).decode("ascii")
                yield "discovery_feature", parent + ":" + encoded, {"node": parent, "token": encoded}
            for ordinal, (path, role, child) in enumerate(self.edges[id(node)]):
                # Unordered ordinals/fingerprints are intentionally not persisted.
                unordered = any(s["kind"] == "set_member" for s in path.to_data()["segments"])
                yield "discovery_link", parent + ":" + str(ordinal), {
                    "parent": parent, "child": self.fact_key(child), "role": role,
                    "path": None if unordered else path.to_data()}


def _discovery_preparation(decoded_owner, block_digest):
    """Borrow bounded immutable query facts under the decoded owner's existing LRU.

    Eviction/clear changes work only. Reserve the entire child fact budget, charge
    derived tables/handles conservatively, and retain no membership/availability
    or validation verdict. The caller still authenticates fresh source bytes.
    """
    from ..node_prepare import _metadata_size
    key = ("discovery-preparation", block_digest)
    with decoded_owner._lock:
        cached = decoded_owner._facts.get(key)
        if cached is not None:
            decoded_owner._facts.move_to_end(key)
            return cached
        p = _DiscoveryPreparation(decoded_owner, block_digest)
        edges = {marker: tuple((path.to_data(), role, id(child)) for path, role, child in links)
                 for marker, links in p.edges.items()}
        metadata = (p.owner._rows, p.owner._children, p.owner._closures, p.owner._incidences,
                    p.owner._construction, edges, p.summaries,
                    tuple(p.creation), tuple(p.objects), tuple(p.fact_keys.values()))
        size = 2048 + 4096 * len(p.owner.nodes) + 4 * _metadata_size(metadata, decoded_owner.max_cached_bytes // 4)
        decoded_owner._cache_put(key, p, size + p.owner.max_cached_bytes)
        return p


def _discovery_rows(root, record, *, preparation=None):
    """Yield compact source contributions using portable authenticated locators.

    A supplied block preparation serves every requested descendant/domain. Local
    facts are exported separately once for the block; no row embeds a subtree.
    The private convenience form decodes the block representation when no owner
    is supplied, so it follows the same portable-locator contract.
    """
    if preparation is None:
        from ..node_codec import encode_node_graph, _decode_node_graph_views
        from ..node_prepare import NodePreparation
        packet = encode_node_graph(NodePreparation((root,)))
        roots, owner = _decode_node_graph_views(packet, with_owner=True)
        preparation = _discovery_preparation(owner, record.block_digest)
        root = roots[0]
    p = preparation
    pending, seen = [root], set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        pending.extend(child for _, _, child in reversed(p.edges[id(node)]))
        has_id, has_state, saved, _, _ = p.summaries[id(node)]
        choices = [("cdef", "creation" if isinstance(node, ConcreteDefinition) else "recipe", p.creation[id(node)])]
        if has_id or record.root_kind == "saved":
            choices.append(("object_ref", "object", p.objects[id(node)]))
        if saved and (has_state or record.root_kind == "saved"):
            choices.append(("state_ref", "saved", node))
        for kind, projection, selected in choices:
            obj = p.objects[id(node)]
            yield {"kind": kind, "digest": p.owner.digest(selected), "root": record.to_data(),
                   "key_kind": kind, "fact": p.fact_key(selected),
                   "object_digest": (p.object_digest(obj)
                                     if kind in {"object_ref", "state_ref"} else None),
                   "recipe_digest": p.owner.digest(selected, domain="dryml-recipe-root/v1") if kind == "cdef" else None,
                   "saved_digest": p.owner.digest(selected, domain="dryml-saved-root/v1") if kind == "state_ref" else None,
                   "locator": p.decoded._transport_locators[id(node)], "projection": projection,
                   "is_root": node is root,
                   "stored": kind == "object_ref" or node is root and record.root_kind == ("recipe" if kind == "cdef" else "saved")}


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


RelationshipValue = ConcreteDefinition | ObjectRef | StateRef
RelationshipMatcher = Callable[[RelationshipValue], bool]


class RelationshipDepthExceeded(Exception):
    """Raised when a requested relationship traversal exceeds its depth budget."""


def relationship_closure(
    roots: Iterable[RelationshipValue], *, edges: EdgePolicy | None = None,
    max_depth: int | None = None,
) -> tuple[RelationshipValue, ...]:
    """Return complete identities reachable from ``roots``, including roots.

    Args:
        roots: Detached CDefs, ObjectRefs, or StateRefs to expand.
        edges: Immutable permitted relationship policy, defaulting to all kinds.

    Returns:
        Each complete retained identity once, without Store reads or payload IO.

    Raises:
        TypeError: If a root or policy is unsupported.
    """

    from .identity import identity_key

    policy = normalize_edge_policy(edges)
    _validate_max_depth(max_depth)
    retained: dict[object, RelationshipValue] = {}
    pending = [(value, 0) for value in roots]
    while pending:
        value, depth = pending.pop()
        if not isinstance(value, (ConcreteDefinition, ObjectRef, StateRef, LocalState)):
            raise TypeError("Relationship roots must be ConcreteDefinition, ObjectRef, or StateRef values.")
        key = identity_key(value)
        if key in retained:
            continue
        retained[key] = value
        for edge in iter_direct_relationships(value):
            if not policy.permits(edge.kind):
                continue
            if max_depth is not None and depth >= max_depth:
                raise RelationshipDepthExceeded("Query V3 relationship depth budget exceeded.")
            pending.append((edge.target, depth + 1))
    return tuple(retained.values())


def iter_relationship_occurrences(
    roots: Iterable[RelationshipValue],
    matches: RelationshipMatcher,
    *,
    edges: EdgePolicy | None = None,
    through: frozenset[RelationshipKind] = frozenset(),
    path: RelationshipPath | GraphPath | None = None,
    max_occurrences: int | None = None,
    max_depth: int | None = None,
) -> Iterator[object]:
    """Yield every non-empty root-local typed occurrence matching ``matches``.

    Shared nodes remain visible through each distinct path. Repeated nodes stop
    only the active path, preserving the established containment-cycle behavior.
    ``max_occurrences`` caps only raw output; direct projections use the
    existential helpers below and intentionally ignore it.
    """


    policy = normalize_edge_policy(edges)
    _validate_traversal_inputs(matches, through, path, max_occurrences, max_depth)
    if max_occurrences == 0:
        return
    seen_occurrences = set()
    for root in roots:
        _require_relationship_value(root, "Relationship roots")
        for occurrence in _iter_root_occurrences(
            root, root, matches, policy, through, path,
            frozenset((_node_key(root),)), RelationshipPath(),
            0, max_depth,
            lambda occurrence: _retain_occurrence(
                occurrence, seen_occurrences, max_occurrences,
            ),
        ):
            yield occurrence
            if max_occurrences is not None and len(seen_occurrences) >= max_occurrences:
                return


def iter_relationship_owners(
    roots: Iterable[RelationshipValue],
    matches: RelationshipMatcher,
    *,
    edges: EdgePolicy | None = None,
    through: frozenset[RelationshipKind] = frozenset(),
    path: RelationshipPath | GraphPath | None = None,
    max_depth: int | None = None,
) -> Iterator[RelationshipValue]:
    """Yield roots with an existential qualified non-empty target path.

    This traversal memoizes private node/path-policy states rather than
    enumerating raw occurrence paths, so duplicate DAG paths cannot make owner
    projection exponential.
    """

    policy = normalize_edge_policy(edges)
    _validate_traversal_inputs(matches, through, path, None, max_depth)
    for root in roots:
        _require_relationship_value(root, "Relationship roots")
        if _root_has_match(root, matches, policy, through, path, max_depth):
            yield root


def iter_relationship_targets(
    roots: Iterable[RelationshipValue],
    matches: RelationshipMatcher,
    *,
    edges: EdgePolicy | None = None,
    through: frozenset[RelationshipKind] = frozenset(),
    path: RelationshipPath | GraphPath | None = None,
    max_depth: int | None = None,
) -> Iterator[RelationshipValue]:
    """Yield distinct reachable terminals using existential traversal states.

    Raw occurrence caps are deliberately absent: targets are a direct
    projection of the uncapped relationship relation, not a projection of a
    previously materialized occurrence collection.
    """

    from .identity import identity_key

    policy = normalize_edge_policy(edges)
    _validate_traversal_inputs(matches, through, path, None, max_depth)
    seen_targets = set()
    for root in roots:
        _require_relationship_value(root, "Relationship roots")
        for target in _iter_root_targets(root, matches, policy, through, path, max_depth):
            key = identity_key(target)
            if key not in seen_targets:
                seen_targets.add(key)
                yield target


def _iter_root_occurrences(
    root: RelationshipValue,
    node: RelationshipValue,
    matches: RelationshipMatcher,
    policy: EdgePolicy,
    through: frozenset[RelationshipKind],
    path_filter: RelationshipPath | GraphPath | None,
    active_nodes: frozenset[object],
    current_path: RelationshipPath,
    depth: int,
    max_depth: int | None,
    retain: Callable[[object], bool],
) -> Iterator[object]:
    from .identity import Occurrence

    for edge in _ordered_direct_edges(node):
        if not policy.permits(edge.kind):
            continue
        if max_depth is not None and depth >= max_depth:
            raise RelationshipDepthExceeded("Query V3 relationship depth budget exceeded.")
        next_path = current_path.child(edge.kind, edge.path)
        if not _path_prefix_matches(next_path, path_filter):
            continue
        if matches(edge.target) and _path_matches(next_path, path_filter) and _through_matches(next_path, through):
            occurrence = Occurrence(root, next_path, edge.target)
            if retain(occurrence):
                yield occurrence
        child_key = _node_key(edge.target)
        if child_key not in active_nodes:
            yield from _iter_root_occurrences(
                root, edge.target, matches, policy, through, path_filter,
                active_nodes | frozenset((child_key,)), next_path, depth + 1,
                max_depth, retain,
            )


def _retain_occurrence(occurrence, seen: set[object], limit: int | None) -> bool:
    if limit is not None and len(seen) >= limit:
        return False
    if occurrence.key in seen:
        return False
    seen.add(occurrence.key)
    return True


def _root_has_match(
    root: RelationshipValue,
    matches: RelationshipMatcher,
    policy: EdgePolicy,
    through: frozenset[RelationshipKind],
    path_filter: RelationshipPath | GraphPath | None,
    max_depth: int | None,
) -> bool:
    return any(_iter_root_targets(root, matches, policy, through, path_filter, max_depth))


def _iter_root_targets(
    root: RelationshipValue,
    matches: RelationshipMatcher,
    policy: EdgePolicy,
    through: frozenset[RelationshipKind],
    path_filter: RelationshipPath | GraphPath | None,
    max_depth: int | None,
) -> Iterator[RelationshipValue]:
    stack = [(root, RelationshipPath(), frozenset(), 0)]
    seen_states = set()
    while stack:
        node, current_path, seen_kinds, depth = stack.pop()
        state = (
            _node_key(node), tuple(sorted(kind.value for kind in seen_kinds)),
            _path_state(current_path, path_filter),
        )
        if state in seen_states:
            continue
        seen_states.add(state)
        for edge in reversed(_ordered_direct_edges(node)):
            if not policy.permits(edge.kind):
                continue
            if max_depth is not None and depth >= max_depth:
                raise RelationshipDepthExceeded("Query V3 relationship depth budget exceeded.")
            next_path = current_path.child(edge.kind, edge.path)
            if not _path_prefix_matches(next_path, path_filter):
                continue
            next_kinds = seen_kinds | frozenset((edge.kind,))
            if matches(edge.target) and _path_matches(next_path, path_filter) and through <= next_kinds:
                yield edge.target
            stack.append((edge.target, next_path, next_kinds, depth + 1))


def _ordered_direct_edges(value: RelationshipValue) -> tuple[DirectRelationship, ...]:
    return tuple(sorted(
        iter_direct_relationships(value),
        key=lambda edge: (edge.kind.value, graph_path_sort_key(edge.path)),
    ))


def _node_key(value: RelationshipValue) -> object:
    if isinstance(value, ConcreteDefinition):
        return "cdef", cdef_node_key(value)
    from .identity import identity_key

    return type(value).__name__, identity_key(value)


def _path_state(
    path: RelationshipPath, path_filter: RelationshipPath | GraphPath | None,
) -> object:
    if path_filter is None:
        return None
    return path.sort_key if isinstance(path_filter, RelationshipPath) else graph_path_sort_key(path.graph_path)


def _path_prefix_matches(
    path: RelationshipPath, path_filter: RelationshipPath | GraphPath | None,
) -> bool:
    if path_filter is None:
        return True
    if isinstance(path_filter, RelationshipPath):
        return path.hops == path_filter.hops[:len(path.hops)]
    field_path = path.graph_path
    return field_path.segments == path_filter.segments[:len(field_path)]


def _path_matches(
    path: RelationshipPath, path_filter: RelationshipPath | GraphPath | None,
) -> bool:
    if path_filter is None:
        return True
    return path == path_filter if isinstance(path_filter, RelationshipPath) else path.graph_path == path_filter


def _through_matches(path: RelationshipPath, through: frozenset[RelationshipKind]) -> bool:
    return through <= frozenset(hop.kind for hop in path.hops)


def _validate_traversal_inputs(matches, through, path, max_occurrences, max_depth) -> None:
    if not callable(matches):
        raise TypeError("Relationship traversal matcher must be callable.")
    if not isinstance(through, frozenset) or not all(
            isinstance(kind, RelationshipKind) for kind in through
    ):
        raise TypeError("Relationship traversal through kinds must be a frozenset of RelationshipKind values.")
    if path is not None and not isinstance(path, (RelationshipPath, GraphPath)):
        raise TypeError("Relationship traversal path must be a RelationshipPath or GraphPath.")
    if max_occurrences is not None and (type(max_occurrences) is not int or max_occurrences < 0):
        raise ValueError("max_occurrences must be a non-negative exact int or None.")
    _validate_max_depth(max_depth)


def _validate_max_depth(max_depth: int | None) -> None:
    if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
        raise ValueError("max_depth must be a non-negative exact int or None.")


def _require_relationship_value(value: object, label: str) -> None:
    if not isinstance(value, (ConcreteDefinition, ObjectRef, StateRef, LocalState)):
        raise TypeError(f"{label} must be ConcreteDefinition, ObjectRef, or StateRef values.")
