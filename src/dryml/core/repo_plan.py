from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import copy
import hashlib
import os
import re
from types import MappingProxyType
from threading import Lock
import uuid
from typing import Any, Callable, Generic, Iterable, Iterator, Literal, Mapping, TYPE_CHECKING, TypeAlias, TypeVar

from .canonical import NodeKind, is_runtime_leaf, node_kind
from .cdef_graph import ConcreteDefinitionGraph, EdgeKind
from .definition import ConcreteDefinition
from .object import Object, Serializable
from .policies import RepoGraphOptions
from .selector import Selector
from .store.store import Store, StoreCapabilityError
from .utils.graph.path import GraphPath, graph_path_sort_key
from .utils.graph.value import iter_value_edges
from .utils.graph.path import Parameter
from .cdef_identity import cdef_node_key

if TYPE_CHECKING:
    from .reference_values import ObjectId, StateRef


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SaveRouting:
    """Detached, immutable ordered destination policy for future Repo saves.

    Args:
        routes: Ordered ``(Selector, Store)`` bindings.  Store handles are
            retained, never opened or created, and must be connected to a Repo
            before that Repo installs this policy.
        match_mode: ``"first"`` selects the first matching binding and
            ``"all"`` selects every distinct matching destination.
        graph_mode: ``"per-object"`` or ``"closure"`` placement mode.

    Raises:
        TypeError: If fields have unsupported types or route entries are not
            Selector/Store pairs.
        ValueError: If a mode is not one of the supported closed values.

    Side Effects:
        Normalizes routes to an immutable tuple.  Constructing this detached
        policy never opens, creates, validates, or publishes to a Store.
    """

    routes: tuple[tuple[Selector, Store], ...] = ()
    match_mode: str = "first"
    graph_mode: str = "per-object"

    def __post_init__(self) -> None:
        """Normalize route pairs and validate closed routing modes."""

        if not isinstance(self.routes, (tuple, list)):
            raise TypeError("SaveRouting routes must be an ordered sequence of pairs.")
        routes = []
        for route in self.routes:
            if not isinstance(route, (tuple, list)) or len(route) != 2:
                raise TypeError("SaveRouting routes must contain (Selector, Store) pairs.")
            selector, store = route
            if not isinstance(selector, Selector):
                raise TypeError("SaveRouting route selectors must be Selectors.")
            if not isinstance(store, Store):
                raise TypeError("SaveRouting route destinations must be Stores.")
            routes.append((selector, store))
        if not isinstance(self.match_mode, str):
            raise TypeError("SaveRouting match_mode must be a string.")
        if self.match_mode not in {"first", "all"}:
            raise ValueError("SaveRouting match_mode must be 'first' or 'all'.")
        if not isinstance(self.graph_mode, str):
            raise TypeError("SaveRouting graph_mode must be a string.")
        if self.graph_mode not in {"per-object", "closure"}:
            raise ValueError("SaveRouting graph_mode must be 'per-object' or 'closure'.")
        object.__setattr__(self, "routes", tuple(routes))


@dataclass(frozen=True, slots=True)
class SaveRoutingContext:
    """One retained immutable Repo source and routing view for an active save.

    This internal value isolates source lookup and destination selection from
    subsequent Repo configuration changes.  The Repo owns its lifetime lease.

    Args:
        stores: Ordered, deduplicated connected Store handles.
        default_store: Store used when no route matches, or ``None``.
        routing: Normalized installed policy, or ``None`` for legacy closure.
        version: Monotonic Repo configuration version captured with the view.
    """

    stores: tuple[Store, ...]
    default_store: Store | None
    routing: SaveRouting | None
    version: int


@dataclass(frozen=True, slots=True)
class RealizationScope:
    """Opaque identifier and ephemeral labels for one live realization."""

    token: str
    workspace_label: str = field(
        default_factory=lambda: f"scope-{uuid.uuid4().hex}",
        repr=False,
    )
    _workspace_node_labels: dict[object, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
        hash=False,
    )
    _workspace_lock: Lock = field(
        default_factory=Lock,
        repr=False,
        compare=False,
        hash=False,
    )
    _claim_cleanups: list[Callable[[], Any]] = field(
        default_factory=list,
        repr=False,
        compare=False,
        hash=False,
    )

    def workspace_node_label(self, node_key: object) -> str:
        """Return one token-neutral label for a private node in this scope."""

        with self._workspace_lock:
            label = self._workspace_node_labels.get(node_key)
            if label is None:
                label = f"node-{uuid.uuid4().hex}"
                self._workspace_node_labels[node_key] = label
            return label

    def add_claim_cleanup(self, cleanup: Callable[[], Any]) -> None:
        """Register claim abandonment if this realization later fails.

        Args:
            cleanup: No-argument callback that abandons claims owned by this
                realization without touching successor generations.

        Side Effects:
            Retains the callback until the scope exits and invokes it in reverse
            order only when realization raises.
        """
        self._claim_cleanups.append(cleanup)


@dataclass(frozen=True, slots=True)
class AggregateMaterializationPlan:
    """Detached selected roots admitted together by :class:`dryml.core.repo.Repo`.

    Args:
        roots: Selected materializing boundary values in caller delivery order.
        cache: Effective cache policy retained for structural realization.
        reuse_live: Effective exact-state live-reuse policy.

    This plan deliberately carries no live Objects, payload bytes, or callable
    constructors.  Repo completes its authority preflight and claim admission
    before executing it, so a signature boundary cannot recreate restoration
    policy or publication behavior.
    """

    roots: tuple[Any, ...]
    cache: str
    reuse_live: str


_CURRENT_REALIZATION_SCOPE: ContextVar[RealizationScope | None] = ContextVar(
    "dryml_realization_scope", default=None
)


@contextmanager
def realization_scope() -> Iterator[RealizationScope]:
    """Install a new scope unless a nested realization already owns one."""

    existing = _CURRENT_REALIZATION_SCOPE.get()
    if existing is not None:
        yield existing
        return
    scope = RealizationScope(uuid.uuid4().hex)
    token = _CURRENT_REALIZATION_SCOPE.set(scope)
    try:
        yield scope
    except BaseException as error:
        for cleanup in reversed(scope._claim_cleanups):
            try:
                cleanup()
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        f"Realization claim cleanup also failed: {cleanup_error!r}"
                    )
        raise
    finally:
        _CURRENT_REALIZATION_SCOPE.reset(token)


def current_realization_scope() -> RealizationScope | None:
    """Return the active scope without creating a new realization."""

    return _CURRENT_REALIZATION_SCOPE.get()


class _NodeBindings(dict):
    """Map CDefs by private node token rather than structural equality."""

    @staticmethod
    def _key(key):
        return cdef_node_key(key) if isinstance(key, ConcreteDefinition) else key

    def __contains__(self, key):
        return super().__contains__(self._key(key))

    def __getitem__(self, key):
        return super().__getitem__(self._key(key))

    def get(self, key, default=None):
        return super().get(self._key(key), default)

    def __setitem__(self, key, value):
        super().__setitem__(self._key(key), value)


class RuntimeBindingConflict(KeyError):
    def __init__(self, *, definition: ConcreteDefinition, first: Object, second: Object, path: GraphPath):
        self.definition = definition
        self.first = first
        self.second = second
        self.path = path
        super().__init__(
            f"Repo already has a different object matching {definition} at {path.legacy_str()}!"
        )


@dataclass(frozen=True, slots=True)
class RuntimeRoot:
    definition: ConcreteDefinition
    path: GraphPath
    obj: Object | None = None


@dataclass(slots=True)
class RuntimeGraphBinding:
    graph: ConcreteDefinitionGraph
    roots: tuple[RuntimeRoot, ...]
    objects: _NodeBindings
    missing: frozenset[object] = frozenset()
    scope: RealizationScope | None = None


@dataclass(frozen=True, slots=True)
class GraphObjectOccurrence:
    path: GraphPath
    definition: ConcreteDefinition
    obj: Object


@dataclass(frozen=True, slots=True)
class GraphApplyResult(Generic[T]):
    path: GraphPath
    definition: ConcreteDefinition
    value: T


@dataclass(frozen=True, slots=True)
class SaveAction:
    """One owned Serializable node selected for graph-state publication."""

    path: GraphPath
    definition: ConcreteDefinition
    obj: Object
    state_hash: str | None = None
    source_store: Any = None


@dataclass(frozen=True, slots=True)
class SnapshotAction:
    """One exact materializing subtree selected for StateRef projection."""

    path: GraphPath
    definition: ConcreteDefinition
    obj: Object | None


@dataclass(slots=True)
class SavePlan:
    """Preflighted capture and exact projection evidence for one StateRef."""

    graph: ConcreteDefinitionGraph
    binding: RuntimeGraphBinding
    actions: tuple[SaveAction, ...]
    snapshots: tuple[SnapshotAction, ...]
    nodes: tuple[Object, ...]
    object_ref: object
    object_ids: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class RoutedSavePlan:
    """Retained routing enrichment for one route-neutral :class:`SavePlan`.

    ``SavePlan`` remains the sole source of graph bindings, object identity, and
    capture actions.  This value attaches only one retained context's immutable
    destination choices, so a configuration change cannot alter an active save.
    """

    plan: SavePlan
    context: SaveRoutingContext
    root_destinations: tuple[Store, ...]
    destinations: MappingProxyType
    graph_mode: str


# Publication boundary labels for SavePublication: definition/lineage/state/
# snapshot identify immutable authority, current-metadata phases identify their
# separate LWW records, membership promotes a stored root, claim records
# construction completion, alias/main update mutable names, index updates derived
# query state, and commit flushes a buffered backend transaction.
PublicationPhase: TypeAlias = Literal[
    "definition", "lineage", "state", "snapshot", "object_metadata",
    "state_metadata", "membership", "claim", "alias", "main", "index",
    "commit",
]
# Publication outcome labels for SavePublication: completed is read-back proof,
# failed is a negative read-back, unattempted has no attempted boundary, and
# uncertain means the checker could not establish authoritative outcome.
PublicationStatus: TypeAlias = Literal[
    "completed", "failed", "unattempted", "uncertain",
]
PUBLICATION_PHASES = frozenset({
    "definition", "lineage", "state", "snapshot", "object_metadata",
    "state_metadata", "membership", "claim", "alias", "main", "index",
    "commit",
})
PUBLICATION_STATUSES = frozenset({
    "completed", "failed", "unattempted", "uncertain",
})


@dataclass(frozen=True, slots=True)
class SavePublication:
    """One immutable observed boundary in a routed save work ledger.

    Args:
        store: Connected Store at the observed publication boundary.
        path: Canonical requested-root graph path, or ``None`` for Store-wide
            work such as commit.
        object_id: Exact ObjectId for stateful path work, or ``None`` when no
            ObjectId applies.
        state_ref: Exact snapshot identity when it is available, or ``None``.
        phase: Closed publication phase, including separate membership and claim
            boundaries.
        status: ``completed``, ``failed``, ``unattempted``, or ``uncertain``.

    ``completed`` means the named Store boundary has authoritative read-back
    evidence, even if its enclosing call later raises. ``failed`` has definitive
    negative evidence, ``unattempted`` was planned but never called, and
    ``uncertain`` lacks sufficient evidence either way. A completed boundary does
    not turn an interrupted or failed save call into success, nor imply a
    cross-Store transaction. This value carries no exception, payload,
    credential, or mutable Store state and remains useful after a partial save
    failure.
    """

    store: Store
    path: GraphPath | None
    object_id: ObjectId | None
    state_ref: StateRef | None
    phase: PublicationPhase
    status: PublicationStatus


@dataclass(frozen=True, slots=True)
class SavedSnapshot:
    """One immutable independently confirmed exact snapshot.

    Args:
        state_ref: Confirmed exact root or child StateRef.
        stores: Ordered snapshot-record destinations that confirmed this StateRef.
        required_stores: Ordered sufficient connected Store dependencies for
            exact recovery.

    Per-object snapshots can require external connected Stores. A closure-mode
    root destination instead contains a self-contained complete root closure.
    This live inspection record is neither portable configuration nor persistent
    authority.
    """

    state_ref: StateRef
    stores: tuple[Store, ...]
    required_stores: tuple[Store, ...]


@dataclass(frozen=True, slots=True)
class StoreReport:
    """Immutable save publication and exact-recovery inspection evidence.

    Args:
        target_stores: Ordered selected root snapshot Stores.
        state_stores: Confirmed local-state Stores by canonical graph path.
        required_stores: One deterministic sufficient connected recovery set.
        snapshots: Fully confirmed independently published root/child snapshots.
        publications: Every planned publication unit and its observed status.

    Constructor inputs are detached into tuples and an immutable mapping. A
    partial report is failure evidence, not a certificate that every listed
    snapshot remains recoverable; callers must inspect publication status and
    exact authority. Reports retain Store handles only for the lifetime chosen by
    their caller and never alter routing, Store ownership, or publication state.
    """

    target_stores: tuple[Store, ...]
    state_stores: Mapping[GraphPath, tuple[Store, ...]]
    required_stores: tuple[Store, ...]
    snapshots: tuple[SavedSnapshot, ...]
    publications: tuple[SavePublication, ...]

    def __init__(
            self, target_stores: Iterable[Store], state_stores: Mapping[GraphPath, Iterable[Store]],
            required_stores: Iterable[Store], snapshots: Iterable[SavedSnapshot] = (),
            publications: Iterable[SavePublication] = ()):
        object.__setattr__(self, "target_stores", _unique_stores(target_stores))
        object.__setattr__(
            self, "state_stores", MappingProxyType({
                path: _unique_stores(stores) for path, stores in state_stores.items()
            }),
        )
        object.__setattr__(self, "required_stores", _unique_stores(required_stores))
        object.__setattr__(self, "snapshots", tuple(snapshots))
        object.__setattr__(self, "publications", tuple(publications))


@dataclass(slots=True)
class _PublicationLedger:
    """Mutable save-local ledger which produces detached immutable reports."""

    target_stores: tuple[Store, ...]
    publications: list[SavePublication] = field(default_factory=list)

    def plan(self, store, path, obj, state_ref, phase: PublicationPhase, *, object_id=None) -> int:
        self.publications.append(SavePublication(
            store,
            path,
            getattr(obj, "object_id", None) if object_id is None else object_id,
            state_ref,
            phase,
            "unattempted",
        ))
        return len(self.publications) - 1

    def set_state_ref(self, state_ref) -> None:
        self.publications = [
            replace(
                item,
                state_ref=(
                    item.state_ref if item.state_ref is not None
                    else state_ref if not item.path else state_ref.at(item.path)
                ),
            )
            for item in self.publications
        ]

    def confirm(self, index: int, checker: Callable[[], bool]) -> bool:
        try:
            completed = checker()
        except Exception:
            self.publications[index] = replace(self.publications[index], status="uncertain")
            return False
        self.publications[index] = replace(
            self.publications[index], status="completed" if completed else "failed",
        )
        return completed

    def failed(self, index: int, checker: Callable[[], bool]) -> None:
        self.confirm(index, checker)

    def report(self, states, required, snapshots=()) -> StoreReport:
        return StoreReport(
            self.target_stores, states, required, snapshots, self.publications,
        )


def collect_runtime_roots(value: Any) -> tuple[RuntimeRoot, ...]:
    roots: list[RuntimeRoot] = []
    _collect_runtime_roots(value, GraphPath(), roots)
    return tuple(roots)


def build_runtime_binding(
        repo,
        value: Any) -> RuntimeGraphBinding:
    roots = collect_runtime_roots(value)
    graph = ConcreteDefinitionGraph.from_roots(root.definition for root in roots)
    materialize_nodes = _materialize_reachable_nodes(graph, roots)
    objects = _NodeBindings()
    for root in roots:
        if root.obj is not None:
            bind_runtime_object(objects, root.definition, root.obj, path=root.path)
    for node in graph.nodes():
        if node.definition not in materialize_nodes:
            continue
        if node.definition in objects:
            continue
        for root in roots:
            if root.obj is None:
                continue
            relative = graph.primary_path(root.definition, node.definition)
            if relative is None:
                continue
            # Bindings belong to the realized Object, so they are rooted at
            # that receiver rather than any outer input container occurrence.
            bound = getattr(root.obj, "_runtime_bindings", {}).get(relative)
            if isinstance(bound, Object):
                bind_runtime_object(
                    objects, node.definition, bound,
                    path=root.path.join(relative),
                )
                break
        if node.definition in objects:
            continue
        obj = _cached_object(repo, node.definition)
        if obj is not None:
            path = _node_primary_path(graph, roots, node.definition)
            bind_runtime_object(objects, node.definition, obj, path=path)
    missing = frozenset(
        cdef_node_key(cdef) for cdef in materialize_nodes if cdef not in objects
    )
    return RuntimeGraphBinding(
        graph=graph,
        roots=roots,
        objects=objects,
        missing=missing,
        scope=current_realization_scope(),
    )


def bind_runtime_object(
        bindings: _NodeBindings,
        cdef: ConcreteDefinition,
        obj: Object,
        *,
        path: GraphPath) -> None:
    existing = bindings.get(cdef)
    if existing is None:
        bindings[cdef] = obj
        return
    if existing is not obj:
        raise RuntimeBindingConflict(definition=cdef, first=existing, second=obj, path=path)


def attach_runtime_binding(
        repo, cdef: ConcreteDefinition, obj: Object, memo,
        runtime_parameters: dict[str, Any] | None = None) -> None:
    """Attach completed realization evidence to a successfully built Object.

    Args:
        repo: Repo that owns the completed realization.
        cdef: Exact private-node CDef for ``obj``.
        obj: Successfully initialized live Object.
        memo: Current private-node materialization memo containing dependencies.
        runtime_parameters: Optional bound runtime values supplied to the
            constructor. They are authoritative for materializing reference
            leaves and avoid decoding an exact StateRef a second time.

    Side Effects:
        Stores immutable construction bindings, defensive runtime projections,
        scope, ObjectId/ObjectRef metadata, and the current Store affinity on
        ``obj``. No user attributes are inspected.
    """

    from .canonical import from_canonical
    from .reference_values import ObjectId, ObjectRef

    bindings = _NodeBindings()
    bindings[cdef] = obj
    graph = ConcreteDefinitionGraph.from_root(cdef)
    for occurrence in graph.iter_occurrences(include_roots=True):
        candidate = memo.get(occurrence.definition)
        if candidate is not None:
            bindings[occurrence.definition] = candidate

    runtime_values: dict[GraphPath, Any] = {GraphPath(): obj}
    projection = runtime_parameters or from_canonical(
        cdef.parameters,
        repo=repo,
        resolve_cdef=lambda child: bindings[child],
    )
    for name, canonical_value in cdef.parameters.items():
        runtime_value = projection.get(name)
        if name not in projection:
            runtime_value = from_canonical(
                canonical_value, repo=repo,
                resolve_cdef=lambda child: bindings[child],
            )
        _record_runtime_values(
            canonical_value, runtime_value,
            GraphPath((Parameter(name),)),
            runtime_values,
        )
    for occurrence in graph.iter_occurrences(include_roots=True):
        bound = bindings.get(occurrence.definition)
        if bound is not None:
            runtime_values[occurrence.path] = bound
    _rebase_imported_runtime_projections(cdef, obj, GraphPath(), runtime_values)

    object_ids = {}
    for node in graph.nodes():
        if not getattr(node.definition, "_stateful_role", False):
            continue
        bound = bindings.get(node.definition)
        if bound is None:
            continue
        object_id = getattr(bound, "_object_id", None)
        if object_id is None:
            object_id = ObjectId()
            bound._object_id = object_id
            created_at = repo._record_lineage_candidate(object_id)
        else:
            created_at = repo._lineage_candidates.get(object_id)
        from .snapshot_capture import install_lineage_fact

        install_lineage_fact(bound, object_id, created_at)
        path = GraphPath() if node.definition is cdef else graph.primary_path(cdef, node.definition)
        object_ids[path] = object_id

    for name, value in cdef.parameters.items():
        _collect_imported_object_ids(value, GraphPath((Parameter(name),)), object_ids)

    obj._realization_scope = current_realization_scope()
    obj._runtime_bindings = runtime_values
    obj._runtime_projection = runtime_values
    obj._store_affinity = repo.obj_default_store.get(cdef)
    obj._last_state_hash = getattr(obj, "_last_state_hash", None)
    obj._object_id = getattr(obj, "_object_id", None) if isinstance(obj, Serializable) else None
    obj._object_ref = ObjectRef(cdef, object_ids)

    pending_claims = []
    claim_leases = []
    seen_objects = set()
    seen_claims = set()
    for candidate in runtime_values.values():
        if not isinstance(candidate, Object) or id(candidate) in seen_objects:
            continue
        seen_objects.add(id(candidate))
        pairs = list(getattr(candidate, "_pending_claim_dependencies", ()))
        lease = getattr(candidate, "_claim_lease", None)
        if lease is not None:
            pairs.append((lease, candidate))
        for dependency_lease, dependency_obj in pairs:
            key = id(dependency_lease)
            if key not in seen_claims:
                pending_claims.append((dependency_lease, dependency_obj))
                seen_claims.add(key)
        for dependency_lease in getattr(candidate, "_claim_leases", ()):
            if dependency_lease not in claim_leases:
                claim_leases.append(dependency_lease)
    if pending_claims:
        obj._pending_claim_dependencies = tuple(pending_claims)
        obj._claim_leases = tuple(claim_leases)


def _collect_imported_object_ids(value: Any, path: GraphPath, out: dict) -> None:
    """Expand immutable materializing reference IDs under an outer occurrence.

    Repeated materializing exact references retain one ObjectId entry at the
    minimum canonical path, matching ObjectRef's alias representation.
    """

    from .cdef_graph import EdgeKind
    from .links import DefLink
    from .reference_values import ObjectRef, StateRef

    if isinstance(value, StateRef):
        value = value.object
    if isinstance(value, ObjectRef):
        for child_path, object_id in value.objects.items():
            candidate_path = path.join(child_path)
            existing_path = next(
                (known_path for known_path, known_id in out.items()
                 if known_id == object_id),
                None,
            )
            if existing_path is None:
                out[candidate_path] = object_id
            elif graph_path_sort_key(candidate_path) < graph_path_sort_key(existing_path):
                del out[existing_path]
                out[candidate_path] = object_id
        return
    if isinstance(value, DefLink):
        if value.kind is EdgeKind.MATERIALIZE:
            _collect_imported_object_ids(value.target, path, out)
        return
    for edge in iter_value_edges(value):
        _collect_imported_object_ids(edge.value, path.child(edge.segment), out)


def apply_exact_reference_identity(obj: Object, reference, *, lineage_facts=None) -> None:
    """Rebind a completed materialized subtree to supplied exact ObjectIds.

    Args:
        obj: Freshly materialized root for ``reference.definition``.
        reference: ObjectRef whose topology and ObjectIds are authoritative.
        lineage_facts: Optional primary-path or ObjectId-keyed known/unknown
            creation facts. Omitted facts are explicitly restored as unknown.

    Raises:
        ValueError: If a reference path does not resolve to the expected live
            Object or the supplied topology cannot be retained.

    Side Effects:
        Replaces IDs and subtree ObjectRefs only after construction has
        completed. State restoration remains the later exact-load boundary.
    """

    from .metadata import LineageMetadata
    from .reference_values import ObjectRef
    from .snapshot_capture import install_lineage_fact

    if not isinstance(reference, ObjectRef):
        raise TypeError("Exact runtime materialization requires an ObjectRef.")
    if not obj.definition.graph_equal(reference.definition):
        raise ValueError("Materialized exact reference topology does not match its ObjectRef.")
    for path, object_id in reference.objects.items():
        bound = obj.graph_at(path)
        if not isinstance(bound, Object):
            raise ValueError(f"Exact ObjectRef path {path!s} did not resolve to an Object.")
        bound._object_id = object_id
        fact = None
        if lineage_facts is not None:
            if path in lineage_facts:
                fact = lineage_facts[path]
            elif object_id in lineage_facts:
                fact = lineage_facts[object_id]
        if isinstance(fact, LineageMetadata):
            fact = fact.created_at
        install_lineage_fact(bound, object_id, fact)
        try:
            bound._object_ref = reference.at(path)
        except ValueError:
            # An alias occurrence still names the same completed Object; its
            # primary ObjectRef remains attached by the corresponding path.
            pass
    obj._object_ref = reference
    obj._object_id = reference.object_id


def _record_runtime_values(canonical: Any, runtime: Any, path: GraphPath, out: dict[GraphPath, Any]) -> None:
    """Record runtime-form values without traversing into materialized Objects."""

    from .links import DefLink
    from .cdef_graph import EdgeKind

    out[path] = _copy_runtime_value(runtime)
    if isinstance(canonical, ConcreteDefinition):
        return
    if isinstance(canonical, DefLink):
        if canonical.kind is EdgeKind.REF:
            out[path] = canonical.target
        return
    canonical_edges = tuple(iter_value_edges(canonical))
    runtime_edges = {edge.segment: edge.value for edge in iter_value_edges(runtime)} if canonical_edges else {}
    for edge in canonical_edges:
        if edge.segment in runtime_edges:
            _record_runtime_values(edge.value, runtime_edges[edge.segment], path.child(edge.segment), out)


def _rebase_imported_runtime_projections(
        canonical: Any, runtime: Any, path: GraphPath, out: dict[GraphPath, Any]) -> None:
    """Expose materialized exact-reference bindings below their outer occurrence.

    Bare, materializing ``Mat(ObjectRef)``, and ``StateRef`` values construct a
    live subtree. Its private runtime projection remains authoritative, but its
    paths must also be visible from the enclosing root for exact graph saves.
    ``Ref`` links deliberately retain their immutable reference value and never
    contribute owned runtime bindings.
    """

    from .cdef_graph import EdgeKind
    from .links import DefLink
    from .reference_values import ObjectRef, StateRef

    if isinstance(canonical, (ObjectRef, StateRef)):
        if not isinstance(runtime, Object):
            raise ValueError(f"Materialized exact reference at {path!s} did not produce an Object.")
        for inner_path, value in runtime._runtime_projection.items():
            target = path.join(inner_path)
            previous = out.get(target)
            if previous is not None and previous is not value:
                raise ValueError(f"Conflicting exact runtime binding at {target!s}.")
            out[target] = value
        return
    if isinstance(canonical, ConcreteDefinition):
        for edge in iter_value_edges(canonical):
            child_path = path.child(edge.segment)
            runtime_child = out.get(child_path)
            if runtime_child is not None:
                _rebase_imported_runtime_projections(
                    edge.value, runtime_child, child_path, out
                )
        return
    if isinstance(canonical, DefLink):
        if canonical.kind is EdgeKind.MATERIALIZE:
            _rebase_imported_runtime_projections(canonical.target, runtime, path, out)
        return
    canonical_edges = tuple(iter_value_edges(canonical))
    runtime_edges = {edge.segment: edge.value for edge in iter_value_edges(runtime)} if canonical_edges else {}
    for edge in canonical_edges:
        if edge.segment in runtime_edges:
            _rebase_imported_runtime_projections(edge.value, runtime_edges[edge.segment], path.child(edge.segment), out)


def _copy_runtime_value(value: Any) -> Any:
    """Copy mutable runtime data while retaining Object and exact-reference identity."""

    if isinstance(value, Object):
        return value
    if isinstance(value, dict):
        return {key: _copy_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_runtime_value(item) for item in value)
    if isinstance(value, set):
        return {_copy_runtime_value(item) for item in value}
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def iter_graph_objects(repo, root: Any, options: RepoGraphOptions) -> Iterator[Object]:
    _validate_graph_options(options)
    binding = build_runtime_binding(repo, root)
    for occurrence in _iter_bound_graph_object_occurrences(repo, binding, options):
        yield occurrence.obj


def _iter_bound_graph_objects(
        repo,
        binding: RuntimeGraphBinding,
        options: RepoGraphOptions) -> tuple[Object, ...]:
    return tuple(
        occurrence.obj
        for occurrence in _iter_bound_graph_object_occurrences(
            repo,
            binding,
            options,
        )
    )


def _iter_bound_graph_object_occurrences(
        repo,
        binding: RuntimeGraphBinding,
        options: RepoGraphOptions) -> tuple[GraphObjectOccurrence, ...]:
    out: list[GraphObjectOccurrence] = []
    load_memo = _NodeBindings()
    seen: set[object] = set()

    def visit(cdef: ConcreteDefinition, path: GraphPath, explicit_obj: Object | None = None) -> None:
        obj = explicit_obj if explicit_obj is not None else binding.objects.get(cdef)
        if obj is None:
            obj = _resolve_missing_for_traversal(repo, cdef, path, options, load_memo)
            if obj is None:
                return
            bind_runtime_object(binding.objects, cdef, obj, path=path)

        if options.dedupe:
            if cdef_node_key(cdef) in seen:
                return
            seen.add(cdef_node_key(cdef))

        should_apply = options.include_root or bool(path)
        if options.order == "pre" and should_apply:
            out.append(GraphObjectOccurrence(path, cdef, obj))

        for edge in binding.graph.outgoing(cdef):
            if edge.kind is not EdgeKind.MATERIALIZE:
                continue
            visit(edge.child, path.join(edge.path))

        if options.order == "post" and should_apply:
            out.append(GraphObjectOccurrence(path, cdef, obj))

    for root_occ in binding.roots:
        visit(root_occ.definition, root_occ.path, root_occ.obj)
    return tuple(out)


def apply_graph_objects(
        repo,
        root: Any,
        func: Callable[[Object], T],
        options: RepoGraphOptions) -> dict[ConcreteDefinition, T] | tuple[GraphApplyResult[T], ...]:
    _validate_graph_options(options)
    binding = build_runtime_binding(repo, root)
    occurrences = _iter_bound_graph_object_occurrences(repo, binding, options)
    if not options.dedupe:
        return tuple(
            GraphApplyResult(occ.path, occ.definition, func(occ.obj))
            for occ in occurrences
        )
    results: dict[ConcreteDefinition, T] = {}
    for occ in occurrences:
        results[occ.definition] = func(occ.obj)
    return results


def add_objects(repo, values: Iterable[Any], *, store=None) -> None:
    for value in values:
        binding = build_runtime_binding(repo, value)
        for obj in _iter_bound_graph_objects(
                repo,
                binding,
                RepoGraphOptions(include_root=True, order="post", missing="raise", dedupe=False)):
            _add_object_single(repo, obj, store=store)
        for missing in binding.missing:
            if _cached_object(repo, missing) is None:
                raise KeyError(f"No object linked to definition {missing} found in repo!")


def build_save_plan(
        repo,
        value: Object) -> SavePlan:
    """Build complete owned-state save evidence from retained bindings.

    Args:
        repo: Repository owning live bindings.
        value: Live root Object to publish.
    Returns:
        A graph plan with each unique owned Serializable node exactly once.

    Raises:
        RepoSaveError: If retained bindings cannot prove a total ObjectRef map.
    """
    if not isinstance(value, Object):
        raise TypeError("Graph save requires one live Object root.")
    if getattr(value, "_restore_failed", False):
        from .repo import RepoSaveError

        raise RepoSaveError("Cannot save an invalidated restore target.")
    binding = build_runtime_binding(repo, value)
    from .repo import RepoSaveError

    if binding.missing:
        missing = next(iter(binding.missing))
        raise RepoSaveError(f"Definition of object {missing} is not reachable in this repo!")
    reference = getattr(value, "_object_ref", None)
    if reference is None or not reference.definition.graph_equal(value.definition):
        raise RepoSaveError("Live root lacks complete retained ObjectRef evidence.")
    actions: list[SaveAction] = []
    seeds = _seed_state_refs(value.definition)
    for path in reference.objects:
        try:
            obj = value.graph_at(path)
        except Exception:
            obj = None
        if isinstance(obj, Serializable):
            if getattr(obj, "_restore_failed", False):
                raise RepoSaveError(
                    f"Cannot save invalidated restore target at {path!s}."
                )
            actions.append(SaveAction(path, obj.definition, obj))
            continue
        seed = seeds.get(path)
        if seed is not None:
            definition, state_hash = seed
            actions.append(SaveAction(path, definition, value, state_hash=state_hash))
            continue
        raise RepoSaveError(f"No retained Serializable binding or exact seed state at {path!s}.")
    snapshots: list[SnapshotAction] = []
    nodes: list[Object] = []
    seen_nodes: set[int] = set()
    seen_snapshot_nodes: set[int] = set()
    seen_paths: set[GraphPath] = set()

    def add_snapshot(path: GraphPath, definition: ConcreteDefinition, obj: Object | None) -> None:
        if path in seen_paths or (obj is not None and id(obj) in seen_snapshot_nodes):
            return
        seen_paths.add(path)
        snapshots.append(SnapshotAction(path, definition, obj))
        if obj is not None:
            seen_snapshot_nodes.add(id(obj))
            if id(obj) not in seen_nodes:
                seen_nodes.add(id(obj))
                nodes.append(obj)

    projections = getattr(value, "_runtime_projection", {})
    for path in sorted(projections, key=graph_path_sort_key):
        candidate = projections[path]
        if isinstance(candidate, Object):
            add_snapshot(path, candidate.definition, candidate)
    add_snapshot(GraphPath(), value.definition, value)
    for action in actions:
        # An exact imported seed may have no live payload binding, but its
        # materializing subtree still needs its selected StateRef projection.
        add_snapshot(action.path, action.definition, None)
    return SavePlan(
        graph=binding.graph,
        binding=binding,
        actions=tuple(actions),
        snapshots=tuple(snapshots),
        nodes=tuple(nodes),
        object_ref=reference,
        object_ids=tuple(reference.objects.values()),
    )


def validate_retained_save_plan(plan: SavePlan, value: Object) -> None:
    """Verify admitted save evidence still names the same usable live graph.

    This is intentionally a read-through validation of the already captured
    paths. It neither traverses a new graph nor selects routes, so a reservation
    cannot become authorization for changed or invalidated bindings.
    """

    from .repo import RepoSaveError

    if plan.binding.roots[0].obj is not value or value.object_ref != plan.object_ref:
        raise RepoSaveError("State graph reservation does not cover this exact live graph.")
    if any(getattr(node, "_restore_failed", False) for node in plan.nodes):
        raise RepoSaveError("State graph contains an invalidated restore target.")
    for snapshot in plan.snapshots:
        if snapshot.obj is None:
            continue
        try:
            current = value.graph_at(snapshot.path)
        except Exception as error:
            raise RepoSaveError("State graph has lost a retained runtime binding.") from error
        if current is not snapshot.obj:
            raise RepoSaveError("State graph has changed a retained runtime binding.")


def _register_retained_save_plan(repo, plan: SavePlan) -> None:
    """Register every retained live save node without recapturing its graph.

    The pre-routing registration preserves the ordinary ``add_objects()`` cache
    and default-affinity behavior for successful or partially published saves.
    It deliberately uses no routed destination: route selection remains owned by
    the later routed save plan, while the retained graph records cached query
    structure exactly once.
    """

    if not isinstance(plan, SavePlan):
        raise TypeError("Retained save registration requires a SavePlan.")
    for node in plan.nodes:
        _add_object_single(repo, node)
    repo._query_catalog.register_graph(plan.graph)


def build_routed_save_plan(repo, plan: SavePlan, context, *, store=None) -> RoutedSavePlan:
    """Attach retained destinations to route-neutral live graph evidence.

    Args:
        repo: Repo that owns ``plan`` and the retained ``context``.
        plan: Previously built graph/binding/ObjectId evidence.
        context: Active immutable SaveRoutingContext for this save.
        store: Explicit whole-graph destination, if one was supplied.

    Returns:
        An immutable per-snapshot destination map. It never captures state or
        writes records. Selector matching can invoke trusted user predicates.

    Raises:
        RepoSaveError: If no effective destination can be selected.
    """
    if not isinstance(plan, SavePlan):
        raise TypeError("Routed save planning requires a SavePlan.")
    if not isinstance(context, SaveRoutingContext):
        raise TypeError("Routed save planning requires a SaveRoutingContext.")
    root = plan.binding.roots[0].obj
    if store is not None:
        root_destinations = (store,)
        graph_mode = "closure"
    else:
        root_destinations = repo._select_save_destinations(context, root)
        graph_mode = "closure" if context.routing is None else context.routing.graph_mode
    destinations = {}
    for action in plan.snapshots:
        if graph_mode == "closure":
            selected = root_destinations
        else:
            selected = repo._select_save_destinations(context, action.definition)
        destinations[action.path] = tuple(selected)
    return RoutedSavePlan(
        plan, context, tuple(root_destinations), MappingProxyType(destinations), graph_mode,
    )


_CODEC_RE = re.compile(r"^[A-Za-z0-9]{1,32}$")


class _SourceSelectionError(ValueError):
    """Reject an invalid source selector before any destination publication."""


def _validate_codecs(actions: Iterable[SaveAction]) -> None:
    """Validate every selected developer codec before any serializer runs."""
    for action in actions:
        if action.state_hash is not None:
            continue
        codec = getattr(action.obj, "state_codec", None)
        if not isinstance(codec, str) or not _CODEC_RE.fullmatch(codec):
            raise _save_error(action.path, "state_codec must match [A-Za-z0-9]{1,32}")


def _manifest_files(data_dir: str) -> tuple[tuple[str, int, str], ...]:
    """Return the exhaustive regular payload manifest for a staging data tree."""
    files: list[tuple[str, int, str]] = []
    for root, directories, names in os.walk(data_dir, followlinks=False):
        for directory in directories:
            path = os.path.join(root, directory)
            if os.path.islink(path) or not os.path.isdir(path):
                raise ValueError(f"unsupported payload directory entry {path!r}")
        for name in names:
            path = os.path.join(root, name)
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"unsupported payload file entry {path!r}")
            digest = hashlib.sha256()
            with open(path, "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            relative = os.path.relpath(path, data_dir).replace(os.sep, "/")
            files.append((relative, os.path.getsize(path), digest.hexdigest()))
    return tuple(sorted(files))


def _seed_state_refs(definition: ConcreteDefinition) -> dict[GraphPath, tuple[ConcreteDefinition, str]]:
    """Expand embedded materializing exact StateRefs into enclosing state paths."""
    result: dict[GraphPath, tuple[ConcreteDefinition, str]] = {}
    for outer, reference in _embedded_state_refs(definition):
        for path, state_hash in reference.states.items():
            result[outer.join(path)] = (reference.object.at(path).definition, state_hash)
    return result


def _embedded_state_refs(definition: ConcreteDefinition):
    """Yield materializing StateRefs with their paths in one enclosing CDef."""
    from .cdef_graph import EdgeKind
    from .links import DefLink
    from .reference_values import StateRef

    result = []
    visited: set[object] = set()

    def visit_value(value: Any, path: GraphPath) -> None:
        if isinstance(value, StateRef):
            result.append((path, value))
            return
        if isinstance(value, ConcreteDefinition):
            visit_cdef(value, path)
            return
        if isinstance(value, DefLink):
            if value.kind is EdgeKind.MATERIALIZE:
                visit_value(value.target, path)
            return
        for edge in iter_value_edges(value):
            visit_value(edge.value, path.child(edge.segment))

    def visit_cdef(cdef: ConcreteDefinition, path: GraphPath) -> None:
        key = cdef_node_key(cdef)
        if key in visited:
            return
        visited.add(key)
        for edge in iter_value_edges(cdef):
            visit_value(edge.value, path.child(edge.segment))

    visit_cdef(definition, GraphPath())
    return tuple(result)


def _unique_stores(stores: Iterable[Any]) -> tuple[Any, ...]:
    """Deduplicate Store objects by identity without relying on their equality."""
    result = []
    seen = set()
    for store in stores:
        if store is None or id(store) in seen:
            continue
        seen.add(id(store))
        result.append(store)
    return tuple(result)


def _has_completed_claim(store, lease, state_ref) -> bool:
    """Return whether an initial claim survived as this exact completed receipt."""

    claim = store.read_claim_record(lease.object_ref.digest())
    return (
        claim is not None and claim.generation == lease.generation
        and claim.status == "completed" and claim.state_ref_digest == state_ref.digest()
    )


def _has_complete_local_states(store, state_ref, sources) -> bool:
    """Return whether an exact snapshot retains every selected local payload."""

    record = store.read_state_ref_record(state_ref.digest())
    if record is None or record.state_ref != state_ref:
        return False
    return all(
        store.validate_local_state(state_ref, path).state_hash
        == source.manifest.state_hash
        for path, source in sources.items()
    )


def _supports_annotation_phase_callback(store) -> bool:
    """Return whether this Store override implements exact annotation callbacks."""

    return bool(getattr(
        type(store).publish_snapshot,
        "_dryml_annotation_phase_callback",
        False,
    ))


def _prepare_snapshot_local_state(obj: Object, definition: ConcreteDefinition, store: Store, path: GraphPath):
    """Serialize one payload into owned staging without publishing shared authority."""

    from .store.records import DefinitionRecord, LocalStateManifest

    stage = store.create_local_state_staging()
    reservation = getattr(obj, "_save_load_reservation", None)
    if reservation is None or not reservation.acquire(blocking=False):
        store.discard_local_state_staging(stage)
        raise _save_error(path, "local state is already reserved by save or restore")
    try:
        data_dir = os.path.join(os.fspath(stage), "data")
        obj.save_state_to_dir(data_dir, codec=obj.state_codec)
        record = DefinitionRecord(definition)
        definition_bytes = record.to_bytes()
        Path = __import__("pathlib").Path
        Path(stage, "def.pkl").write_bytes(definition_bytes)
        manifest = LocalStateManifest(
            obj.state_codec, record.graph_hash, record.digest,
            hashlib.sha256(definition_bytes).hexdigest(), _manifest_files(data_dir),
        )
        Path(stage, "manifest.record").write_bytes(manifest.to_bytes())
        source = store.prepare_local_state(stage, manifest)
        obj._last_state_hash = manifest.state_hash
        return source
    except BaseException as error:
        store.discard_local_state_staging(stage)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise _save_error(path, "snapshot-local state preparation failed", error) from error
    finally:
        reservation.release()


def execute_routed_save_plan(
        repo, routed: RoutedSavePlan, *, deep_capture: bool = False,
        report_stores: bool = False, late_publications=(), snapshot_observer=None,
        annotations=None, source_store=None, source_stores=None):
    """Publish one routed graph through complete v3 snapshot directories.

    Payload bytes remain private staging until ``Store.publish_snapshot`` atomically
    installs the StateRef, placement, captured metadata, and local payload tree.
    The ledger deliberately records snapshot-local payload completion only after
    the directory read-back succeeds.
    """

    from .metadata import LineageMetadata, SaveAnnotations, SnapshotCapture
    from .reference_values import StateRef
    from .repo import MetadataConflictError, RepoSaveError
    from .snapshot_capture import capture_snapshot
    from .store.records import DefinitionRecord

    plan, context = routed.plan, routed.context
    if annotations is not None and not isinstance(annotations, SaveAnnotations):
        raise TypeError("annotations must be a SaveAnnotations or None.")
    if source_store is not None:
        if not isinstance(source_store, Store) or not any(
                candidate is source_store for candidate in context.stores):
            raise ValueError("source_store must be an already-connected Store handle.")
    if source_stores is not None and not isinstance(source_stores, Mapping):
        raise TypeError("source_stores must be a mapping or None.")
    if source_stores is not None:
        for reference, selected in source_stores.items():
            if not isinstance(reference, StateRef) or not isinstance(selected, Store):
                raise TypeError("source_stores must map StateRefs to connected Store handles.")
            if not any(candidate is selected for candidate in context.stores):
                raise ValueError("source_stores must contain only connected Store handles.")
            candidate = selected.read_snapshot_metadata(reference.digest())
            if candidate is None or candidate.state_ref != reference:
                raise KeyError(reference.digest())
    _validate_codecs(plan.actions)
    destinations = _unique_stores(
        store for values in routed.destinations.values() for store in values
    )
    for store in destinations:
        store.preflight_publication("routed snapshot save", local_state=True)
        if (
                any(candidate is store for candidate in routed.root_destinations)
                and annotations is not None
                and (annotations.object is not None or annotations.state is not None)
                and not _supports_annotation_phase_callback(store)):
            raise StoreCapabilityError(
                "Save-time annotations require exact publication phase callbacks."
            )
    claims = repo._preflight_routed_claims(plan, routed)
    ledger = _PublicationLedger(routed.root_destinations)
    source_by_path = {}
    state_stores = {action.path: [] for action in plan.actions}
    snapshots = []
    state_ref = None
    claims_by_path = {action.path: lease for lease, action in claims}
    root_action = next(action for action in plan.snapshots if not action.path)
    snapshot_actions = plan.snapshots if routed.graph_mode == "per-object" else (root_action,)
    publication_indexes = {}
    # Freeze every possible boundary before serializer hooks run.  StateRef
    # projections do not exist until capture completes, so the ledger fills them
    # in atomically below once the exact root receipt is known.
    for action in snapshot_actions:
        for store in routed.destinations[action.path]:
            indexes = {
                phase: ledger.plan(store, action.path, action.obj, None, phase)
                for phase in ("definition", "state", "snapshot", "membership")
            }
            projection_object = (
                plan.object_ref if not action.path else plan.object_ref.at(action.path)
            )
            indexes["lineage"] = {
                path: ledger.plan(
                    store, action.path.join(path), action.obj, None, "lineage",
                    object_id=(projection_object if not path else projection_object.at(path)).object_id,
                )
                for path in dict.fromkeys((GraphPath(), *projection_object.objects))
            }
            if not action.path and annotations is not None:
                if annotations.object is not None:
                    indexes["object_metadata"] = ledger.plan(
                        store, action.path, action.obj, None, "object_metadata",
                    )
                if annotations.state is not None:
                    indexes["state_metadata"] = ledger.plan(
                        store, action.path, action.obj, None, "state_metadata",
                    )
            lease = claims_by_path.get(action.path)
            if lease is not None and store is lease.store:
                indexes["claim"] = ledger.plan(store, action.path, action.obj, None, "claim")
            publication_indexes[id(store), action.path] = indexes
    for phase, stores in late_publications:
        if phase not in PUBLICATION_PHASES:
            raise ValueError(f"Unsupported late publication phase: {phase!r}.")
        for store in stores:
            ledger.plan(store, None, None, None, phase)
    seed_sources = {}
    embedded_snapshots = {}
    for outer, embedded in _embedded_state_refs(plan.binding.roots[0].definition):
        selected_embedded_source = (
            source_stores.get(embedded) if source_stores is not None else None
        )
        embedded_candidates = (
            (selected_embedded_source,) if selected_embedded_source is not None
            else tuple(
                candidate for candidate in (source_store, *context.stores)
                if candidate is not None
            )
        )
        matches = []
        for candidate in embedded_candidates:
            try:
                sources = {
                    path: candidate.open_local_state(embedded, path)
                    for path in embedded.states
                }
                metadata = candidate.read_snapshot_metadata(embedded.digest())
            except Exception:
                continue
            if metadata is not None:
                matches.append((sources, metadata))
        if matches:
            snapshot_sources, snapshot_metadata = matches[0]
            if any(metadata != snapshot_metadata for _, metadata in matches[1:]):
                raise MetadataConflictError(
                    "Connected Stores disagree about embedded snapshot evidence."
                )
            for path, source in snapshot_sources.items():
                seed_sources[outer.join(path)] = source
            embedded_snapshots[embedded.digest()] = (
                embedded, snapshot_sources, snapshot_metadata,
            )
    try:
        for action in plan.actions:
            if action.state_hash is not None:
                source = seed_sources.get(action.path)
                if source is None or source.manifest.state_hash != action.state_hash:
                    raise _save_error(action.path, "embedded StateRef lacks snapshot-local source authority")
                source_by_path[action.path] = source
            else:
                source_by_path[action.path] = _prepare_snapshot_local_state(
                    action.obj, action.definition, routed.destinations[action.path][0], action.path,
                )
        state_ref = StateRef(plan.object_ref, {
            path: source.manifest.state_hash for path, source in source_by_path.items()
        })
        ledger.set_state_ref(state_ref)
        if source_stores is not None:
            expected_sources = {
                state_ref,
                *(embedded for _, embedded in _embedded_state_refs(
                    plan.binding.roots[0].definition
                )),
            }
            if any(reference not in expected_sources for reference in source_stores):
                raise _SourceSelectionError(
                    "source_stores contains a StateRef outside the saved snapshot closure."
                )
        # Existing complete evidence wins before any environment observation. A
        # source selection is authority input, independent of destination routes.
        selected_sources = {}
        if source_stores is not None:
            for reference, selected in source_stores.items():
                candidate = selected.read_snapshot_metadata(reference.digest())
                selected_sources[reference] = selected
        mapped_root = selected_sources.get(state_ref)
        if source_store is not None and mapped_root is not None and mapped_root is not source_store:
            raise ValueError("source_store and source_stores select different root Stores.")
        selected_root = source_store or mapped_root
        evidence = None
        with repo._authority_read_fences(context.stores):
            if selected_root is not None:
                evidence = selected_root.read_snapshot_metadata(state_ref.digest())
                if evidence is None or evidence.state_ref != state_ref:
                    raise KeyError(state_ref.digest())
            else:
                for store in context.stores:
                    candidate = store.read_snapshot_metadata(state_ref.digest())
                    if candidate is None:
                        continue
                    if candidate.state_ref != state_ref:
                        raise RepoSaveError("StateRef digest collision has incompatible snapshot authority.")
                    if evidence is not None and evidence != candidate:
                        raise MetadataConflictError("Connected Stores disagree about snapshot evidence.")
                    evidence = candidate
        if evidence is None:
            evidence = capture_snapshot(plan, observer=snapshot_observer)

        # A parent association can only name already complete child snapshots.
        snapshot_actions = tuple(sorted(
            snapshot_actions, key=lambda item: len(item.path), reverse=True,
        ))
        if routed.graph_mode == "closure":
            # A closure target cannot retain an uncommitted historical Store as
            # the only authority for an embedded exact seed. Copy each complete
            # seed snapshot before publishing the enclosing root snapshot.
            for embedded, sources, metadata in embedded_snapshots.values():
                records = tuple(
                    DefinitionRecord(node.definition)
                    for node in ConcreteDefinitionGraph.from_root(embedded.definition).nodes()
                )
                for store in routed.root_destinations:
                    for record in records:
                        store.write_definition_record(record, stored_root=False)
                    store.publish_snapshot(
                        embedded, evidence=metadata, local_states=sources,
                    )
        for action in snapshot_actions:
            if routed.graph_mode == "per-object":
                # Each snapshot owns only its direct state. Descendant state is
                # supplied by exact child projections, never a shared pool.
                projection = state_ref if not action.path else state_ref.at(action.path)
                selected_sources = ({GraphPath(): source_by_path[action.path]}
                                    if action.path in source_by_path else {})
                child_actions = [
                    candidate for candidate in plan.snapshots
                    if candidate.path.startswith(action.path) and candidate.path != action.path
                    and not any(
                        other.path != candidate.path
                        and other.path != action.path
                        and candidate.path.startswith(other.path)
                        and other.path.startswith(action.path)
                        for other in plan.snapshots
                    )
                ]
                children = {
                    candidate.path.relative_to(action.path): state_ref.at(candidate.path)
                    for candidate in child_actions
                }
                lineage = {}
                for path in (*projection.object.objects, GraphPath()):
                    expected = projection.object if not path else projection.object.at(path)
                    captured = evidence.lineages.get(action.path.join(path))
                    lineage[path] = (
                        captured if captured is not None and captured.object_ref == expected
                        else LineageMetadata(expected, "unknown", None)
                    )
                snapshot_evidence = SnapshotCapture(
                    lineage, evidence.saved_at, evidence.environment, evidence.environment_status,
                    evidence.requirements, evidence.requirements_status,
                    evidence.requirements_coverage, evidence.diagnostics,
                ) if isinstance(evidence, SnapshotCapture) else evidence
            else:
                projection, selected_sources, snapshot_evidence, children = state_ref, source_by_path, evidence, {}
            for store in routed.destinations[action.path]:
                indexes = publication_indexes[id(store), action.path]
                definition_index = indexes["definition"]
                state_index = indexes["state"]
                snapshot_index = indexes["snapshot"]
                membership_index = indexes["membership"]
                lineage_indexes = indexes["lineage"]
                object_metadata_index = indexes.get("object_metadata")
                state_metadata_index = indexes.get("state_metadata")
                claim_index = indexes.get("claim")
                lease = claims_by_path.get(action.path)
                record = DefinitionRecord(projection.definition)
                attempted = set()
                definition_records = tuple(
                    DefinitionRecord(node.definition)
                    for node in ConcreteDefinitionGraph.from_root(projection.definition).nodes()
                )
                try:
                    attempted.add(definition_index)
                    for definition_record in definition_records:
                        store.write_definition_record(definition_record, stored_root=False)
                    if not ledger.confirm(
                            definition_index,
                            lambda store=store, records=definition_records: all(
                                store.read_definition_record(item.digest) is not None
                                for item in records
                            ),
                    ):
                        raise RepoSaveError("Definition publication did not survive read-back.")
                    if isinstance(snapshot_evidence, SnapshotCapture):
                        resolved_lineages = {
                            path: (
                                existing
                                if lineage.creation_status == "unknown"
                                and (existing := store.read_lineage_metadata(lineage.object_ref)) is not None
                                else lineage
                            )
                            for path, lineage in snapshot_evidence.lineages.items()
                        }
                        snapshot_evidence = SnapshotCapture(
                            resolved_lineages, snapshot_evidence.saved_at,
                            snapshot_evidence.environment, snapshot_evidence.environment_status,
                            snapshot_evidence.requirements, snapshot_evidence.requirements_status,
                            snapshot_evidence.requirements_coverage, snapshot_evidence.diagnostics,
                        )
                    for path, lineage in snapshot_evidence.lineages.items():
                        lineage_index = lineage_indexes[path]
                        attempted.add(lineage_index)
                        store.write_lineage_metadata(lineage)
                        if not ledger.confirm(
                            lineage_index,
                            lambda store=store, lineage=lineage:
                            store.read_lineage_metadata(lineage.object_ref) == lineage,
                        ):
                            raise RepoSaveError("Lineage publication did not survive read-back.")
                    attempted.update((state_index, snapshot_index))

                    def begin_annotation_write(scope):
                        attempted.add(
                            object_metadata_index if scope == "object"
                            else state_metadata_index
                        )

                    publication_arguments = {
                        "evidence": snapshot_evidence,
                        "annotations": annotations if not action.path else None,
                        "local_states": selected_sources,
                        "children": children,
                    }
                    if _supports_annotation_phase_callback(store):
                        publication_arguments["_before_annotation_write"] = begin_annotation_write
                    published_metadata = store.publish_snapshot(
                        projection, **publication_arguments,
                    )
                    if annotations is not None and not action.path:
                        if annotations.object is not None:
                            if not ledger.confirm(
                                    object_metadata_index,
                                    lambda store=store, projection=projection, annotations=annotations:
                                    store.read_metadata(projection.object) == annotations.object,
                            ):
                                raise RepoSaveError("Object metadata publication did not survive read-back.")
                        if annotations.state is not None:
                            if not ledger.confirm(
                                    state_metadata_index,
                                    lambda store=store, projection=projection, annotations=annotations:
                                    store.read_metadata(projection) == annotations.state,
                            ):
                                raise RepoSaveError("State metadata publication did not survive read-back.")
                    # The first root installation is the capture winner for later
                    # replicas. They copy its immutable captured mappings instead
                    # of rereading their own current annotations.
                    if not action.path:
                        evidence = published_metadata
                        snapshot_evidence = published_metadata
                    if not ledger.confirm(
                            state_index,
                            lambda store=store, projection=projection, selected_sources=selected_sources:
                            _has_complete_local_states(store, projection, selected_sources),
                    ):
                        raise RepoSaveError("Snapshot payload publication did not survive read-back.")
                    if not ledger.confirm(
                            snapshot_index,
                            lambda store=store, projection=projection: store.read_state_ref_record(projection.digest()) is not None,
                        ):
                            raise RepoSaveError("Snapshot publication did not survive read-back.")
                    attempted.add(membership_index)
                    store.write_definition_record(record, stored_root=True)
                    if not ledger.confirm(
                            membership_index,
                            lambda store=store, record=record: store.read_stored_root_record(record.digest) is not None,
                    ):
                        raise RepoSaveError("Snapshot membership did not survive read-back.")
                    if lease is not None and store is lease.store:
                        attempted.add(claim_index)
                        repo._mark_initial_state_ref_complete(projection, store, lease)
                        if not ledger.confirm(
                                claim_index,
                                lambda store=store, lease=lease, projection=projection:
                                _has_completed_claim(store, lease, projection),
                        ):
                            raise RepoSaveError("Initial snapshot claim did not survive read-back.")
                        repo._clear_completed_routed_claim(plan, lease)
                except BaseException as error:
                    if definition_index in attempted:
                        ledger.failed(
                            definition_index,
                            lambda store=store, records=definition_records: all(
                                store.read_definition_record(item.digest) is not None
                                for item in records
                            ),
                        )
                    for path, lineage_index in lineage_indexes.items():
                        if lineage_index in attempted:
                            lineage = snapshot_evidence.lineages[path]
                            ledger.failed(
                                lineage_index,
                                lambda store=store, lineage=lineage:
                                store.read_lineage_metadata(lineage.object_ref) == lineage,
                            )
                    if state_index in attempted:
                        ledger.failed(
                            state_index,
                            lambda store=store, projection=projection, selected_sources=selected_sources:
                            _has_complete_local_states(store, projection, selected_sources),
                        )
                    if snapshot_index in attempted:
                        ledger.failed(
                            snapshot_index,
                            lambda store=store, projection=projection:
                            store.read_state_ref_record(projection.digest()) is not None,
                        )
                    if object_metadata_index in attempted:
                        ledger.failed(
                            object_metadata_index,
                            lambda store=store, projection=projection, annotations=annotations:
                            store.read_metadata(projection.object) == annotations.object,
                        )
                    if state_metadata_index in attempted:
                        ledger.failed(
                            state_metadata_index,
                            lambda store=store, projection=projection, annotations=annotations:
                            store.read_metadata(projection) == annotations.state,
                        )
                    if membership_index in attempted:
                        ledger.failed(
                            membership_index,
                            lambda store=store, record=record:
                            store.read_stored_root_record(record.digest) is not None,
                        )
                    if claim_index in attempted:
                        ledger.failed(
                            claim_index,
                            lambda store=store, lease=lease, projection=projection:
                            _has_completed_claim(store, lease, projection),
                        )
                    _raise_publication_failure(error, ledger.report(state_stores, (), snapshots))
                for path in selected_sources:
                    state_stores.setdefault(action.path.join(path), []).append(store)
                snapshots.append(SavedSnapshot(projection, (store,), (store,)))
            if action.path and action.obj is not None:
                action.obj._last_state_ref = projection
        root_complete = [item for item in snapshots if item.state_ref == state_ref]
        if not root_complete:
            raise RepoSaveError("Root snapshot placement was not completed.", report=ledger.report(state_stores, (), snapshots))
        plan.binding.roots[0].obj._last_state_ref = state_ref
        repo._num_saves += 1
        required = _unique_stores(
            [
                *(store for saved in snapshots for store in saved.required_stores),
                *(source.store for source in seed_sources.values()
                  if routed.graph_mode != "closure"),
            ]
        )
        result = (state_ref, ledger.report(state_stores, required, snapshots))
        return result if report_stores else state_ref
    except BaseException as error:
        if state_ref is not None:
            for lease, action in claims:
                projection = state_ref if not action.path else state_ref.at(action.path)
                if _has_completed_claim(lease.store, lease, projection):
                    repo._clear_completed_routed_claim(plan, lease)
        _raise_publication_failure(error, ledger.report(state_stores, (), snapshots))
    finally:
        for source in source_by_path.values():
            if os.path.commonpath((
                    os.path.realpath(os.fspath(source.handle)),
                    os.path.realpath(getattr(source.store, "_staging_root", "")),
            )) == os.path.realpath(getattr(source.store, "_staging_root", "")):
                source.store.discard_local_state_staging(source.handle)


def _raise_publication_failure(error: BaseException, report: StoreReport) -> None:
    """Raise a save failure without losing interruption identity or evidence."""

    from .repo import MetadataConflictError, RepoSaveError

    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        error.report = report
        raise error
    if isinstance(error, (MetadataConflictError, _SourceSelectionError)):
        raise error
    if isinstance(error, RepoSaveError):
        if error.report is None:
            error.report = report
        raise error
    raise RepoSaveError("Save publication failed.", report=report) from error


def _save_error(path: GraphPath, message: str, cause: BaseException | None = None):
    """Return a path-specific RepoSaveError retaining a useful failure cause."""
    from .repo import RepoSaveError

    error = RepoSaveError(f"Graph save at {path!s}: {message}.")
    if cause is not None:
        error.__cause__ = cause
    return error


def _collect_runtime_roots(value: Any, path: GraphPath, roots: list[RuntimeRoot]) -> None:
    if is_runtime_leaf(value):
        return
    kind = node_kind(value)
    if kind is NodeKind.OBJECT:
        roots.append(RuntimeRoot(value.definition, path, value))
        return
    if kind is NodeKind.CONCRETE_DEFINITION:
        roots.append(RuntimeRoot(value, path, None))
        return
    if kind is NodeKind.DEFINITION:
        from .repo import RepoGraphError

        raise RepoGraphError("Plain Definitions aren't allowed here.")
    if kind in {
        NodeKind.LIST,
        NodeKind.TUPLE,
        NodeKind.SET,
        NodeKind.DICT,
        NodeKind.FROZEN_LIST,
        NodeKind.FROZEN_TUPLE,
        NodeKind.FROZEN_SET,
        NodeKind.FROZEN_DICT,
    }:
        for edge in iter_value_edges(value):
            _collect_runtime_roots(edge.value, path.child(edge.segment), roots)
        return
    from .repo import RepoGraphError

    raise RepoGraphError(f"Unexpected object of type {type(value).__name__} at {path.legacy_str()}!")


def _cached_object(repo, cdef: ConcreteDefinition) -> Object | None:
    """Return one unambiguous candidate from the explicitly supplied Repo."""

    return repo.get_cached(cdef)


def _resolve_missing_for_traversal(
        repo,
        cdef: ConcreteDefinition,
        path: GraphPath,
        options: RepoGraphOptions,
        load_memo: dict[ConcreteDefinition, Object]) -> Object | None:
    obj = repo.get_cached(cdef)
    if obj is not None:
        return obj
    if options.missing == "skip":
        return None
    if options.missing == "load":
        return repo._materialize_cdef(cdef, memo=load_memo, path=list(path.legacy_tuple()))
    from .repo import RepoGraphError

    raise RepoGraphError(
        f"Definition {cdef} is not reachable as a live object in this repo at {path.legacy_str()}."
    )


def _node_primary_path(
        graph: ConcreteDefinitionGraph,
        roots: tuple[RuntimeRoot, ...],
        cdef: ConcreteDefinition) -> GraphPath:
    for root in roots:
        rel_path = graph.primary_path(root.definition, cdef)
        if rel_path is not None:
            return root.path.join(rel_path)
    return GraphPath()


def _add_object_single(repo, obj: Object, *, store=None) -> None:
    repo.pin(obj)
    if store is not None:
        repo.set_object_store(obj, store)
    elif obj.definition not in repo.obj_default_store:
        if repo.default_store is not None:
            repo.set_object_store(obj, repo.default_store)


def _materialize_reachable_nodes(graph: ConcreteDefinitionGraph, roots: tuple[RuntimeRoot, ...]) -> set[ConcreteDefinition]:
    out: set[ConcreteDefinition] = set()
    stack = [root.definition for root in roots]
    while stack:
        cdef = stack.pop()
        if cdef in out:
            continue
        out.add(cdef)
        for edge in graph.outgoing(cdef):
            if edge.kind is EdgeKind.MATERIALIZE:
                stack.append(edge.child)
    return out


def _validate_graph_options(options: RepoGraphOptions) -> None:
    if options.order not in ("pre", "post"):
        raise ValueError("Repo graph order must be 'pre' or 'post'.")
    if options.missing not in ("raise", "skip", "load"):
        raise ValueError("Repo graph missing policy must be 'raise', 'skip', or 'load'.")
