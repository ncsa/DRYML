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
from .generator import GeneratorSelector
from .store.store import Store, StoreCapabilityError
from .utils.graph.path import GraphPath, graph_path_sort_key
from .utils.graph.value import iter_value_edges
from .utils.graph.path import Parameter
from .cdef_identity import cdef_node_key

if TYPE_CHECKING:
    from .reference_values import ObjectId


T = TypeVar("T")


def state_use_leases(sources):
    """Return one globally ordered active-use lease context for exact plan sources.

    sources is an iterable of (Store, StateKey) pairs. The context yields True
    after native shared acquisition, before any Store writer/read fences. Callers
    then freshly validate their source availability/compatibility; a plan itself
    confers no payload validity. Growing an already held set raises
    StoreAuthorityError: release/reacquire the complete set and revalidate.
    This primitive does not execute legacy save/load or managed association policy.
    """
    from .store.retention import state_leases
    return state_leases(sources)


@dataclass(frozen=True, slots=True)
class SaveRouting:
    """Detached, immutable ordered destination policy for future Repo saves.

    Args:
        routes: Ordered ``(Selector | GeneratorSelector, Store)`` bindings. Store handles are
            retained, never opened or created, and must be connected to a Repo
            before that Repo installs this policy.
        match_mode: ``"first"`` selects the first matching binding and
            ``"all"`` selects every distinct matching destination.
        graph_mode: ``"per-object"`` or ``"closure"`` placement mode.

    Raises:
        TypeError: If fields have unsupported types or route entries are not
            Selector or GeneratorSelector/Store pairs.
        ValueError: If a mode is not one of the supported closed values.

    Side Effects:
        Normalizes routes to an immutable tuple.  Constructing this detached
        policy never opens, creates, validates, or publishes to a Store.
    """

    routes: tuple[tuple[Selector | GeneratorSelector, Store], ...] = ()
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
            if not isinstance(selector, (Selector, GeneratorSelector)):
                raise TypeError("SaveRouting route selectors must be Selectors or GeneratorSelectors.")
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
    """Retained live bindings and pure Node preparation for one logical save.

    graph/binding/actions/snapshots retain exact live occurrences and routes;
    object_ref/object_ids adapt the pre-cutover live identity interface. node is
    the sole Node realization carrier, not a second StateRef state map.
    preparation owns the final shared export and observable work counts;
    capture_counts detaches earlier creation-key work without retaining a second
    graph owner. No field proves current Store availability or grants a lease.
    Construct through build_save_plan; validation/capture/publication failures
    raise RepoSaveError at their owning operations. Preparation itself performs
    no hooks, payload IO, construction, ObjectId allocation or publication.
    """

    graph: ConcreteDefinitionGraph
    binding: RuntimeGraphBinding
    actions: tuple[SaveAction, ...]
    snapshots: tuple[SnapshotAction, ...]
    nodes: tuple[Object, ...]
    object_ref: object
    object_ids: tuple[object, ...]
    node: object = None
    preparation: object = None
    capture_counts: Mapping = field(default_factory=lambda: MappingProxyType({}))


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
        state_ref: Planned SavedNode metadata, or ``None``. Confirmed graph
            authority is listed separately in StoreReport.snapshots.
        phase: Closed publication phase, including separate membership and claim
            boundaries.
        status: ``completed``, ``failed``, ``unattempted``, or ``uncertain``.
        store_uuid: Exact destination authority UUID.
        cut: Observed AuthorityRecord; never a cross-Store transaction cut.
        mutation_id: Exact journal mutation UUID, or None before an intent.
        index_disposition: Complete or repair_pending for the combined batch.
        retention_token: Target-local owner token; provisional/uncertain grants
            must not be retired without owner-fenced resolution.
        state_keys: Authoritatively confirmed available full keys at read-back.
        planned_state_keys: All intended target-local full keys, including failures.
        protected_state_keys: Subset of state_keys confirmed with this batch's
            matching owner intent and physical grants. Only these qualify recovery.

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
    state_ref: object | None
    phase: PublicationPhase
    status: PublicationStatus
    store_uuid: str | None = None
    cut: object = None
    mutation_id: str | None = None
    index_disposition: str | None = None
    retention_token: str | None = None
    state_keys: tuple = ()
    planned_state_keys: tuple = ()
    protected_state_keys: tuple = ()


@dataclass(frozen=True, slots=True)
class SavedSnapshot:
    """One immutable independently confirmed exact saved-graph metadata root.

    Args:
        state_ref: Confirmed exact root or child SavedNode (historical field name).
        stores: Ordered snapshot-record destinations that confirmed this StateRef.
        required_stores: Ordered sufficient connected Store dependencies for
            protected exact recovery at reconciliation, or empty when incomplete.

    Per-object snapshots can require external connected Stores. A closure-mode
    root destination instead contains a self-contained complete root closure on
    success. Confirmed metadata alone does not prove payload availability; an
    empty required_stores never certifies recovery, including on partial failure.
    This live inspection record is neither portable configuration nor persistent
    authority.
    """

    state_ref: object
    stores: tuple[Store, ...]
    required_stores: tuple[Store, ...]


@dataclass(frozen=True, slots=True)
class StoreReport:
    """Immutable save publication and exact-recovery inspection evidence.

    Args:
        target_stores: Ordered selected root snapshot Stores.
        state_stores: Confirmed available local-state Stores by canonical path;
            protection is reported separately in publications.
        required_stores: One deterministic sufficient connected recovery set,
            or empty when the root closure was not confirmed protected.
        snapshots: Confirmed root/child metadata, each separately recovery-qualified.
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
    """Expose nested materialized bindings below their outer occurrence.

    Nested CDefs and materializing exact references construct a live subtree. Its
    private runtime projection remains authoritative, but its paths must also be
    visible from the enclosing root for exact graph saves. ``Ref`` links
    deliberately retain their immutable reference value and never contribute
    owned runtime bindings.
    """

    from .cdef_graph import EdgeKind
    from .links import DefLink
    from .reference_values import ObjectRef, StateRef

    if isinstance(canonical, (ObjectRef, StateRef)) or (
            isinstance(canonical, ConcreteDefinition)
            and path.segments and isinstance(runtime, Object)):
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
    from .node import _as_node_view
    retained = getattr(value, "_construction_node", None)
    node = (_as_node_view(value.definition) if retained is None else retained).with_realization(identities=reference.objects)
    return SavePlan(
        graph=binding.graph,
        binding=binding,
        actions=tuple(actions),
        snapshots=tuple(snapshots),
        nodes=tuple(nodes),
        object_ref=reference,
        object_ids=tuple(reference.objects.values()),
        node=node,
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


def _validate_codecs(actions: Iterable[SaveAction]) -> None:
    """Validate every selected developer codec before any serializer runs."""
    for action in actions:
        if action.state_hash is not None:
            continue
        codec = getattr(action.obj, "state_codec", None)
        if not isinstance(codec, str) or not _CODEC_RE.fullmatch(codec):
            raise _save_error(action.path, "state_codec must match [A-Za-z0-9]{1,32}")


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


@dataclass(frozen=True)
class NodeSaveBatch:
    """Detached shared graph and exact state inputs for one Store publication.

    block/roots/creations/projections are immutable Node records plus their pure
    semantic handles. states contains (manifest, owned staging or None) pairs;
    None requires currently installed authenticated data. token is an owner UUID,
    graph_digest the logical saved root, names NodeNameRecords. provisional keeps
    the owner ledger prepared for a later association. This value grants no
    authority or lease; only its target Store may validate and publish it.
    creation_sources carries Store-owned detached metadata witnesses from explicit
    selection reads. Current-cut and byte revalidation may reuse interpreted
    headers after the borrowed data context expires; this never extends payload
    validity, source lifetime or key protection.
    """

    block: object
    roots: tuple
    creations: tuple
    projections: tuple
    states: tuple
    token: str
    graph_digest: str
    names: tuple = ()
    provisional: bool = False
    creation_sources: tuple = ()


def _save_graph_records(saved, roots, origins, previous):
    from .node import _all_nodes
    from .node_prepare import NodePreparation
    from .node_views import creation_cdef, object_projection, CREATION_VIEW, OBJECT_PROJECTION_VIEW
    from .node_codec import encode_node_graph
    from .store.records import NodeBlockRecord, NodeRootRecord, _json_bytes

    projected = object_projection(saved)
    projections = {}
    for node in _all_nodes(projected):
        if node.object_id is None:
            continue
        previous_node = projections.get(node.object_id)
        if previous_node is not None and not previous_node.graph_equal(node):
            from .repo import RepoSaveError
            raise RepoSaveError("Conflicting ObjectId declarations within the saved graph.")
        projections[node.object_id] = node
    recipe = creation_cdef(saved)
    entries = [("saved", node) for node in roots]
    entries += [("recipe", recipe)]
    entries += [("construction", node) for node in origins]
    entries += [("projection", node) for node in projections.values()]
    prepared = NodePreparation((node for _, node in entries), previous=previous)
    packet = encode_node_graph(prepared)
    views = packet.pop("views")
    block = NodeBlockRecord(_json_bytes(packet))
    records = []
    block_digest = None
    for (kind, node), view in zip(entries, views):
        policy = {"construction": CREATION_VIEW.name,
                  "projection": OBJECT_PROJECTION_VIEW.name}.get(kind, "exact")
        view.update(view=policy, expected_key=prepared.semantic_key(node, policy=policy).decode("ascii"))
        if block_digest is None:
            block_digest = block.digest
        records.append((NodeRootRecord(block_digest, kind, _json_bytes(view)), node))
    return block, records, prepared


def _confirmed_batch_keys(target, batch):
    """Return (available keys, protected subset) from one validated local cut."""
    from .store.retention import RetentionGrantRecord, _grant_path

    result = []
    protected = []
    try:
        with target._metadata_read():
            origins = {r.digest: n for r, n in batch.creations}
            for manifest, _ in batch.states:
                descriptor = target._read_variant(manifest.key, origins[manifest.key.cdef_digest])
                if descriptor is not None and descriptor.available and descriptor.manifest == manifest:
                    result.append(manifest.key)
            intent = target.retention.read(batch.token)
            keys = tuple(sorted((m.key for m, _ in batch.states),
                                key=lambda k: (k.cdef_digest, k.codec_hash)))
            if (intent is None or intent.disposition not in {"prepared", "adopted"}
                    or intent.graph_digest != batch.graph_digest
                    or len(intent.resources) != 1
                    or intent.resources[0].domain != target.retention_domain
                    or intent.resources[0].keys != keys):
                return tuple(result), ()
            grant = RetentionGrantRecord(target.retention_domain, batch.token, batch.graph_digest, keys) if keys else None
            for key in result:
                if target._read_file(_grant_path(target, key, grant), RetentionGrantRecord) == grant:
                    protected.append(key)
    except Exception:
        return (), ()
    return tuple(result), tuple(protected)


def _ready_save_batch(pending, confirmed_keys, required_keys):
    """Prefer resolved closures, then safely prepared mutually dependent batches.

    Store grouping can introduce cycles absent from the live Object DAG. Owned
    staging and the already acquired complete key-lease set permit local batch
    publication in such a component; only later protected-provider reconciliation
    can qualify its metadata for recovery. Prune targets blocked by failed keys
    transitively so this fallback cannot attempt their dependent associations.
    """
    local = [{m.key for m, _ in batch.states} for _, batch, _ in pending]
    for index, (_, batch, _) in enumerate(pending):
        if required_keys[id(batch)] <= local[index] | confirmed_keys:
            return index
    viable = list(range(len(pending)))
    while viable:
        possible = confirmed_keys.union(*(local[index] for index in viable))
        remaining = [index for index in viable if required_keys[id(pending[index][1])] <= possible]
        if remaining == viable:
            return viable[0]
        viable = remaining
    return None


def execute_routed_save_plan(
        repo, routed, *, states=None, main=False, alias=None,
        report_stores=False, annotations=None, source_store=None,
        source_stores=None, snapshot_observer=None, deep_capture=False,
        provisional=False):
    """Capture remaining distinct live nodes once and publish format-4 batches.

    states is a partial rooted StateSpec/LocalState map. Explicit selections are
    compatibility/availability checked before hooks; plain saves never infer reuse
    from receipts. Destinations are retained by routed. Each Store gets one shared
    table, full-key variants and protected graph membership in one journal/index
    batch. Returns SavedNode, optionally paired with StoreReport. Failures raise
    RepoSaveError with per-target confirmed/uncertain evidence; independent targets
    continue. Prepared local batches may publish across aggregate Store cycles;
    recovery sets and live receipts qualify only after every required full key
    has a confirmed protected provider. Incomplete metadata never advances a
    receipt. No graph preparation, hook or payload authentication runs under a
    Store writer fence. Buffered durable ownership and unsupported metadata feeds
    fail explicitly before capture. Repair uses Store journal APIs, never capture.
    """
    from contextlib import ExitStack
    import shutil
    from .node import require_saved
    from .node_views import creation_cdef, CREATION_VIEW
    from .state_values import LocalState, StateKey
    from .store.records import StateVariantManifest, NodeNameRecord, _json_bytes
    from .repo import RepoSaveError

    plan = routed.plan
    root_obj = plan.binding.roots[0].obj
    targets = _unique_stores(s for values in routed.destinations.values() for s in values)
    if annotations is not None:
        from .metadata import SaveAnnotations
        if not isinstance(annotations, SaveAnnotations):
            raise TypeError("annotations must be a SaveAnnotations or None.")
        raise StoreCapabilityError("Node save annotation/lifecycle metadata feeds are not integrated.")
    if snapshot_observer is not None:
        raise StoreCapabilityError("Node save lifecycle observation coverage is not integrated.")
    if any(getattr(node, "_claim_lease", None) or getattr(node, "_pending_claim_dependencies", ()) for node in plan.nodes):
        raise StoreCapabilityError("Legacy construction-claim publication requires the Node materialization cutover.")
    if any(action.state_hash is not None for action in plan.actions):
        raise StoreCapabilityError("Legacy exact-reference seed publication requires the Node materialization cutover.")
    for target in targets:
        target.preflight_save_batch()
    from .node import _owned_nodes
    from .node_prepare import NodePreparation
    selected = plan.node.with_realization(states={p: None for p, _ in _owned_nodes(plan.node)})
    selected = selected.with_realization(states=states)
    selected_map = dict(selected.state_spec().items())
    actions = {action.path: action for action in plan.actions}
    for path, state in selected_map.items():
        if path not in actions:
            raise ValueError("Selected state is outside the live stateful ownership closure.")
        state.validate_for(selected.at(path), codec=actions[path].obj.state_codec)
    _validate_codecs(plan.actions)
    if source_store is not None:
        if not isinstance(source_store, Store):
            raise TypeError("source_store must be a Store.")
        if source_store not in routed.context.stores:
            raise ValueError("State source Store must be connected.")
    if source_stores is not None and not isinstance(source_stores, Mapping):
        raise TypeError("source_stores must be a mapping or None.")
    source_stores = {} if source_stores is None else dict(source_stores)
    if any(not isinstance(key, StateKey) or not isinstance(store, Store) for key, store in source_stores.items()):
        raise TypeError("source_stores must map full StateKeys to Store handles.")
    if set(source_stores) - {s.key for s in selected_map.values()}:
        raise ValueError("source_stores contains a key outside the explicit state selection.")
    if any(s not in routed.context.stores for s in source_stores.values()):
        raise ValueError("State source Stores must be connected.")
    source_by_key = {}
    staged = []
    publications, snapshots, state_stores = [], [], {}
    saved = None
    with ExitStack() as lifetime:
        # Source contexts protect explicit selections through copying/publication.
        # Acquire the entire source key set in global order before borrowing.
        candidates = {state.key: tuple(s for s in
                      ((source_stores.get(state.key) or source_store),)
                      if s is not None) or routed.context.stores
                      for state in selected_map.values()}
        lifetime.enter_context(state_use_leases((s, key) for key, stores in candidates.items() for s in stores))
        for source in _unique_stores(s for stores in candidates.values() for s in stores):
            requested = [key for key, stores in candidates.items() if source in stores and key not in source_by_key]
            if requested:
                borrowed = lifetime.enter_context(source.read_states(requested))
                source_by_key.update(borrowed)
        for state in selected_map.values():
            if state.key not in source_by_key:
                raise RepoSaveError("Explicit selected state is unavailable in the connected source scope.")
            if source_by_key[state.key].state != state:
                raise RepoSaveError("Selected state creation authority conflicts with its source.")
            if source_by_key[state.key].manifest.codec_version != 1:
                raise RepoSaveError("Selected state codec version is not admitted by the live destination.")
        # Source leases cannot be grown with destination keys discovered by capture.
        # Borrowed bytes are copied to owned staging before releasing that set.
        try:
            states_by_path = {}
            origins = [creation_cdef(plan.node.at(action.path)) for action in plan.actions]
            origin_preparation = NodePreparation(origins)
            payloads = {}
            for action, origin in zip(plan.actions, origins):
                reused = selected_map.get(action.path)
                if reused is not None:
                    states_by_path[action.path] = reused
                    payloads[reused.key] = source_by_key[reused.key]
                    continue
                target = routed.destinations[action.path][0]
                stage = target.create_local_state_staging()
                staged.append((target, stage))
                lock = action.obj._save_load_reservation
                if not lock.acquire(blocking=False):
                    raise _save_error(action.path, "local state is already reserved by save or restore")
                try:
                    data = os.path.join(os.fspath(stage), "data")
                    action.obj.save_state_to_dir(data, codec=action.obj.state_codec)
                    files = target._state_inventory(data)
                    interpretation = {"deferred_paths": list(action.obj.deferred_state_payload_paths(data, codec=action.obj.state_codec))}
                    codec_hash = StateVariantManifest.content_hash(action.obj.state_codec, 1, interpretation, files)
                    creation_key = origin_preparation.semantic_key(origin, policy=CREATION_VIEW.name)
                    digest = hashlib.sha256(b"dryml-creation-cdef/v1\0" + creation_key).hexdigest()
                    key = StateKey(digest, codec_hash)
                    manifest = StateVariantManifest(key, 1, _json_bytes(interpretation), files)
                    action.obj._last_state_hash = key.codec_hash
                    local = LocalState(key, origin)
                    states_by_path[action.path] = local
                    payloads.setdefault(key, (target, stage, manifest))
                except BaseException as error:
                    if not isinstance(error, Exception):
                        raise
                    raise _save_error(action.path, "snapshot-local state preparation failed", error) from error
                finally:
                    lock.release()
            saved = require_saved(plan.node.with_realization(states=states_by_path))
            roots = [saved] if routed.graph_mode == "closure" else [saved.at(a.path) for a in plan.snapshots]
            all_origins = list({state.key.cdef_digest: state.creation_cdef for state in states_by_path.values()}.values())
            plan.capture_counts = MappingProxyType(dict(origin_preparation.counts))
            block, records, plan.preparation = _save_graph_records(saved, roots, all_origins, origin_preparation)
            saved_records = records[:len(roots)]
            recipe_record = records[len(roots)]
            creation_records = records[len(roots) + 1:len(roots) + 1 + len(all_origins)]
            projection_records = records[len(roots) + 1 + len(all_origins):]
            root_record = next(record for record, node in saved_records if node is saved)
            batches = []
            for target in targets:
                local_paths = [a.path for a in plan.actions if target in routed.destinations[a.path]]
                local_keys = tuple(dict.fromkeys(states_by_path[p].key for p in local_paths))
                installs = []
                for key in local_keys:
                    source = payloads[key]
                    if not isinstance(source, tuple) and source.store is target:
                        installs.append((source.manifest, None))
                        continue
                    if isinstance(source, tuple):
                        owner, stage, manifest = source
                        data = os.path.join(os.fspath(stage), "data")
                    else:
                        manifest, data = source.manifest, source.data_directory
                    # Stage once per distinct full key and target, never per snapshot.
                    if not isinstance(source, tuple) or owner is not target:
                        stage = target.create_local_state_staging()
                        staged.append((target, stage))
                        shutil.copytree(data, os.path.join(stage, "data"), dirs_exist_ok=True)
                    installs.append((manifest, stage))
                chosen = [(record, node) for index, (record, node) in enumerate(saved_records)
                           if target in (routed.root_destinations if routed.graph_mode == "closure" else
                                         routed.destinations[plan.snapshots[index].path])]
                if target in routed.root_destinations:
                    chosen.append(recipe_record)
                names = []
                if target in routed.root_destinations:
                    if alias is not None:
                        names.append(NodeNameRecord(alias, "saved", root_record.digest))
                    if main:
                        names.append(NodeNameRecord("main", "recipe", recipe_record[0].digest))
                user_root = _json_bytes([root_record.digest, sorted((k.cdef_digest, k.codec_hash) for k in local_keys),
                                         sorted((r.root_kind, r.digest) for r, _ in chosen)]).decode("ascii")
                token = str(uuid.uuid4()) if provisional else str(uuid.uuid5(uuid.UUID(target.store_uuid), "dryml-user-root/v1:" + user_root))
                old_intent = target.retention.read(token)
                if old_intent is not None and old_intent.disposition != "adopted":
                    token = str(uuid.uuid4())
                batch = NodeSaveBatch(block, tuple(chosen), tuple(creation_records), tuple(projection_records),
                                      tuple(installs), token, root_record.digest, tuple(names), provisional,
                                      tuple(source_by_key.values()))
                batches.append((target, batch, local_paths))
            # Every explicit source has now been detached into owned staging.
            lifetime.close()
            errors = []
            lease_targets = [(target, manifest.key) for target, batch, _ in batches for manifest, _ in batch.states]
            from .store.retention import state_leases
            recovery = any(target.save_batch_needs_recovery(batch) for target, batch, _ in batches)
            lease_scope = state_leases(lease_targets, exclusive=True, blocking=False) if recovery else state_use_leases(lease_targets)
            with lease_scope as acquired:
                if not acquired:
                    raise RepoSaveError("State recovery is in-use; retry after active leases end.")
                validate_retained_save_plan(plan, root_obj)
                pending = list(batches)
                confirmed_provider_keys = set()
                required_keys = {
                    id(batch): {s.key for record, node in batch.roots if record.root_kind == "saved"
                                for _, s in node.state_spec().items()}
                    for _, batch, _ in batches}
                while pending:
                    ready = _ready_save_batch(pending, confirmed_provider_keys, required_keys)
                    if ready is None:
                        for target, batch, _ in pending:
                            publications.append(SavePublication(target, None, None, saved, "snapshot", "unattempted",
                                                target.store_uuid, retention_token=batch.token,
                                                planned_state_keys=tuple(m.key for m, _ in batch.states)))
                        errors.append(RepoSaveError("Dependent save targets were not attempted; required state publication is incomplete."))
                        break
                    target, batch, local_paths = pending.pop(ready)
                    try:
                        identity, cut = target.publish_save_batch(batch)
                    except BaseException as error:
                        identity, cut = getattr(error, "mutation_id", None), getattr(error, "cut", None)
                        status = "uncertain" if identity else "failed"
                        errors.append(error)
                    else:
                        status = "completed"
                    confirmed_keys, protected_keys = _confirmed_batch_keys(target, batch)
                    confirmed_provider_keys.update(protected_keys)
                    publications.append(SavePublication(
                        target, None, None, saved, "snapshot", status, target.store_uuid, cut,
                        identity, "complete" if status == "completed" else "repair_pending",
                        batch.token, confirmed_keys, tuple(m.key for m, _ in batch.states), protected_keys))
                    # Read-back may prove authority even when projection acknowledgement failed.
                    for record, node in batch.roots:
                        if record.root_kind != "saved":
                            continue
                        try:
                            confirmed = target.read_node(record.digest, kind="saved")
                        except Exception:
                            confirmed = None
                        if confirmed is not None:
                            snapshots.append(SavedSnapshot(node, (target,), ()))
                    for path in local_paths:
                        if states_by_path[path].key in confirmed_keys:
                            state_stores.setdefault(path, []).append(target)
                    if errors and not isinstance(errors[-1], Exception):
                        for waiting, waiting_batch, _ in pending:
                            publications.append(SavePublication(waiting, None, None, saved, "snapshot", "unattempted",
                                                waiting.store_uuid, retention_token=waiting_batch.token,
                                                planned_state_keys=tuple(m.key for m, _ in waiting_batch.states)))
                        break
                # Reconcile while the complete target key set is still leased.
                # Local root adoption protects only that Store's exact resources;
                # saved metadata can outlive remote payloads and is not a receipt.
                providers = {}
                for publication in publications:
                    for key in publication.protected_state_keys:
                        providers.setdefault(key, []).append(publication.store)
                sufficient = []
                for snapshot in snapshots:
                    required = list(snapshot.stores)
                    complete = True
                    for _, state in snapshot.state_ref.state_spec().items():
                        options = providers.get(state.key, ())
                        complete &= bool(options)
                        if not any(store in options for store in required) and options:
                            required.append(options[0])
                    sufficient.append(replace(snapshot, required_stores=_unique_stores(required) if complete else ()))
                root_snapshot = next((s for s in sufficient if s.state_ref is saved), None)
                required = () if root_snapshot is None else root_snapshot.required_stores
                report = StoreReport(routed.root_destinations, state_stores, required, sufficient, publications)
                qualified_nodes = {id(s.state_ref) for s in sufficient if s.required_stores}
                for action in plan.snapshots:
                    node = saved.at(action.path)
                    if action.obj is not None and (required or id(node) in qualified_nodes):
                        action.obj._last_state_ref = node
            if errors:
                failure = errors[-1] if not isinstance(errors[-1], Exception) else errors[0]
                _raise_publication_failure(failure, report)
            if root_snapshot is None or not required:
                raise RepoSaveError("Saved graph publication could not be confirmed at read-back.", report=report)
            repo._num_saves += 1
            return (saved, report) if report_stores else saved
        finally:
            for owner, stage in staged:
                # An ambiguous journal may still need these exact staged resources.
                if any(p.status != "completed" and p.mutation_id for p in publications):
                    owner.release_local_state_staging(stage)
                else:
                    owner.discard_local_state_staging(stage)


def _raise_publication_failure(error: BaseException, report: StoreReport) -> None:
    """Raise a save failure without losing interruption identity or evidence."""

    from .repo import MetadataConflictError, RepoSaveError

    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        error.report = report
        raise error
    if isinstance(error, MetadataConflictError):
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
