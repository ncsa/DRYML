from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..cdef_graph import EdgeKind
from .path import DefinitionPath


RefreshPolicy = Literal[False, "auto", True]
ClassMatchPolicy = Literal["selector", "exact"]
QueryDomain = Literal["stored", "cached", "known", "nested"]
QueryProjection = Literal["definitions", "owners", "object_refs", "state_refs"]
ContainmentEdgePolicy = Literal["materialize", "ref", "all"]
ContainmentTargetKind = Literal["definition", "object_ref", "state_ref"]
ContainmentCarrier = Literal["target", "owner", "occurrence"]


def validate_containment_policy(
        edges: Any, contains_ref: Any) -> tuple[ContainmentEdgePolicy, bool]:
    """Validate the immutable traversal controls for a nested containment query.

    Args:
        edges: Literal edge kinds permitted at every containment traversal hop.
        contains_ref: Exact boolean requiring a qualifying path to include a
            retained ``REF`` hop.

    Returns:
        The validated policy pair.

    Raises:
        ValueError: If ``edges`` is not ``"materialize"``, ``"ref"``, or
            ``"all"``.
        TypeError: If ``contains_ref`` is not an exact bool.

    Side Effects:
        None. Validation does not enumerate Store authority or traverse CDefs.
    """

    if not isinstance(edges, str) or edges not in {"materialize", "ref", "all"}:
        raise ValueError("nested edges must be 'materialize', 'ref', or 'all'.")
    if type(contains_ref) is not bool:
        raise TypeError("nested contains_ref must be an exact bool.")
    return edges, contains_ref


def validate_containment_target(target: Any) -> Any:
    """Require one exact value supported by containment query entry points.

    Args:
        target: Concrete CDef, ObjectRef, or StateRef retained as immutable
            containment-query intent.

    Returns:
        The unchanged validated target.

    Raises:
        TypeError: If ``target`` is not an exact supported containment value.

    Side Effects:
        None. Exact reference targets remain data and are not resolved.
    """

    from ..definition import ConcreteDefinition
    from ..reference_values import ObjectRef, StateRef

    if not isinstance(target, (ConcreteDefinition, ObjectRef, StateRef)):
        raise TypeError(
            "containment target must be a ConcreteDefinition, ObjectRef, or StateRef."
        )
    return target


def is_exact_reference_target(target: Any) -> bool:
    """Return whether a containment target is an exact reference value.

    Exact reference values select containment only; they do not select ordinary
    reference-authority query domains.
    """

    from ..reference_values import ObjectRef, StateRef

    return isinstance(target, (ObjectRef, StateRef))


class QueryError(Exception):
    pass


class QueryDomainError(QueryError):
    pass


class QueryCardinalityError(QueryError):
    pass


@dataclass(frozen=True, slots=True)
class QueryDiagnostic:
    """Bounded structural diagnostics for detached Query V3 fixed results.

    The diagnostic intentionally contains counts and a result domain only. It
    never retains member representations, source values, paths, or upstream
    exception text.
    """

    domain: Literal["identity", "occurrence"]
    member_count: int
    member_count_exact: bool
    bounded: bool
    source_count: int
    source_count_exact: bool
    # The terminal-provided prefix limit, not a safety or verification budget.
    requested_limit: int | None = None

    @classmethod
    def for_fixed(
        cls, domain: Literal["identity", "occurrence"], member_count: int,
        bounded: bool, source_count: int, *, requested_limit: int | None = None,
    ) -> "QueryDiagnostic":
        """Create a disclosure-safe diagnostic with count caps applied."""

        limit = 9_999
        return cls(
            domain=domain,
            member_count=min(member_count, limit),
            member_count_exact=member_count <= limit,
            bounded=bounded,
            source_count=min(source_count, limit),
            source_count_exact=source_count <= limit,
            requested_limit=requested_limit,
        )


class QueryIndexError(QueryError):
    pass


class QueryIndexUnavailable(QueryIndexError):
    pass


class QueryIndexIncompatible(QueryIndexError):
    pass


class QueryIndexCorrupt(QueryIndexError):
    pass


class QueryIndexBusy(QueryIndexError):
    pass


class QueryIndexDirty(QueryIndexError):
    pass


class QueryIndexGenerationChanged(QueryIndexError):
    pass


class QueryWouldScanError(QueryIndexError):
    pass


class QueryVerifyBudgetExceeded(QueryError):
    pass


@dataclass(frozen=True, slots=True)
class QueryIndexStatus:
    backend: str
    store_key: str
    generation: int | None
    schema_version: int | None
    semantic_versions: dict[str, int]
    state: Literal["building", "ready", "missing", "dirty", "incompatible", "corrupt", "disabled", "unavailable"]
    journal_mode: str | None = None
    sqlite_version: tuple[int, int, int] | None = None
    path: str | None = None
    row_counts: dict[str, int] | None = None
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class V3ProjectionCoverage:
    """Integrity and authority coverage for a derived Query V3 projection.

    ``projection_complete`` means the backend's recorded row-set witnesses
    still match its projection rows.  It is deliberately distinct from
    ``authority_complete``: only the latter can support an indexed-only answer.
    Current V3 projections remain candidate accelerators until every authority
    inventory and mutable publication path has a compatible fenced witness.
    """

    backend: str
    store_key: str
    generation: int | None
    projection_complete: bool
    authority_complete: bool
    indexed_only: bool
    source_roots: int = 0
    identities: int = 0
    relationships: int = 0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    message: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Report returned by query-index validation operations."""

    backend: str
    store_key: str
    ok: bool
    issues: tuple[ValidationIssue, ...] = ()
    row_counts: dict[str, int] | None = None
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """Summary of a query-index reconciliation or rebuild pass.

    `changed` records whether the index was modified. `validated` captures
    whether the pass performed consistency validation in addition to any
    rebuild or repair action.
    """

    backend: str
    store_key: str
    changed: bool
    action: Literal["none", "validate", "repair", "rebuild"] = "none"
    generation_before: int | None = None
    generation_after: int | None = None
    roots_added: int = 0
    roots_removed: int = 0
    definitions_scanned: int = 0
    validated: bool = False
    issues: tuple[ValidationIssue, ...] = ()
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SourceQueryPlan:
    source_key: str
    backend: str
    generation: int | None = None
    candidate_count: int = 0
    verified_count: int = 0
    result_count: int | None = None
    refresh_action: str = "none"


@dataclass(frozen=True, slots=True)
class IndexWriteResult:
    generation: int
    changed: bool
    definitions_added: int = 0
    edges_added: int = 0
    postings_added: int = 0
    roots_added: int = 0
    roots_removed: int = 0


@dataclass(frozen=True, slots=True)
class FeatureToken:
    kind: str
    path: DefinitionPath | None
    payload: Any = None


@dataclass(frozen=True, slots=True)
class FeatureRequirement:
    token: FeatureToken
    count: int = 1


# This version covers canonical values embedded in query features and CDef rows.
CANONICAL_QUERY_SEMANTICS_VERSION = 4
FINGERPRINT_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class DefinitionFingerprint:
    counts: dict[FeatureToken, int]
    schema_version: int = FINGERPRINT_SCHEMA_VERSION


DefinitionId = str
StoreId = str
EdgeKey = tuple[DefinitionId, DefinitionPath, DefinitionId, EdgeKind]


@dataclass(frozen=True, slots=True)
class DefinitionRecord:
    definition_id: DefinitionId
    cdef: Any
    class_key: str
    local_fingerprint: DefinitionFingerprint


@dataclass(frozen=True, slots=True)
class DefinitionEdgeRecord:
    edge_key: EdgeKey
    parent_id: DefinitionId
    path: DefinitionPath
    child_id: DefinitionId
    edge_kind: EdgeKind = EdgeKind.MATERIALIZE


@dataclass(frozen=True, slots=True)
class StoreReplica:
    """Replica relationship between a definition and a Store source."""

    definition_id: DefinitionId
    store_id: StoreId


StoredReplica = StoreReplica


@dataclass(frozen=True, slots=True)
class StoredRootMetadata:
    """Persisted metadata recorded for a Store root in a query index."""

    definition_id: DefinitionId
    store_id: StoreId
    storage_hash: str
    relative_def_path: str | None = None
    def_size: int | None = None
    def_mtime_ns: int | None = None
    indexed_generation: int | None = None


@dataclass(frozen=True, slots=True)
class ContainmentHop:
    """One direct edge in a root-local containment witness.

    Attributes:
        path: Typed path from the preceding CDef node to this direct target.
        kind: Literal retained edge role at this hop.

    The path is deliberately local to one CDef boundary.  Joining ordered hops
    recovers the occurrence's full owner-to-target path without losing the
    edge kind associated with each boundary.
    """

    path: DefinitionPath
    kind: EdgeKind


@dataclass(frozen=True, slots=True)
class DefinitionOccurrence:
    """One CDef occurrence, optionally with root-local containment evidence.

    Attributes:
        owner: Enclosing stored or query-owner CDef.
        path: Typed path from ``owner`` to ``definition``.
        definition: Matched CDef value.
        hops: Ordered direct-edge evidence when supplied by containment.

    ``hops`` is intentionally excluded from equality and hashing so existing
    indexed occurrence construction and set behavior remain unchanged.
    """

    owner: Any
    path: DefinitionPath
    definition: Any
    hops: tuple[ContainmentHop, ...] = field(
        default=(), compare=False, hash=False, repr=False,
    )

    @property
    def target(self) -> Any:
        """Return the matched terminal value without changing legacy fields."""

        return self.definition


@dataclass(frozen=True, slots=True)
class ReferenceOccurrence:
    """One exact lightweight reference occurrence in immutable authority.

    Attributes:
        owner: Complete aggregate reference or Definition record containing the
            value.
        path: Typed GraphPath from ``owner`` to ``value``.
        value: Exact ObjectRef or StateRef found at the path.
        hops: Ordered direct-edge evidence when supplied by containment.

    ``hops`` is intentionally excluded from equality and hashing to preserve
    ordinary reference-query occurrence identity and construction.
    """

    owner: Any
    path: DefinitionPath
    value: Any
    hops: tuple[ContainmentHop, ...] = field(
        default=(), compare=False, hash=False, repr=False,
    )

    @property
    def target(self) -> Any:
        """Return the matched terminal value without changing legacy fields."""

        return self.value


def containment_target_kind(value: Any) -> ContainmentTargetKind:
    """Return the non-coercing containment kind for one terminal value.

    Args:
        value: A retained CDef, ObjectRef, or StateRef terminal.

    Returns:
        The terminal's exact containment kind.

    Raises:
        TypeError: If ``value`` is not a supported containment terminal.

    Side Effects:
        None. This classification neither resolves nor projects references.
    """

    from ..definition import ConcreteDefinition
    from ..reference_values import ObjectRef, StateRef

    if isinstance(value, ConcreteDefinition):
        return "definition"
    if isinstance(value, ObjectRef):
        return "object_ref"
    if isinstance(value, StateRef):
        return "state_ref"
    raise TypeError(f"Unsupported containment terminal {type(value).__name__}.")


def containment_witness_key(item: DefinitionOccurrence | ReferenceOccurrence) -> tuple[Any, ...]:
    """Return the complete deduplication key for a containment witness.

    Args:
        item: A CDef or exact-reference occurrence with optional ordered hops.

    Returns:
        A canonical owner/path/hops/terminal key that preserves exact references.

    Side Effects:
        None. The key only reads immutable occurrence evidence.
    """

    from ..definition import ConcreteDefinition
    from ..utils.graph.path import graph_path_sort_key

    def cdef_key(value: ConcreteDefinition) -> tuple[str, str, str]:
        return ("definition", value.stable_hash(), repr(value))

    def value_key(value: Any) -> tuple[Any, ...]:
        kind = containment_target_kind(value)
        if kind == "definition":
            return cdef_key(value)
        return (kind, value.digest())

    return (
        value_key(item.owner),
        graph_path_sort_key(item.path),
        tuple((graph_path_sort_key(hop.path), hop.kind.value) for hop in item.hops),
        value_key(item.target),
    )


@dataclass(frozen=True, slots=True)
class ContainmentContext:
    """Immutable fixed-universe semantics for one containment result.

    Args:
        target_kind: Exact terminal type selected by the containment query.
        edges: Literal traversal policy retained by every result transformation.
        contains_ref: Whether qualified paths require a reference hop.
        source_scope: Canonical selected source identifiers.
        complete: Whether retained witness evidence covers the selected universe.
        bounded: Whether raw occurrence truncation may have omitted witnesses.

    ``complete`` and ``bounded`` describe evidence rather than query identity, so
    compatible set operations combine them conservatively.
    """

    target_kind: ContainmentTargetKind
    edges: ContainmentEdgePolicy = "materialize"
    contains_ref: bool = False
    source_scope: tuple[str, ...] = ()
    complete: bool = True
    bounded: bool = False

    def __post_init__(self) -> None:
        """Validate detached policy metadata without touching Store authority."""

        if self.target_kind not in {"definition", "object_ref", "state_ref"}:
            raise ValueError("Containment target_kind must be definition, object_ref, or state_ref.")
        validate_containment_policy(self.edges, self.contains_ref)
        if not isinstance(self.source_scope, tuple) or not all(isinstance(key, str) for key in self.source_scope):
            raise TypeError("Containment source_scope must be a tuple of canonical source keys.")
        if type(self.complete) is not bool or type(self.bounded) is not bool:
            raise TypeError("Containment completeness and boundedness must be exact bools.")

    def compatible_with(self, other: "ContainmentContext") -> bool:
        """Return whether two contexts may combine without widening authority."""

        return (
            self.target_kind == other.target_kind
            and self.edges == other.edges
            and self.contains_ref == other.contains_ref
            and self.source_scope == other.source_scope
        )

    def combined_with(self, other: "ContainmentContext") -> "ContainmentContext":
        """Conservatively merge evidence state from compatible result contexts.

        Raises:
            ValueError: If policy, target kind, or source scope differs.
        """

        if not self.compatible_with(other):
            raise ValueError("Cannot combine results with incompatible containment contexts.")
        return replace(
            self,
            complete=self.complete and other.complete,
            bounded=self.bounded or other.bounded,
        )


@dataclass(frozen=True, slots=True)
class OwnerProjection:
    owner_ids: frozenset[DefinitionId]
    cdefs: tuple[Any, ...]
    replicas: dict[Any, tuple[Any, ...]]


class OccurrenceTraversalSnapshot:
    def __init__(
            self,
            *,
            targets: set[DefinitionId],
            cdefs: dict[DefinitionId, Any],
            stored_ids: set[DefinitionId],
            incoming: dict[DefinitionId, tuple[DefinitionEdgeRecord, ...]],
            owner_replicas: dict[Any, tuple[Any, ...]] | None = None,
            copy_data: bool = True):
        self.targets = frozenset(targets)
        self.cdefs = dict(cdefs) if copy_data else cdefs
        self.stored_ids = frozenset(stored_ids) if copy_data else stored_ids
        self.incoming = dict(incoming) if copy_data else incoming
        self.owner_replicas = {} if owner_replicas is None else (dict(owner_replicas) if copy_data else owner_replicas)

    def restrict_targets(self, targets: set[DefinitionId] | frozenset[DefinitionId]) -> "OccurrenceTraversalSnapshot":
        return OccurrenceTraversalSnapshot(
            targets=set(targets) & set(self.targets),
            cdefs=self.cdefs,
            stored_ids=self.stored_ids,
            incoming=self.incoming,
            owner_replicas=self.owner_replicas,
            copy_data=False,
        )

    def _owner_ids_for_nested_ids(self, ids: set[DefinitionId] | frozenset[DefinitionId]) -> set[DefinitionId]:
        owners: set[DefinitionId] = set()
        seen: set[DefinitionId] = set(ids)
        stack = list(ids)
        while stack:
            cur = stack.pop()
            for edge in self.incoming.get(cur, ()):
                parent_id = edge.parent_id
                if parent_id in self.stored_ids:
                    owners.add(parent_id)
                if parent_id not in seen:
                    seen.add(parent_id)
                    stack.append(parent_id)
        return owners

    def project_owners(self, ids: set[DefinitionId] | frozenset[DefinitionId]) -> OwnerProjection:
        owner_ids = self._owner_ids_for_nested_ids(ids)
        owners = tuple(self.cdefs[owner_id] for owner_id in owner_ids if owner_id in self.cdefs)
        replicas = {owner: self.owner_replicas.get(owner, ()) for owner in owners}
        return OwnerProjection(frozenset(owner_ids), owners, replicas)

    def iter_occurrences(self, *, max_occurrences: int | None = None):
        yielded = 0
        if max_occurrences is not None and max_occurrences <= 0:
            return
        for target_id in sorted(self.targets):
            if target_id not in self.cdefs:
                continue
            stack = [(target_id, DefinitionPath(), ())]
            while stack:
                cur_id, suffix, suffix_hops = stack.pop()
                edges = sorted(
                    self.incoming.get(cur_id, ()),
                    key=lambda edge: (edge.parent_id, str(edge.path)),
                    reverse=True,
                )
                for edge in edges:
                    path = edge.path.join(suffix)
                    hops = (ContainmentHop(edge.path, edge.edge_kind),) + suffix_hops
                    if edge.parent_id in self.stored_ids:
                        yielded += 1
                        yield DefinitionOccurrence(
                            self.cdefs[edge.parent_id], path,
                            self.cdefs[target_id], hops,
                        )
                        if max_occurrences is not None and yielded >= max_occurrences:
                            return
                    stack.append((edge.parent_id, path, hops))


class AllOccurrenceTraversalSnapshot:
    def __init__(
            self,
            *,
            cdefs: dict[DefinitionId, Any],
            stored_ids: set[DefinitionId],
            outgoing: dict[DefinitionId, tuple[DefinitionEdgeRecord, ...]]):
        self.cdefs = dict(cdefs)
        self.stored_ids = frozenset(stored_ids)
        self.outgoing = dict(outgoing)

    def iter_occurrences(self, *, max_occurrences: int | None = None):
        yielded = 0
        if max_occurrences is not None and max_occurrences <= 0:
            return
        for owner_id in sorted(self.stored_ids):
            stack = []
            for edge in sorted(self.outgoing.get(owner_id, ()), key=lambda edge: str(edge.path), reverse=True):
                if edge.edge_kind is not EdgeKind.MATERIALIZE:
                    continue
                stack.append((
                    edge.child_id, edge.path,
                    (ContainmentHop(edge.path, edge.edge_kind),),
                ))
            while stack:
                did, path, hops = stack.pop()
                yielded += 1
                yield DefinitionOccurrence(
                    self.cdefs[owner_id], path, self.cdefs[did], hops,
                )
                if max_occurrences is not None and yielded >= max_occurrences:
                    return
                for edge in sorted(self.outgoing.get(did, ()), key=lambda edge: str(edge.path), reverse=True):
                    if edge.edge_kind is not EdgeKind.MATERIALIZE:
                        continue
                    stack.append((
                        edge.child_id, path.join(edge.path),
                        hops + (ContainmentHop(edge.path, edge.edge_kind),),
                    ))


@dataclass(frozen=True, slots=True)
class QueryExplanation:
    domain: str
    refresh: RefreshPolicy
    refresh_action: str = "none"
    store_scan_count: int = 0
    universe_size: int | None = None
    selected_features: tuple[FeatureRequirement, ...] = ()
    posting_sizes: tuple[int, ...] = ()
    candidate_count: int = 0
    verified_count: int = 0
    result_count: int | None = None
    fast_path: str | None = None
    graph_node_count: int = 0
    graph_edge_count: int = 0
    graph_anchor_path: DefinitionPath | None = None
    graph_anchor_mode: str | None = None
    graph_candidate_count: int = 0
    generation_vector: dict[str, int] | None = None
    source_plans: tuple[SourceQueryPlan, ...] = ()
    lowering_strategy: str | None = None
    scan_required: bool = False
    scan_reason: str | None = None
    candidate_rows_read: int = 0
    cdef_blobs_decoded: int = 0
    python_verifications: int = 0
    pages_fetched: int = 0
    count_witness_reloads: int = 0
    count_collision_buckets: int = 0
    terminal_stop_reason: str | None = None
    lowering_diagnostics: dict[str, Any] | None = None
    containment_target_kind: ContainmentTargetKind | None = None
    containment_edges: ContainmentEdgePolicy | None = None
    containment_contains_ref: bool | None = None
    containment_source_scope: tuple[str, ...] = ()
    source_cuts: int = 0
    capture_rounds: int = 0
    demand_recaptures: int = 0
    instability_retries: int = 0

    def format(self) -> str:
        lines = [
            f"domain: {self.domain}",
            f"refresh: {self.refresh!r}",
            f"refresh action: {self.refresh_action}",
            f"store scans: {self.store_scan_count}",
            f"universe size: {self.universe_size if self.universe_size is not None else 'unknown'}",
            f"features: {len(self.selected_features)}",
            f"candidates: {self.candidate_count}",
            f"verified: {self.verified_count}",
            f"results: {self.result_count if self.result_count is not None else 'unknown'}",
        ]
        if self.fast_path is not None:
            lines.append(f"fast path: {self.fast_path}")
        if self.graph_node_count or self.graph_edge_count:
            lines.append(f"graph nodes: {self.graph_node_count}")
            lines.append(f"graph edges: {self.graph_edge_count}")
        if self.graph_anchor_path is not None:
            lines.append(f"graph anchor: {self.graph_anchor_path!s}")
        if self.graph_anchor_mode is not None:
            lines.append(f"graph anchor mode: {self.graph_anchor_mode}")
        if self.graph_candidate_count:
            lines.append(f"graph candidates: {self.graph_candidate_count}")
        if self.generation_vector:
            lines.append(f"generations: {dict(sorted(self.generation_vector.items()))!r}")
        if self.lowering_strategy is not None:
            lines.append(f"lowering: {self.lowering_strategy}")
        if self.scan_required:
            lines.append(f"scan required: {self.scan_reason or 'unknown'}")
        if self.containment_target_kind is not None:
            lines.append(f"containment target: {self.containment_target_kind}")
            lines.append(f"containment edges: {self.containment_edges}")
            lines.append(f"containment contains ref: {self.containment_contains_ref}")
            lines.append(f"containment sources: {self.containment_source_scope!r}")
        if self.candidate_rows_read or self.cdef_blobs_decoded or self.python_verifications:
            lines.append(f"candidate rows read: {self.candidate_rows_read}")
            lines.append(f"CDef blobs decoded: {self.cdef_blobs_decoded}")
            lines.append(f"Python verifications: {self.python_verifications}")
        if self.pages_fetched:
            lines.append(f"pages fetched: {self.pages_fetched}")
        if self.count_witness_reloads or self.count_collision_buckets:
            lines.append(f"count witness reloads: {self.count_witness_reloads}")
            lines.append(f"count collision buckets: {self.count_collision_buckets}")
        if self.terminal_stop_reason is not None:
            lines.append(f"terminal stop: {self.terminal_stop_reason}")
        if self.source_cuts or self.capture_rounds:
            lines.append(f"source cuts: {self.source_cuts}")
            lines.append(f"capture rounds: {self.capture_rounds}")
        if self.demand_recaptures:
            lines.append(f"demand recaptures: {self.demand_recaptures}")
        if self.instability_retries:
            lines.append(f"instability retries: {self.instability_retries}")
        for source in self.source_plans:
            result_count = source.result_count if source.result_count is not None else "unknown"
            lines.append(
                f"source {source.source_key} ({source.backend}): "
                f"generation={source.generation if source.generation is not None else 'unknown'}, "
                f"candidates={source.candidate_count}, verified={source.verified_count}, results={result_count}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True, slots=True)
class ResultUniverse:
    kind: Literal["definitions", "occurrences"]
    definitions: tuple[Any, ...] = ()
    occurrences: tuple[DefinitionOccurrence | ReferenceOccurrence, ...] = ()
    materializable: bool = False
    domain: str = "stored"
    replicas: dict[Any, tuple[Any, ...]] | None = None
    witnesses: tuple[Any, ...] = ()
    witness_complete: bool = False
    containment: ContainmentContext | None = None
    containment_witnesses: tuple[DefinitionOccurrence | ReferenceOccurrence, ...] = ()
    containment_private_witnesses: tuple[DefinitionOccurrence | ReferenceOccurrence, ...] = ()
    containment_private_replicas: dict[object, tuple[Any, ...]] | None = None
    containment_carrier: ContainmentCarrier = "target"


@dataclass(slots=True)
class QueryStats:
    refresh_action: str = "none"
    store_scan_count: int = 0
    universe_size: int | None = None
    selected_features: tuple[FeatureRequirement, ...] = ()
    posting_sizes: tuple[int, ...] = ()
    candidate_count: int = 0
    verified_count: int = 0
    result_count: int | None = None
    fast_path: str | None = None
    graph_node_count: int = 0
    graph_edge_count: int = 0
    graph_anchor_path: DefinitionPath | None = None
    graph_anchor_mode: str | None = None
    graph_candidate_count: int = 0
    generation_vector: dict[str, int] | None = None
    source_plans: tuple[SourceQueryPlan, ...] = ()
    lowering_strategy: str | None = None
    scan_required: bool = False
    scan_reason: str | None = None
    candidate_rows_read: int = 0
    cdef_blobs_decoded: int = 0
    python_verifications: int = 0
    pages_fetched: int = 0
    count_witness_reloads: int = 0
    count_collision_buckets: int = 0
    terminal_stop_reason: str | None = None
    lowering_diagnostics: dict[str, Any] | None = None
    source_cuts: int = 0
    capture_rounds: int = 0
    demand_recaptures: int = 0
    instability_retries: int = 0

    def explanation(self, *, domain: str, refresh: RefreshPolicy) -> QueryExplanation:
        return QueryExplanation(
            domain=domain,
            refresh=refresh,
            refresh_action=self.refresh_action,
            store_scan_count=self.store_scan_count,
            universe_size=self.universe_size,
            selected_features=self.selected_features,
            posting_sizes=self.posting_sizes,
            candidate_count=self.candidate_count,
            verified_count=self.verified_count,
            result_count=self.result_count,
            fast_path=self.fast_path,
            graph_node_count=self.graph_node_count,
            graph_edge_count=self.graph_edge_count,
            graph_anchor_path=self.graph_anchor_path,
            graph_anchor_mode=self.graph_anchor_mode,
            graph_candidate_count=self.graph_candidate_count,
            generation_vector=None if self.generation_vector is None else dict(self.generation_vector),
            source_plans=self.source_plans,
            lowering_strategy=self.lowering_strategy,
            scan_required=self.scan_required,
            scan_reason=self.scan_reason,
            candidate_rows_read=self.candidate_rows_read,
            cdef_blobs_decoded=self.cdef_blobs_decoded,
            python_verifications=self.python_verifications,
            pages_fetched=self.pages_fetched,
            count_witness_reloads=self.count_witness_reloads,
            count_collision_buckets=self.count_collision_buckets,
            terminal_stop_reason=self.terminal_stop_reason,
            lowering_diagnostics=None if self.lowering_diagnostics is None else dict(self.lowering_diagnostics),
            source_cuts=self.source_cuts,
            capture_rounds=self.capture_rounds,
            demand_recaptures=self.demand_recaptures,
            instability_retries=self.instability_retries,
        )
