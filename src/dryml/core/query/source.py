"""Private Query V3 producer inventory and coherent Store-cut capture.

This layer turns Repo, Store, and fixed members into detached identity evidence
without publishing a replacement query API. It snapshots Repo topology once per
capture and groups only identical authority-fence domains.
The terminal's capture also enforces outer scan/index controls and cumulative
identity admissions, retaining expenditure across replacement and retry without
changing authoritative records or introducing cross-terminal proof reuse.
Retained presentations associate opaque Query-owned requests with accepted targets;
this layer owns neither request synchronization nor cross-terminal completion.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Iterable

from ...paths import local_path_key
from ..object import Object
from ..definition import ConcreteDefinition
from ..errors import ParameterizationLimitError
from ..reference_values import ObjectRef, StateRef
from ..store.records import DefinitionRecord
from ..store.store import Store, StoreCapabilityError
from .authority import (
    CapturedStoreFacts,
    IdentityValue,
    _derive_identities,
    validate_captured_reference_authority,
)
from .identity import IdentitySet, SourceEvidence, identity_key


@dataclass(slots=True)
class _GeneratorWitnessBudget:
    limit: int | None
    visited: int = 0

    def consume(self) -> None:
        self.visited += 1
        if self.limit is not None and self.visited > self.limit:
            raise ParameterizationLimitError("generator query witness limit exceeded")


class InventoryCapabilityError(StoreCapabilityError):
    """Raised before a broad Query V3 capture lacks complete record coverage."""


class _SourceRecapture(Exception):
    """Signal that dependent terminal stages must restart after a new cut."""

    def __init__(self, invalidated: bool):
        super().__init__("Query V3 source cut changed.")
        self.invalidated = invalidated


@dataclass(frozen=True, slots=True)
class _MetadataReadFailure:
    """Record a failed read without retaining its exception or private message."""


@dataclass(frozen=True, slots=True)
class StoreSource:
    """Private Store producer retaining only its declared authority scope."""

    store: Store

    def __post_init__(self) -> None:
        if not isinstance(self.store, Store):
            raise TypeError("StoreSource requires a Store.")


@dataclass(frozen=True, slots=True)
class RepoSource:
    """Private Repo producer that adds captured live-cache identity knowledge."""

    repo: object
    weak: bool = True

    def __post_init__(self) -> None:
        if type(self.weak) is not bool:
            raise TypeError("RepoSource weak must be an exact bool.")


@dataclass(frozen=True, slots=True)
class RetainedRepoLookupSource:
    """Private V3 producer for retained Repo root-or-cache lookup APIs.

    Unlike :class:`RepoSource`, this producer intentionally excludes derived
    identities represented below stored roots unless independently cached.
    It preserves the historical ``Repo.get`` and ``Repo.__getitem__`` universe:
    all independently authoritative root families plus strong/weak cache
    CDefs. Incidental embedded definitions are not roots unless cached.
    """

    repo: object
    weak: bool = True

    def __post_init__(self) -> None:
        if type(self.weak) is not bool:
            raise TypeError("RetainedRepoLookupSource weak must be an exact bool.")


@dataclass(frozen=True, slots=True)
class _IndexPresentation:
    """Associate a producer or scoped obligation with its retained Store handle."""

    producer: object
    scope: object
    store: Store
    requests: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class _IndexTargetDescriptor:
    """Destination identity and compatibility, independent of query evidence."""

    destination: tuple
    authority_domain: str
    source_key: str
    dirty_domain: str | None
    semantics: tuple


@dataclass(frozen=True, slots=True)
class _IndexTarget:
    """One compatible backend with every producer/scope presentation retained."""

    descriptor: _IndexTargetDescriptor
    index: object
    presentations: tuple[_IndexPresentation, ...]


def _index_target_descriptor(store, index):
    """Validate configuration before coalescing or touching a backend destination."""

    from .model import CANONICAL_QUERY_SEMANTICS_VERSION, QueryIndexIncompatible
    from .sqlite import SQLiteQueryIndexConfig
    from .sqlite.index import SQLiteStoreQueryIndex

    authority = store.authority_fence_key()
    source_key = index.source_key
    expected_source = getattr(store, "catalog_key", lambda: source_key)()
    if source_key != expected_source:
        raise QueryIndexIncompatible("Query index source key disagrees with its Store presentation.")
    owner = getattr(index, "store", None)
    if owner is not None and owner.authority_fence_key() != authority:
        raise QueryIndexIncompatible("Query index authority domain disagrees with its Store presentation.")
    if not isinstance(index, SQLiteStoreQueryIndex):
        return _IndexTargetDescriptor(("instance", id(index)), authority, source_key, None, (type(index),))

    def normalized(path):
        return None if path is None else local_path_key(path)

    destination = normalized(index._connections.path)
    if normalized(index.path) != destination or normalized(index.config.path) != destination:
        raise QueryIndexIncompatible("Query index backend, manager, and configuration paths disagree.")
    if replace(index.config, path=index._connections.config.path) != index._connections.config:
        raise QueryIndexIncompatible("Query index backend and manager settings disagree.")
    configured = getattr(store, "_query_index_config", None)
    default_path = getattr(store, "query_index_path", None)
    if default_path is not None:
        expected_path = default_path if configured is None or configured.path is None else configured.path
        if normalized(expected_path) != destination:
            raise QueryIndexIncompatible("Query index destination disagrees with its configured Store path.")
    expected_config = configured
    if expected_config is None and default_path is not None:
        expected_config = SQLiteQueryIndexConfig(path=index.config.path)
    if expected_config is not None and replace(expected_config, path=index.config.path) != index.config:
        raise QueryIndexIncompatible("Query index settings disagree with its configured Store settings.")
    if index.canonical_version != CANONICAL_QUERY_SEMANTICS_VERSION:
        raise QueryIndexIncompatible("Query index canonical semantics disagree with its Store presentation.")
    dirty = normalized(index.dirty_path)
    if dirty != normalized(getattr(store, "query_index_dirty_path", index.dirty_path)):
        raise QueryIndexIncompatible("Query index dirty-token domain disagrees with its Store presentation.")
    settings = tuple(
        getattr(index.config, name) for name in index.config.__dataclass_fields__ if name != "path"
    )
    return _IndexTargetDescriptor(
        ("sqlite", destination), authority, source_key, dirty,
        (type(index), index.canonical_version, *settings),
    )


class SourceCapture:
    """One terminal-local registry of detached Store facts and cache inventory.

    ``source_cuts`` counts distinct authority-fence domains, while
    ``capture_rounds`` and ``demand_recaptures`` distinguish monotonic metadata
    demand growth from source instability retries owned by later execution work.
    Complete-combination proofs use sorted unique domains and fresh fact tokens;
    selective direct reads cannot stand in for complete coverage. Store knowledge,
    retained lookup roots, topology, and Repo cache tiers remain distinct.
    Effective outer controls govern every consumer. ``verified_count`` records
    cumulative identity admissions, while ``store_scan_count`` records inventory
    and started maintenance scans separately from page fetch/decode counters.
    Index presentations preserve retained producer/scope associations; compatible
    destinations coalesce only when a terminal actually requests index work.
    """

    def __init__(self, controls=None):
        """Retain effective terminal policies; standalone captures default to allow.

        ``controls`` is the outer IdentityQuery. Its immutable policy declarations
        govern every nested consumer of this capture. Admissions are graph-exact,
        cumulative across replacements, and independent of record/replica IO.
        """

        self.refresh_policy = getattr(controls, "refresh_policy", "auto")
        self.scan_policy_mode = getattr(controls, "scan_policy_mode", "allow")
        self.indexed_required = getattr(controls, "indexed_required", False)
        self.max_verify_limit = getattr(controls, "max_verify_limit", None)
        self.max_depth_limit = getattr(controls, "max_depth_limit", None)
        self.max_witness_limit = getattr(controls, "max_witness_limit", None)
        self._witness_budget = None
        self._admitted = set()
        self.verified_count = 0
        self.store_scan_count = 0
        self.refresh_action = "none"
        self.index_presentations: tuple[_IndexPresentation, ...] = ()
        self.scan_warning_emitted = False
        self._facts: dict[str, CapturedStoreFacts] = {}
        self._node_stored_keys = {}
        self._exact_states: dict[str, dict[str, StateRef | None]] = {}
        self._exact_stored_cdefs: dict[str, set[object]] = {}
        self._topologies: dict[int, tuple[Store, ...]] = {}
        self._cache_knowledge: dict[tuple[int, bool], IdentitySet] = {}
        self._retained_cache_cdefs: dict[tuple[int, bool], tuple] = {}
        self._cut_domains: set[str] = set()
        self._validated_combinations: set[tuple] = set()
        self.active = False
        self.source_cuts = 0
        self.capture_rounds = 0
        self.demand_recaptures = 0
        self.instability_retries = 0
        self.candidate_rows_read = 0
        self.cdef_blobs_decoded = 0
        self.pages_fetched = 0

    def resolve_index_targets(self, *, explicit: bool = False) -> tuple[_IndexTarget, ...]:
        """Open and validate every retained index presentation before maintenance.

        Args:
            explicit: Surface configured backend unavailability for an explicit
                cycle. Optional absent backends create no target in either mode.

        Returns:
            Destination-deduplicated targets retaining all producer/scope pairs.

        Raises:
            QueryIndexIncompatible: For conflicting destination ownership,
                settings, path agreement, or multiple destinations sharing one
                dirty-token domain, including duplicate handles inside a Repo.
            QueryIndexUnavailable: For explicitly configured backend unavailability.

        Side Effects:
            Calls Store-owned lazy openers without inventory or maintenance.
            Inspects persisted SQLite compatibility/ownership through read-only
            connections after checking all visible descriptors.
            Re-resolution checks current descriptors; it never revisits live Repo
            topology or stores cross-terminal completion/authority evidence.
        """

        from .model import QueryIndexIncompatible, QueryIndexUnavailable

        targets = {}
        presentations_by_destination = {}
        dirty_destinations = {}
        for presentation in self.index_presentations:
            store = presentation.store
            if getattr(store, "query_index_policy", None) in {"memory", "none"}:
                continue
            try:
                index = store.open_query_index()
            except QueryIndexUnavailable:
                if explicit:
                    raise
                continue
            if index is None:
                if explicit and getattr(store, "query_index_policy", None) == "sqlite":
                    raise QueryIndexUnavailable("The configured query index backend is unavailable.")
                continue
            descriptor = _index_target_descriptor(store, index)
            if descriptor.dirty_domain is not None:
                previous = dirty_destinations.setdefault(descriptor.dirty_domain, descriptor.destination)
                if previous != descriptor.destination:
                    raise QueryIndexIncompatible("Multiple query index destinations share one Store dirty-token domain.")
            existing = targets.get(descriptor.destination)
            if existing is not None and existing.descriptor != descriptor:
                raise QueryIndexIncompatible("Query index destination has incompatible Store presentations.")
            if existing is None:
                targets[descriptor.destination] = _IndexTarget(descriptor, index, ())
                presentations_by_destination[descriptor.destination] = []
            presentations_by_destination[descriptor.destination].append(presentation)
        resolved = tuple(
            replace(target, presentations=tuple(presentations_by_destination[destination]))
            for destination, target in targets.items()
        )
        from .sqlite.index import SQLiteStoreQueryIndex

        for target in resolved:
            if isinstance(target.index, SQLiteStoreQueryIndex):
                target.index._assert_sidecar_rebuildable()
        return resolved

    def merge_maintenance_stats(self, stats) -> None:
        """Charge started backend work, including failures, to this terminal.

        ``stats`` is a backend QueryStats for one call. Returns None and adds its
        started inventory count without identity admissions. Actions aggregate as
        rebuild over validate over waited over reused over none; no completion is
        inferred from a report, and waiters never receive owner inventory counts.
        Call in finally, after releasing backend resources. No Store IO occurs.
        """

        self.store_scan_count += stats.store_scan_count
        action = stats.refresh_action
        if action == "sqlite-rebuild":
            self.record_refresh_action("rebuild")
        elif action == "sqlite-validate":
            self.record_refresh_action("validate")

    def record_refresh_action(self, action: str) -> None:
        """Merge one actual maintenance/join/reuse action without charging IO.

        Args:
            action: One of none, reused, waited, validate, or rebuild.

        Returns:
            None; updates only this terminal's diagnostic action by precedence.

        Raises:
            ValueError: For an unsupported action.

        Side Effects:
            No Store reads, synchronization waits, or authority changes.
        """
        precedence = ("none", "reused", "waited", "validate", "rebuild")
        if precedence.index(action) > precedence.index(self.refresh_action):
            self.refresh_action = action

    def require_scan(self) -> None:
        """Enforce effective inventory/recovery policy before scan IO.

        Returns None when allowed. Raises QueryIndexUnavailable for indexed-only
        or QueryWouldScanError for forbidden scans. Warning mode emits at most
        one RuntimeWarning per terminal, without reading or changing authority.
        """

        from .model import QueryIndexUnavailable, QueryWouldScanError

        if self.indexed_required:
            raise QueryIndexUnavailable("Query V3 index coverage requires an authoritative inventory scan.")
        if self.scan_policy_mode == "forbid":
            raise QueryWouldScanError("Query V3 requires an authoritative inventory scan.")
        if self.scan_policy_mode == "warn" and not self.scan_warning_emitted:
            import warnings

            warnings.warn("Query V3 requires an authoritative inventory scan.", RuntimeWarning, stacklevel=4)
            self.scan_warning_emitted = True

    def reserve(self, values: Iterable[IdentityValue]) -> None:
        """Admit a detached candidate domain atomically before its verification.

        Values may repeat across replicas and stages. Budget exhaustion raises
        QueryVerifyBudgetExceeded before any new candidate is admitted. Replacement
        cuts clear reusable admissions, never the cumulative spent count.
        ``values`` supplies detached CDefs/ObjectRefs/StateRefs; unsupported values
        raise TypeError. Returns None and updates only terminal-local work state,
        without Store or index IO.
        """

        self.reserve_keys(identity_key(value) for value in values)

    def reserve_keys(self, keys):
        """Admit already-prepared domain-qualified keys without rederiving graphs."""
        from .model import QueryVerifyBudgetExceeded
        new = set(keys) - self._admitted
        if self.max_verify_limit is not None and self.verified_count + len(new) > self.max_verify_limit:
            raise QueryVerifyBudgetExceeded("Query V3 verification budget exceeded.")
        self._admitted.update(new)
        self.verified_count += len(new)

    def invalidate_admissions(self) -> None:
        """Discard reusable admissions after replacement without refunding work.

        Returns None, retaining cumulative counters and effective controls. This
        only changes terminal-local state and never reads or changes authority.
        """

        self._admitted.clear()

    def has_complete_store_cut(self, store: Store) -> bool:
        """Return whether this terminal already retains complete facts for a Store.

        ``store`` supplies its authority-fence domain. This performs no IO and
        returns a bool; paged stages must reuse existing complete facts rather
        than inspecting a newer selective cut.
        """

        return store.authority_fence_key() in self._facts

    def discard_index_attempt(self, stores: Iterable[Store]) -> None:
        """Discard abandoned selective proofs without refunding terminal work.

        ``stores`` are the attempted authority domains. Call only after releasing
        all retained readers/fences. Complete facts and cumulative IO/admission
        counters remain intact; subsequent inventory uses a fresh complete cut.
        This performs no Store IO and returns None.
        """

        self.invalidate_admissions()
        for store in stores:
            key = store.authority_fence_key()
            self._exact_states.pop(key, None)
            self._exact_stored_cdefs.pop(key, None)

    def inspect_index_main(self, store: Store) -> None:
        """Validate current main's direct-root completeness inside a held fence.

        ``store`` is a bounded producer whose authority fence the caller retains.
        Main inspection is coverage IO, not identity admission. Missing/malformed
        authoritative targets fail visibly; a valid main outside direct-root
        coverage raises QueryIndexUnavailable for post-release fallback/refusal.
        No inventory, payload read, or mutation occurs; returns None.
        """

        from .model import QueryIndexUnavailable

        self._record_cut(store.authority_fence_key())
        main = store.read_main_ref()
        if main is None:
            return
        record = store.read_definition_record(main.definition_digest)
        if record is None:
            raise InventoryCapabilityError("Main-reference authority targets a missing definition record.")
        if (type(store).read_stored_root_record is Store.read_stored_root_record
                or store.read_stored_root_record(main.definition_digest) is None):
            raise QueryIndexUnavailable("Current main-reference authority lacks direct stored-root authority proof.")

    def witness_budget(self):
        """Return the shared terminal-local GeneratorSelector witness counter.

        Lazily creates the counter with the effective outer witness limit. It
        performs no source IO; consuming it raises ParameterizationLimitError
        when the separate witness bound is exceeded.
        """

        if self._witness_budget is None:
            self._witness_budget = _GeneratorWitnessBudget(self.max_witness_limit)
        return self._witness_budget

    def _record_cut(self, key: str) -> None:
        """Account for each fenced read, including selective Store authority."""

        self.capture_rounds += 1
        if key not in self._cut_domains:
            self._cut_domains.add(key)
            self.source_cuts += 1

    def repo_stores(self, repo) -> tuple[Store, ...]:
        """Return the first connected topology retained for this terminal.

        ``repo`` owns topology admission and rejects inherited process handles
        before its configuration lock. This snapshots handles, not inventory;
        attachments after this call participate in the next terminal. Returns a
        tuple of Stores and propagates owner lifecycle/validation errors. Only the
        brief owner lease and terminal-local registry change; no Store IO occurs.
        """

        key = id(repo)
        if key not in self._topologies:
            with repo.retain_topology(allow_physical_duplicates=True):
                self._topologies[key] = tuple(repo.stores)
        return self._topologies[key]

    def cache_knowledge(self, repo, *, weak: bool) -> IdentitySet:
        """Snapshot one Repo cache tier without creating identities or Store reads."""

        key = id(repo), weak
        if key not in self._cache_knowledge:
            self._cache_knowledge[key] = IdentitySet(
                _derive_identities(_cache_values(repo, weak=weak))
            )
        return self._cache_knowledge[key]

    def capture_store(
        self, source: StoreSource | Store, *, metadata_scopes: frozenset[str] = frozenset()
    ) -> CapturedStoreFacts:
        """Capture inventory and requested metadata under one authority fence.

        Growing metadata demand replaces earlier facts with a complete new cut;
        V3 predicate evaluation therefore never reads live metadata after its
        identity inventory was captured. Effective scan/index controls reject
        prohibited work before inventory IO. Replacement discards reusable
        admissions without refunding cumulative terminal work.
        """

        store = source.store if isinstance(source, StoreSource) else source
        if not isinstance(store, Store):
            raise TypeError("Store capture requires a Store or StoreSource.")
        key = store.authority_fence_key()
        cached = self._facts.get(key)
        if cached is not None and metadata_scopes <= cached.metadata_scopes:
            if hasattr(cached, "cut") and store.authority_cut() != cached.cut:
                from .model import QueryError
                raise QueryError("Store changed during complete query capture; start a new query.")
            return cached
        if hasattr(store, "store_uuid"):
            return self._capture_node_facts(store, metadata_scopes)
        selective = cached is None and key in self._cut_domains
        self.require_scan()
        self._require_complete_inventory(store)
        requested_scopes = metadata_scopes | (cached.metadata_scopes if cached else frozenset())
        with store.authority_read_fence():
            self.store_scan_count += 1
            self._record_cut(key)
            if cached is not None:
                self.demand_recaptures += 1
            main = store.read_main_ref()
            definitions, stored_roots = store._validated_definition_root_inventory()
            definition_by_digest = {record.digest: record.definition for record in definitions}
            if main is not None:
                main_definition = definition_by_digest.get(main.definition_digest)
                if main_definition is None:
                    raise InventoryCapabilityError("Main-reference authority targets a missing definition record.")
            else:
                main_definition = None
            facts = CapturedStoreFacts.build(
                definitions=definitions,
                stored_roots=stored_roots,
                declarations=store.iter_declaration_records(),
                state_refs=store.iter_state_ref_records(),
                object_aliases=store.iter_object_alias_records(),
                state_aliases=store.iter_state_alias_records(),
                main_definition=main_definition,
                source=SourceEvidence.from_source(store.authority_fence_key()),
            )
            if requested_scopes:
                facts = facts.with_metadata(
                    self._capture_metadata(store, facts, requested_scopes),
                    scopes=requested_scopes,
                )
        exact = self._exact_states.pop(key, {})
        exact_cdefs = self._exact_stored_cdefs.pop(key, set())
        if cached is not None or selective:
            self.invalidate_admissions()
        exact_changed = False
        if exact:
            current_states = {record.digest: record.state_ref for record in facts.state_refs}
            exact_changed = any(
                previous != current_states.get(digest) for digest, previous in exact.items()
            )
        self._facts[key] = facts
        if self.active and (exact_changed or any(
            not facts.is_stored(identity._value) for identity in exact_cdefs
        )):
            raise _SourceRecapture(True)
        if cached is not None and self.active:
            raise _SourceRecapture(_cut_invalidated(cached, facts))
        return facts

    def capture_stores(
        self, sources: Iterable[StoreSource | Store], *, metadata_scopes: frozenset[str] = frozenset()
    ) -> tuple[CapturedStoreFacts, ...]:
        """Capture and validate the complete selected authority combination.

        Duplicate fence domains share facts. Sorted domain/token keys reuse only
        successful terminal-local proofs; demand recapture creates new tokens
        even without authority mutation. Conflicting authoritative ObjectIds raise
        QueryError before filtering; embedded references are not root authority.
        Distinct Zip transactions are always distinct domains.
        """

        selected = {}
        for source in sources:
            store = source.store if isinstance(source, StoreSource) else source
            key = store.authority_fence_key()
            if key not in selected:
                selected[key] = self.capture_store(store, metadata_scopes=metadata_scopes)
        facts = tuple(selected.values())
        combination = tuple(
            (key, "complete", selected[key].capture_token) for key in sorted(selected)
        )
        if combination not in self._validated_combinations:
            validate_captured_reference_authority(facts)
            self._validated_combinations.add(combination)
        return facts

    def read_exact_state(self, source: StoreSource | Store, target: StateRef) -> StateRef | None:
        """Read one complete exact StateRef without requiring broad inventory.

        A digest-addressed positive proves independent saved-state membership
        for this source only. A miss does not exclude embedded or projected
        knowledge: callers need complete capture before a negative live answer.
        This selective proof is separate from complete captured inventory.
        The target is admitted to the effective verification budget before the
        direct check, including a miss; replicas share the same identity charge.
        Budget exhaustion raises QueryVerifyBudgetExceeded before record IO.
        """

        store = source.store if isinstance(source, StoreSource) else source
        if not isinstance(store, Store):
            raise TypeError("Exact state lookup requires a Store or StoreSource.")
        if not isinstance(target, StateRef):
            raise TypeError("Exact state lookup requires a StateRef.")
        key = store.authority_fence_key()
        digest = target.digest()
        self.reserve((target,))
        if key in self._facts:
            record = next(
                (item for item in self._facts[key].state_refs if item.digest == digest), None
            )
        elif digest in self._exact_states.get(key, {}):
            value = self._exact_states[key][digest]
            if value is not None and value != target:
                raise InventoryCapabilityError("Exact StateRef authority is incompatible with its digest.")
            return value
        else:
            with store.authority_read_fence():
                self._record_cut(key)
                record = store.read_state_ref_record(digest)
        if record is None:
            if key not in self._facts:
                self._exact_states.setdefault(key, {})[digest] = None
            return None
        if record.state_ref != target:
            raise InventoryCapabilityError("Exact StateRef authority is incompatible with its digest.")
        if key not in self._facts:
            self._exact_states.setdefault(key, {})[digest] = record.state_ref
        return record.state_ref

    def read_exact_stored_cdef(
        self, source: StoreSource | Store, target: ConcreteDefinition,
    ) -> ConcreteDefinition | None:
        """Use complete direct root authority for a positive exact CDef hit.

        Args:
            source: Store authority to inspect under its read fence.
            target: Graph-exact CDef whose direct root marker is requested.

        Returns:
            The validated stored root or ``None`` if direct proof is unavailable.

        Raises:
            TypeError: If source or target has an unsupported type.
            InventoryCapabilityError: If direct marker and definition disagree.
            QueryVerifyBudgetExceeded: Before a new target exceeds the effective
                cumulative admission budget, without direct record IO.

        A missing direct marker is not a negative V3 answer: declarations,
        snapshots, aliases, and main references may establish stored authority,
        so callers must fall back to complete inventory. Backends inheriting the
        optional iterator-based direct reader also use that fallback. This
        method reads only direct authority and never changes the Store.
        """

        store = source.store if isinstance(source, StoreSource) else source
        if not isinstance(store, Store) or not isinstance(target, ConcreteDefinition):
            raise TypeError("Exact stored CDef lookup requires a Store and CDef.")
        if type(store).read_stored_root_record is Store.read_stored_root_record:
            return None
        self.reserve((target,))
        key = store.authority_fence_key()
        if key in self._facts:
            return target if self._facts[key].is_stored(target) else None
        digest = DefinitionRecord(target).digest
        with store.authority_read_fence():
            self._record_cut(key)
            marker = store.read_stored_root_record(digest)
            if marker is None:
                return None
            record = store.read_definition_record(digest)
            if record is None or not record.definition.graph_equal(target):
                raise InventoryCapabilityError("Exact stored CDef authority is incompatible.")
        self._exact_stored_cdefs.setdefault(key, set()).add(identity_key(record.definition))
        return record.definition

    def has_exact_stored_cdef(self, source: StoreSource | RepoSource, value: ConcreteDefinition) -> bool:
        """Return whether this cut already verified an exact stored root.

        Args:
            source: Normalized Store or Repo scope for prior direct reads.
            value: Candidate CDef to check by complete graph identity.

        Returns:
            Whether a matching positive marker was verified in this terminal.
            Repo scopes require a nonempty topology and proof from every Store.

        Side Effects:
            None. This does not reread or change Store authority.
        """

        if isinstance(source, RepoSource):
            return self.has_exact_stored_cdef_for_repo(source.repo, value)
        return identity_key(value) in self._exact_stored_cdefs.get(
            source.store.authority_fence_key(), set(),
        )

    def has_exact_stored_cdef_for_repo(self, repo, value: ConcreteDefinition) -> bool:
        """Return whether every selected Store directly proved this stored CDef.

        Args:
            repo: Repo whose first terminal topology was retained.
            value: Complete CDef identity selected for stored membership.

        Returns:
            True only when at least one Store is connected and every selected
            authority fence has a verified positive root marker.

        Side Effects:
            None. This checks terminal-local evidence without Store access.
        """

        stores = self.repo_stores(repo)
        return bool(stores) and all(
            identity_key(value) in self._exact_stored_cdefs.get(
                store.authority_fence_key(), set(),
            ) for store in stores
        )

    def capture_repo(
        self, source: RepoSource, *, metadata_scopes: frozenset[str] = frozenset()
    ) -> IdentitySet:
        """Capture Repo Store topology and retained cache identities once."""

        if not isinstance(source, RepoSource):
            raise TypeError("Repo capture requires a RepoSource.")
        repo = source.repo
        retain_topology = getattr(repo, "retain_topology", None)
        if not callable(retain_topology):
            raise TypeError("RepoSource requires a Repo-like topology producer.")
        with retain_topology(allow_physical_duplicates=True):
            stores = self.repo_stores(repo)
            facts = self.capture_stores(stores, metadata_scopes=metadata_scopes)
            cached = self.cache_knowledge(repo, weak=source.weak)
        members = [
            (value, facts_item.source)
            for facts_item in facts
            for value in facts_item.knowledge()
        ]
        members.extend(
            (value, SourceEvidence.from_source(repo))
            for value in cached
        )
        return IdentitySet(members)

    def capture_retained_lookup_repo(
        self,
        source: RetainedRepoLookupSource,
        *,
        exact_cdef: ConcreteDefinition | None = None,
    ) -> IdentitySet:
        """Capture the historical stored-root plus cache lookup candidate set.

        An exact CDef uses each Store's direct root marker reader, preserving
        the old exact lookup path's avoidance of broad definition hydration.
        Structural selections include declared, saved, alias and main roots,
        but never incidental embedded CDefs without independent authority.
        """

        if not isinstance(source, RetainedRepoLookupSource):
            raise TypeError("Retained Repo lookup capture requires its dedicated source.")
        repo = source.repo
        retain_topology = getattr(repo, "retain_topology", None)
        if not callable(retain_topology):
            raise TypeError("Retained Repo lookup source requires a Repo-like producer.")
        members = []
        with retain_topology():
            seen = set()
            for store in self.repo_stores(repo):
                key = store.authority_fence_key()
                if key in seen:
                    continue
                seen.add(key)
                evidence = SourceEvidence.from_source(key)
                if exact_cdef is not None:
                    value = self.read_exact_stored_cdef(store, exact_cdef)
                    if value is not None:
                        members.append((value, evidence))
                        continue
                facts = self.capture_store(store)
                for value in facts._authoritative_root_definitions():
                    if exact_cdef is None or value.graph_equal(exact_cdef):
                        members.append((value, evidence))
            if exact_cdef is None:
                self.capture_stores(self.repo_stores(repo))
            cache_key = id(repo), source.weak
            if cache_key not in self._retained_cache_cdefs:
                self._retained_cache_cdefs[cache_key] = _retained_lookup_cache_cdefs(repo, weak=source.weak)
            strong, weak = self._retained_cache_cdefs[cache_key]
            members.extend(
                (value, SourceEvidence.from_source(repo))
                for value in (*strong, *weak)
            )
        return IdentitySet(members)

    @staticmethod
    def _capture_metadata(store: Store, facts: CapturedStoreFacts, scopes: frozenset[str]):
        """Detach requested fields for eligible targets while the fence is held."""

        holders = {identity_key(record.object_ref) for record in facts.declarations}
        stored_states = {identity_key(record.state_ref): record for record in facts.state_refs}
        for record in facts.state_refs:
            root = record.state_ref.object
            holders.add(identity_key(root))
            for path in root.objects:
                holders.add(identity_key(root.at(path)))

        snapshot_cache = {}

        def snapshot_for(record):
            if record.digest not in snapshot_cache:
                try:
                    snapshot_cache[record.digest] = store.read_snapshot_metadata(record.digest)
                except Exception:
                    snapshot_cache[record.digest] = _MetadataReadFailure()
            return snapshot_cache[record.digest]

        fallback_lineages = None

        def snapshot_lineages():
            nonlocal fallback_lineages
            if fallback_lineages is None:
                fallback_lineages = {}
                for record in facts.state_refs:
                    snapshot = snapshot_for(record)
                    if isinstance(snapshot, _MetadataReadFailure):
                        raise StoreCapabilityError("Query V3 snapshot lineage read failed.")
                    if snapshot is not None:
                        for lineage in snapshot.lineages.values():
                            fallback_lineages.setdefault(identity_key(lineage.object_ref), lineage)
            return fallback_lineages

        captured = []
        for value in facts.knowledge():
            key = identity_key(value)
            if isinstance(value, ObjectRef) and key in holders:
                if "object" in scopes:
                    try:
                        captured.append((key, "object", deepcopy(store.read_metadata(value))))
                    except Exception:
                        captured.append((key, "object", _MetadataReadFailure()))
                if "lineage" in scopes:
                    try:
                        lineage = store.read_lineage_metadata(value)
                        if lineage is None:
                            lineage = snapshot_lineages().get(key)
                        from ..metadata import LineageMetadata

                        captured.append((
                            key, "lineage",
                            LineageMetadata(value, "unknown", None) if lineage is None else lineage,
                        ))
                    except Exception:
                        captured.append((key, "lineage", _MetadataReadFailure()))
            if not isinstance(value, StateRef) or key not in stored_states:
                continue
            if "state" in scopes:
                try:
                    captured.append((key, "state", deepcopy(store.read_metadata(value))))
                except Exception:
                    captured.append((key, "state", _MetadataReadFailure()))
            if "snapshot" in scopes:
                snapshot = snapshot_for(stored_states[key])
                captured.append((
                    key, "snapshot",
                    snapshot if snapshot is not None and not isinstance(snapshot, _MetadataReadFailure)
                    and snapshot.state_ref == value else _MetadataReadFailure(),
                ))
        return tuple(captured)

    def _capture_node_facts(self, store, metadata_scopes):
        """Use the same current-format consumer for complete algebra/scope facts."""
        from .query import IdentityQuery
        query = IdentityQuery.from_store(store)
        before = store.authority_cut()
        previous = self._facts.get(store.authority_fence_key())
        if previous is not None:
            self.demand_recaptures += 1
            if previous.cut != before:
                from .model import QueryError
                raise QueryError("Store changed during query metadata demand; start a new query.")
            metadata_scopes = metadata_scopes | previous.metadata_scopes
        entries = dict(iter_node_query(query, self, (store,), admit=False))
        facts = _NodeFacts(IdentitySet._from_entries(entries, bounded=False), self._node_stored_keys.get(store.authority_fence_key(), set()),
                           SourceEvidence.from_source(store.authority_fence_key()), metadata_scopes)
        facts.cut = before
        from ..node import Node
        metadata_cache = {}
        for key, (value, _) in entries.items():
            if not isinstance(value, Node) or key.kind == "cdef":
                continue
            with store._metadata_read():
                for scope in metadata_scopes:
                    area = {"object": "current_object", "state": "current_state", "lineage": "lineage", "snapshot": "capture"}[scope]
                    try:
                        eligible = store._holds_node_metadata(value, area)
                    except (TypeError, ValueError):
                        eligible = False
                    if not eligible:
                        continue
                    digest = value.object_projection_digest() if scope in {"object", "lineage"} else value.exact_digest()
                    cached_key = (scope, digest)
                    if cached_key in metadata_cache:
                        facts.metadata[identity_key(value), scope] = metadata_cache[cached_key]
                        continue
                    if scope == "snapshot":
                        data = store._read_node_capture(value)
                    elif scope == "lineage":
                        from ..metadata import NodeLineageMetadata, timestamp_from_seconds
                        import json
                        record = store._read_node_metadata(value, "lineage")
                        row = None if record is None else json.loads(record.content)
                        data = NodeLineageMetadata(value.object_view(), "unknown" if row is None else row["creation_status"],
                                                   None if row is None else timestamp_from_seconds(row["created_at"]))
                    else:
                        try:
                            data = store.read_metadata(value, scope=scope)
                        except Exception:
                            data = _MetadataReadFailure()
                    facts.metadata[identity_key(value), scope] = data
                    metadata_cache[cached_key] = data
        if store.authority_cut() != before:
            from .model import QueryError
            raise QueryError("Store changed during complete query capture.")
        self._facts[store.authority_fence_key()] = facts
        if previous is not None:
            self.invalidate_admissions()
            if self.active:
                from .metadata import _lineage_projection, _snapshot_projection
                from ..metadata import encode_metadata_mapping

                def comparable(scope, value):
                    if value is None or isinstance(value, _MetadataReadFailure):
                        return value
                    if scope == "lineage":
                        return _lineage_projection(value)
                    if scope == "snapshot":
                        return _snapshot_projection(value)
                    return encode_metadata_mapping(value)

                changed = any(type(value) is not type(facts.metadata.get(key))
                              or comparable(key[1], value) != comparable(key[1], facts.metadata.get(key))
                              for key, value in previous.metadata.items())
                raise _SourceRecapture(changed)
        return facts

    @staticmethod
    def _require_complete_inventory(store: Store) -> None:
        capabilities = store._query_v3_inventory_capabilities()
        missing = [
            name for name in capabilities.__dataclass_fields__
            if getattr(capabilities, name) == "unsupported"
        ]
        if missing:
            raise InventoryCapabilityError(
                "Store cannot provide complete Query V3 inventory for required authority families."
            )


def _cut_invalidated(old: CapturedStoreFacts, new: CapturedStoreFacts) -> bool:
    """Compare earlier authoritative facts and mutable demand without rendering data."""

    def record_keys(facts):
        return (
            tuple(record.digest for record in facts.definitions),
            tuple(record.definition_digest for record in facts.stored_roots),
            tuple(record.digest for record in facts.declarations),
            tuple(record.digest for record in facts.state_refs),
            tuple((record.alias, record.object_ref.digest()) for record in facts.object_aliases),
            tuple((record.alias, record.object_ref.digest(), record.state_ref_digest)
                  for record in facts.state_aliases),
            None if facts.main_definition is None else facts.main_definition.graph_hash(),
        )

    if record_keys(old) != record_keys(new):
        return True
    from ..metadata import encode_metadata_mapping

    def comparable(value):
        if isinstance(value, dict):
            return encode_metadata_mapping(value)
        return value

    for key, scope, value in old.metadata:
        try:
            current = new.captured_metadata(key._value, scope)
        except KeyError:
            return True
        if type(value) is not type(current) or comparable(value) != comparable(current):
            return True
    return False


def _cache_values(repo, *, weak: bool) -> tuple[IdentityValue, ...]:
    """Read retained cache facts without constructing, saving, or capturing state."""

    caches = (repo.strong_obj_cache, repo.weak_obj_cache) if weak else (repo.strong_obj_cache,)
    values: list[IdentityValue] = []
    for cache in caches:
        for _, value in tuple(cache.items()):
            if not isinstance(value, Object):
                continue
            values.append(value.definition)
            reference = value.object_ref
            if isinstance(reference, ObjectRef):
                from ..node import _as_node_view
                values.append(_as_node_view(value.definition).with_realization(identities=reference.objects))
            receipt = value.last_state_ref
            from ..node import Node
            if isinstance(receipt, Node) or isinstance(receipt, StateRef) and receipt.object == reference:
                values.append(receipt)
    return tuple(values)


def _retained_lookup_cache_cdefs(
    repo, *, weak: bool,
) -> tuple[tuple[ConcreteDefinition, ...], tuple[ConcreteDefinition, ...]]:
    """Return explicitly retained strong and weak cache CDefs without derivation."""

    strong = tuple(
        obj.definition
        for _, obj in tuple(repo.strong_obj_cache.items())
        if isinstance(obj, Object)
    )
    if not weak:
        return strong, ()
    weak_values = tuple(
        obj.definition
        for _, obj in tuple(repo.weak_obj_cache.items())
        if isinstance(obj, Object)
    )
    return strong, weak_values


def node_query_stores(query, capture):
    """Select current-format producers without opening a Store or query index.

    None denotes a fixed/algebra/relationship source handled by the ordinary
    composition layer. Store handles remain terminal-owned, never cache facts.
    """
    if isinstance(query.source, StoreSource):
        stores = (query.source.store,)
    elif isinstance(query.source, (RepoSource, RetainedRepoLookupSource)):
        stores = capture.repo_stores(query.source.repo)
    else:
        return None
    return stores if all(hasattr(store, "store_uuid") for store in stores) else None


class _NodeFacts:
    """Terminal-local complete evidence for the existing algebra/scope consumer."""

    def __init__(self, members, stored, source, metadata_scopes):
        self.members, self.stored, self.source = members, frozenset(stored), source
        self.metadata_scopes, self.metadata, self.capture_token = metadata_scopes, {}, object()
        self.node_aliases = {}
        from ..node import Node, _all_nodes
        self._reference_projections = {}
        for value in members:
            if isinstance(value, Node):
                for node in _all_nodes(value):
                    if node.object_id is not None:
                        self._reference_projections[node.object_id] = node.object_view()

    def knowledge(self):
        return self.members

    def is_stored(self, value):
        key = identity_key(value)
        return any(candidate.digest == key.digest and candidate._value.graph_equal(value)
                   for candidate in self.stored) if isinstance(value, ConcreteDefinition) else key in self.stored

    def captured_metadata(self, value, scope):
        return self.metadata[identity_key(value), scope]

    def holds_metadata(self, value):
        """Report captured object/saved holder eligibility without live Store IO."""
        from ..node import Node
        if not isinstance(value, Node):
            return False
        return ((value.object_id is not None and value.object_id in self._reference_projections)
                or self.is_stored(value))

    def _authoritative_root_definitions(self):
        return (value for value in self.members if identity_key(value).kind == "cdef" and self.is_stored(value))


def _node_index_token(index, con, identity, cut):
    """Check every coverage family, including guards for empty/count domains."""
    from .model import QueryIndexDirty
    index._store_certificate(con, cut)
    if index._is_dirty() or identity != index._connections._path_identity():
        raise QueryIndexDirty("Query index changed or has pending dirty evidence.")
    return (cut, identity, _index_target_descriptor(index.store, index), index.store.query_index_policy,
            tuple(con.execute("SELECT * FROM store_projection_cut")),
            tuple(con.execute("SELECT * FROM store_projection_coverage ORDER BY family")),
            tuple(con.execute("SELECT index_uuid,generation,schema_version FROM catalog_state")))


class _NodePage:
    """Detached bounded bytes and operation-local shared block interpretation.

    All reads occur inside short cuts; expand() runs only after their release.
    No preparation, callback, open resource or borrowed payload enters a cut.
    """

    def __init__(self, stores, tokens, indexes):
        from ..store.authority import _DetachedRead
        self.stores, self.tokens, self.indexes = stores, tokens, indexes
        self.witnesses = {s.authority_fence_key(): _DetachedRead(s, tokens[s.authority_fence_key()][0]) for s in stores}
        self.records, self.blocks, self.anchors, self.roots, self.preparations = {}, {}, {}, {}, {}
        self.root_kinds = {}
        self.available, self.grants, self.presence = {}, {}, {}
        self.metadata, self.names, self.memberships, self.scoped_roots = {}, {}, {}, {}
        self.state_memberships = {}

    def read(self, store, path, reader, required=False):
        import os
        from ..store import authority as a
        key = store.authority_fence_key(), path
        if key not in self.records:
            raw = a.raw(store, path)
            self.witnesses[key[0]].file(os.path.join(store.base_dir, path), raw)
            if raw is None:
                record = None
            else:
                from ..store.records import DeclarationAnchorRecord, RetiredDeclarationAnchorRecord, StoreRecordError
                try:
                    record = reader.from_bytes(raw)
                except StoreRecordError:
                    if reader is not DeclarationAnchorRecord:
                        raise
                    record = RetiredDeclarationAnchorRecord.from_bytes(raw)
            self.records[key] = record
        record = self.records[key]
        if required and record is None:
            raise StoreCapabilityError("Missing required query authority: " + path)
        return record

    def root(self, store, record):
        from ..store.records import NodeBlockRecord
        key = store.authority_fence_key(), record.block_digest
        path = store._relative(store._node_path("definitions/blocks", record.block_digest))
        block = self.read(store, path, NodeBlockRecord, True)
        if block.digest != record.block_digest:
            raise InventoryCapabilityError("Query block address disagrees with authenticated bytes.")
        if key not in self.blocks:
            self.blocks[key] = block, {}
        self.blocks[key][1][record.view_bytes] = record
        self.root_kinds.setdefault(key, {})[record.root_kind, record.view_bytes] = record

    def declarations(self, identities):
        from ..store.records import DeclarationAnchorRecord, DeclarationRequirementRecord
        for identity in dict.fromkeys(identities):
            for store in self.stores:
                key = store._anchor_key(identity)
                anchor = self.read(store, store._relative(store._node_path("declarations/anchors", key)), DeclarationAnchorRecord)
                required = self.read(store, store._relative(store._node_path("declarations/required", key)), DeclarationRequirementRecord)
                if anchor is None and required is None:
                    continue
                if (anchor is None or required is None or anchor.object_id != identity or required.object_id != identity
                        or anchor.projection.digest != required.projection_digest):
                    raise InventoryCapabilityError("Missing or incompatible required declaration anchor.")
                self.root(store, anchor.projection)
                self.anchors[store.authority_fence_key(), identity] = anchor

    def expand(self, capture, owners, *, max_cached_bytes):
        from ..node_codec import _decode_node_graph_views
        from ..node_views import OBJECT_PROJECTION_VIEW, view_equal
        from .model import QueryError
        for (domain, digest), (block, selected) in self.blocks.items():
            # Owners contain pure metadata only. Their lifetime is this terminal;
            # fresh bytes and final witnesses are mandatory even for a reuse hit.
            key = block.content, tuple(selected)
            previous = owners.get(key)
            cacheable = False
            if previous is None:
                packet = block.blocks
                packet["views"] = [r.view for r in selected.values()]
                roots, prepared = _decode_node_graph_views(packet, with_owner=True)
                previous = roots, prepared, set(), set()
                capture.cdef_blobs_decoded += 1
                # Bound detached pure ownership, rather than accumulating every
                # page's graphs. Oversized blocks are never retained for reuse.
                owners.clear()
                from .._content_facts import _node_block_owner_size
                if _node_block_owner_size(block.content, packet) <= max_cached_bytes:
                    cacheable = True
            roots, prepared, validated_views, validated_slots = previous
            self.preparations[domain, digest] = prepared
            by_view = dict(zip(selected, roots))
            from .authority import validate_node_root_views
            validate_node_root_views(((record, by_view[record.view_bytes])
                                      for record in self.root_kinds[domain, digest].values()), prepared,
                                     validated_views=validated_views, validated_slots=validated_slots)
            if cacheable:
                owners[key] = previous
            for record, root in zip(selected.values(), roots):
                self.roots[domain, digest, record.view_bytes] = root
        projections = {}
        for (domain, identity), anchor in self.anchors.items():
            root = self.roots[domain, anchor.projection.block_digest, anchor.projection.view_bytes]
            if root.object_id != identity:
                raise InventoryCapabilityError("Declaration projection names another ObjectId.")
            old = projections.setdefault(identity, root)
            if not view_equal(old, root, OBJECT_PROJECTION_VIEW):
                raise QueryError("Query object-reference authority conflicts across connected Stores.")

    def value(self, store, record):
        return self.roots[store.authority_fence_key(), record.block_digest, record.view_bytes]

    def preparation(self, store, record):
        return self.preparations[store.authority_fence_key(), record.block_digest]

    def validate(self):
        import os
        from .model import QueryError
        for store in self.stores:
            domain = store.authority_fence_key()
            with store.authority_read_fence():
                from ..store.authority import StoreCutChanged
                try:
                    self.witnesses[domain].validate()
                except StoreCutChanged as error:
                    raise QueryError("Store changed during query continuation; start a new query.") from error
                for (owner, key), present in self.presence.items():
                    if owner == domain and os.path.isdir(store.state_data_path(key)) != present:
                        raise QueryError("State availability changed during query continuation.")
                index = self.indexes.get(domain)
                if index is not None:
                    with index._connections._read_lease() as (con, identity):
                        con.execute("BEGIN")
                        try:
                            if _node_index_token(index, con, identity, store.authority_cut()) != self.tokens[domain]:
                                raise QueryError("Query continuation changed; start a new query.")
                        finally:
                            con.execute("ROLLBACK")


def iter_node_query(query, capture, stores, *, admit=True):
    """Execute the ordinary Repo/Store consumer over current-format authority.

    Direct full-key hit/miss is index-free. Other domains page certified derived
    candidates, capture bounded immutable bytes, release cuts for preparation and
    exact residuals, then revalidate before output. Federation has per-Store cuts;
    no global filesystem/SQLite transaction or cache-based authority is claimed.
    ``admit=False`` is reserved for terminal-owned authority scope capture, whose
    holder identities are evidence rather than producer/residual candidates.
    """
    import heapq
    import json
    import os
    from contextlib import ExitStack
    from ..node import require_saved
    from ..state_values import LocalState
    from ..store import records as r
    from .identity import identity_key, IdentityKey, IdentityEvidence
    from .model import QueryError, QueryDomainError, QueryIndexDirty, QueryIndexUnavailable, QueryStats
    from .node_pattern import _PreparedPredicate
    from ..selector import QueryPredicate
    from .sqlite.lowering import _store_discovery_relation

    producers = {s.authority_fence_key(): s for s in stores}
    scope_stores = {}
    for item in query.restrictions:
        if item.scope is not None and item.kind in {"stored", "alias", "metadata"}:
            scope_stores[id(item)] = ((item.scope.store,) if isinstance(item.scope, StoreSource)
                                     else capture.repo_stores(item.scope.repo))
    domains = {s.authority_fence_key(): s for s in (*stores, *(s for values in scope_stores.values() for s in values))}
    stores = tuple(domains[key] for key in sorted(domains))
    local = any(item.kind == "kind" and item.value == "local_state" for item in query.restrictions)
    exact = next((item.value for item in query.restrictions if item.kind == "state_key"), None)
    # Ordinary recipe selectors leave omitted facets unconstrained. Only the
    # retained get/item adapter has an exact CDef-key candidate universe.
    recipe = None
    if isinstance(query.source, RetainedRepoLookupSource):
        selectors = [item.value for item in query.restrictions if item.kind == "sel"]
        if len(selectors) == 1 and isinstance(selectors[0], ConcreteDefinition):
            from ..node import _as_node_view
            recipe = _as_node_view(selectors[0])
    direct = exact is not None or recipe is not None
    if not local and any(item.kind in {"origin", "compatible", "state_key", "available", "retained"} for item in query.restrictions):
        raise QueryDomainError("StateKey, origin, compatibility, availability and retention require the local-state domain.")
    if exact is not None and not local:
        raise QueryError("Full StateKey selection requires the local-state domain.")
    if exact is not None:
        capture.reserve_keys((IdentityKey("local_state", exact.cdef_digest + ":" + exact.codec_hash, exact),))
    if local and any(item.kind in {"object_id", "namespace", "contains", "alias", "metadata", "cached"} for item in query.restrictions):
        raise QueryDomainError("LocalState has no donor identity or saved-graph metadata; request an explicit occurrence relationship.")
    prepared = {}
    if local:
        for item in query.restrictions:
            if item.kind == "sel":
                predicate = item.value if isinstance(item.value, QueryPredicate) else QueryPredicate(item.value)
                prepared[id(item)] = _PreparedPredicate(predicate, domain="local_state")
    indexes, tokens, owners = {}, {}, {}
    facility = getattr(getattr(query.source, "repo", None), "_validation_evidence", None)
    owner_budget = 1024 * 1024 if facility is None else min(facility.max_bytes, facility.max_entry_bytes)
    cached = capture.cache_knowledge(query.source.repo, weak=query.source.weak) if not local and isinstance(query.source, (RepoSource, RetainedRepoLookupSource)) else IdentitySet()
    cached_only = any(item.kind == "cached" and isinstance(item.scope, RepoSource)
                      and isinstance(query.source, RepoSource) and item.scope.repo is query.source.repo
                      for item in query.restrictions) and all(item.kind not in {"stored", "alias", "metadata"} for item in query.restrictions)
    if cached_only:
        for key in sorted(cached._entries, key=lambda key: key.sort_key):
            value = cached._entries[key][0]
            evidence = IdentityEvidence(frozenset((query._source_evidence(query.source.repo),)))
            capture.reserve((value,))
            if all(query._matches(item, value, evidence, capture, capture.witness_budget()) for item in query.restrictions):
                yield key, (value, evidence)
        return
    if not direct:
        targets = capture.resolve_index_targets()
        by_store = {id(item.store): target.index for target in targets for item in target.presentations}
        for store in stores:
            index = by_store.get(id(store))
            if index is not None:
                state = index.status().state
                if state != "ready":
                    if state == "incompatible":
                        index._assert_sidecar_rebuildable()
                    if capture.refresh_policy is False:
                        error = QueryIndexDirty if state == "dirty" else QueryIndexUnavailable
                        raise error("Current Store query coverage is not ready; refresh(False) forbids maintenance.")
                    capture.require_scan()
                    stats = QueryStats()
                    try:
                        index.refresh("auto", stats=stats, scan_admission=capture.require_scan)
                    finally:
                        capture.merge_maintenance_stats(stats)
                indexes[store.authority_fence_key()] = index
            else:
                capture.require_scan()
    for store in stores:
        domain = store.authority_fence_key()
        with store.authority_read_fence():
            cut = store.authority_cut()
            index = indexes.get(domain)
            if index is None:
                tokens[domain] = (cut,)
            else:
                with index._connections._read_lease() as (con, identity):
                    con.execute("BEGIN")
                    try:
                        tokens[domain] = _node_index_token(index, con, identity, cut)
                    finally:
                        con.execute("ROLLBACK")
            capture._record_cut(domain)
            if not hasattr(capture, "_node_query_tokens"):
                capture._node_query_tokens = {}
            previous = capture._node_query_tokens.setdefault(domain, tokens[domain])
            if previous != tokens[domain]:
                raise QueryError("Store changed during query evaluation; start a new query.")

    def topology():
        sources = (query.source, *(item.scope for item in query.restrictions))
        for source in sources:
            if not isinstance(source, (RepoSource, RetainedRepoLookupSource)):
                continue
            with source.repo.retain_topology(allow_physical_duplicates=True):
                if tuple(id(store) for store in source.repo.stores) != tuple(id(store) for store in capture.repo_stores(source.repo)):
                    raise QueryError("Repo topology changed during query continuation.")

    def check_cut(store):
        if store.authority_cut() != tokens[store.authority_fence_key()][0]:
            raise QueryError("Store changed during query continuation; start a new query.")

    family, where, params = _store_discovery_relation(query)
    metadata_scopes = query._metadata_scopes()
    aliases = tuple(r for r in query.restrictions if r.kind == "alias")
    metadata_facts = {}
    for restriction in query.restrictions:
        if restriction.kind != "metadata":
            continue
        facts = capture.capture_stores(scope_stores[id(restriction)], metadata_scopes=metadata_scopes)
        metadata_facts[id(restriction)] = facts
        # Validate the requested holder scope before candidate narrowing or any
        # positive/empty terminal can conceal a later invalid populated leaf.
        for fact in facts:
            for key, (value, _) in fact.knowledge()._entries.items():
                query._matches_metadata(value, restriction.value, facts, kind=key.kind)

    def capture_names(page):
        for restriction in aliases:
            name = restriction.value
            named = set()
            for selected in scope_stores[id(restriction)]:
                record = page.read(selected, selected._relative(selected._name_path(name)), r.NodeNameRecord)
                if record is not None:
                    if record.name != name:
                        raise InventoryCapabilityError("Name address disagrees with authority.")
                    membership = page.read(selected, selected._relative(selected._node_path("membership/" + record.root_kind, record.root_digest)), r.NodeMembershipRecord, True)
                    if (membership.root_kind, membership.root_digest) != (record.root_kind, record.root_digest):
                        raise InventoryCapabilityError("Name targets incompatible membership.")
                    named.add((record.root_kind, record.root_digest))
            if len(named) > 1:
                raise QueryError("Query name authority conflicts across connected Stores.")
            page.names[id(restriction)] = next(iter(named), None)

    def capture_membership(page, row):
        for restriction in query.restrictions:
            if restriction.kind not in {"stored", "metadata"}:
                continue
            for selected in scope_stores[id(restriction)]:
                for kind, digest in (("recipe", row["recipe_digest"]), ("saved", row["saved_digest"])):
                    if digest is None:
                        continue
                    member = page.read(selected, selected._relative(selected._node_path("membership/" + kind, digest)), r.NodeMembershipRecord)
                    if member is not None and (member.root_kind, member.root_digest) != (kind, digest):
                        raise InventoryCapabilityError("Scoped membership address disagrees with authority.")
                    page.memberships[selected.authority_fence_key(), kind, digest] = member
                    if member is not None:
                        path = selected._relative(selected._node_path("graphs" if kind == "saved" else "definitions/roots", digest))
                        root = page.read(selected, path, r.NodeRootRecord, True)
                        if root.root_kind != kind or root.digest != digest:
                            raise InventoryCapabilityError("Scoped membership targets incompatible root authority.")
                        page.root(selected, root)
                        page.scoped_roots[selected.authority_fence_key(), kind, digest] = root
                        page.declarations(member.identities)

    def capture_metadata(page, row):
        for scope in metadata_scopes:
            digest = row["object_digest"] if scope in {"object", "lineage"} else row["saved_digest"]
            if digest is None:
                continue
            area = {"object": "current_object", "state": "current_state", "lineage": "lineage", "snapshot": "capture"}[scope]
            for selected in stores:
                domain = selected.authority_fence_key()
                record = page.read(selected, selected._relative(selected._node_metadata_path(area, digest)), r.NodeMetadataRecord)
                if record is not None:
                    selected._validate_node_metadata_record(record, area, digest)
                page.metadata[domain, scope, digest] = record

    def object_holder(selected, value, page, item, owner):
        if value.object_id is not None:
            return (selected.authority_fence_key(), value.object_id) in page.anchors
        if selected is owner and not item["path"]["segments"] and item["root"]["root_kind"] == "saved":
            return True
        # A stateless aggregate has no root ObjectId anchor. Its object projection
        # must instead have a real contributing root in the selected source; a
        # nested selective query shares this terminal's controls/admission budget.
        from .query import IdentityQuery
        nested = IdentityQuery.from_store(selected).object_refs().sel(value.object_view())
        stream = iter_node_query(nested, capture, (selected,), admit=False)
        try:
            return next(stream, None) is not None
        finally:
            stream.close()

    def stream(store):
        domain = store.authority_fence_key()
        index = indexes.get(domain)
        after = None
        scanned = None
        page_size = min(query.candidate_page_size, query.candidate_page_size if query.take_limit is None else query.take_limit + 1)
        if index is None and not direct:
            from .authority import capture_store_projection
            if not hasattr(capture, "_node_scan_projections"):
                capture._node_scan_projections = {}
            scan_key = (domain, tokens[domain][0])
            projection = capture._node_scan_projections.get(scan_key)
            if projection is None:
                capture.store_scan_count += 1
                projection = capture_store_projection(store, rebuild=True)
                capture._node_scan_projections[scan_key] = projection
            scanned = [(source, key, payload) for f, source, key, payload in projection.rows if f == family]
            scanned.sort(key=lambda row: (json.loads(row[2]).get("key_kind", "local_state"),
                                         json.loads(row[2]).get("digest", row[0]), row[0], row[1]))
        while True:
            topology()
            page = _NodePage(stores, tokens, indexes)
            candidates, identities = [], []
            with ExitStack() as scopes:
                # These are short independent Store cuts; sorted acquisition
                # avoids inverse lock order during bounded cross-replica reads.
                for selected in stores:
                    scopes.enter_context(selected.authority_read_fence())
                    check_cut(selected)
                capture_names(page)
                if exact is not None:
                    path = store._relative(os.path.dirname(store.state_data_path(exact))) + "/manifest.record"
                    manifest = page.read(store, path, r.StateVariantManifest)
                    if manifest is None and os.path.lexists(os.path.join(store.base_dir, path.rsplit("/", 1)[0])):
                        raise InventoryCapabilityError("Exact StateKey variant lacks required manifest authority.")
                    rows = [] if manifest is None or after is not None else [(path, "", manifest.to_bytes())]
                elif recipe is not None:
                    import hashlib
                    digest = hashlib.sha256(b"dryml-recipe-root/v1\0" + recipe.exact_key()).hexdigest()
                    path = store._relative(store._node_path("membership/recipe", digest))
                    membership = page.read(store, path, r.NodeMembershipRecord)
                    rows = []
                    if membership is not None and after is None:
                        root = page.read(store, store._relative(store._node_path("definitions/roots", digest)), r.NodeRootRecord, True)
                        row = {"kind": "cdef", "key_kind": "cdef", "digest": recipe.exact_digest(),
                               "root": root.to_data(), "path": {"schema_version": 3, "segments": []},
                               "projection": "creation", "stored": True, "recipe_digest": digest,
                               "object_digest": None, "saved_digest": None}
                        rows = [(path, "", r._json_bytes(row))]
                elif index is None:
                    start = 0 if after is None else after
                    rows = scanned[start:start + page_size]
                    after = start + len(rows)
                else:
                    with index._connections._read_lease() as (con, identity):
                        con.execute("BEGIN")
                        try:
                            if _node_index_token(index, con, identity, store.authority_cut()) != tokens[domain]:
                                raise QueryError("Index changed during query continuation.")
                            order = ("p.source,p.row_key" if local else
                                     "json_extract(CAST(p.payload AS TEXT),'$.key_kind'),json_extract(CAST(p.payload AS TEXT),'$.digest'),p.source,p.row_key")
                            clause, arguments = ("", ()) if after is None else (
                                " AND (" + order + ") > (" + ",".join("?" for _ in after) + ")", after)
                            rows = con.execute("SELECT p.source,p.row_key,p.payload FROM store_projection_rows p WHERE p.family=? AND (" + where + ")" + clause + " ORDER BY " + order + " LIMIT ?",
                                               (family, *params, *arguments, page_size)).fetchall()
                        finally:
                            con.execute("ROLLBACK")
                capture.pages_fetched += 1
                capture.candidate_rows_read += len(rows)
                for path, row_key, payload in rows:
                    if local:
                        manifest = page.read(store, path, r.StateVariantManifest, True)
                        key = manifest.key
                        if path.split("/")[2:4] != [key.cdef_digest, key.codec_hash]:
                            raise InventoryCapabilityError("State manifest address disagrees with its full key.")
                        if exact is not None and key != exact:
                            raise InventoryCapabilityError("Exact full-key manifest address conflicts.")
                        header = page.read(store, "/".join(path.split("/")[:3]) + "/def.pkl", r.NodeRootRecord, True)
                        if header.root_kind != "construction" or header.digest != key.cdef_digest:
                            raise InventoryCapabilityError("State creation header address conflicts.")
                        page.root(store, header)
                        availability = page.read(store, path.rsplit("/", 1)[0] + "/availability.record", r.StateAvailabilityRecord, True)
                        if availability.key != key:
                            raise InventoryCapabilityError("State availability key conflicts.")
                        present = os.path.isdir(store.state_data_path(key))
                        page.presence[domain, key] = present
                        page.available[key] = present and availability.status == "published"
                        for restriction in query.restrictions:
                            if restriction.kind == "stored":
                                for selected in scope_stores[id(restriction)]:
                                    target = selected._relative(os.path.dirname(selected.state_data_path(key))) + "/manifest.record"
                                    other = page.read(selected, target, r.StateVariantManifest)
                                    if other is not None and other.key != key:
                                        raise InventoryCapabilityError("Scoped StateKey manifest conflicts.")
                                    page.state_memberships[selected.authority_fence_key(), key] = other
                        if any(item.kind == "retained" for item in query.restrictions):
                            from ..store.retention import RetentionGrantRecord
                            relative = path.rsplit("/", 1)[0] + "/grants"
                            grants = []
                            for name in page.witnesses[domain].directory(relative):
                                if name.startswith(".dryml-"):
                                    continue
                                grant = page.read(store, relative + "/" + name, RetentionGrantRecord, True)
                                if key not in grant.keys or name != grant.address + ".record":
                                    raise InventoryCapabilityError("Retention grant names an incompatible full key.")
                                grants.append(grant)
                            page.grants[key] = bool(grants)
                        candidates.append((manifest, header))
                    else:
                        row = json.loads(payload)
                        record = r.NodeRootRecord.from_data(row["root"])
                        membership = page.read(store, path, r.NodeMembershipRecord, True)
                        if (membership.root_kind, membership.root_digest) != (record.root_kind, record.digest):
                            raise InventoryCapabilityError("Derived candidate has no matching source membership.")
                        actual_path = store._relative(store._node_path("graphs" if record.root_kind == "saved" else "definitions/roots", record.digest))
                        if page.read(store, actual_path, r.NodeRootRecord, True) != record:
                            raise InventoryCapabilityError("Derived candidate disagrees with current root authority.")
                        identities.extend(membership.identities)
                        if not metadata_facts:
                            capture_metadata(page, row)
                        capture_membership(page, row)
                        page.root(store, record)
                        candidates.append((row, record))
                page.declarations(identities)
            page.expand(capture, owners, max_cached_bytes=owner_budget)
            for item, header in candidates:
                node = page.value(store, header)
                if local:
                    value = LocalState(item.key, node)
                else:
                    from ..utils.graph.path import GraphPath
                    from ..node_views import OBJECT_PROJECTION_VIEW, view_equal
                    for declared in store._publication_identities(node):
                        anchor = page.anchors.get((domain, declared.object_id))
                        if anchor is None or (header.root_kind, header.digest) not in anchor.sources:
                            raise InventoryCapabilityError("Candidate has no required declaration contribution.")
                        if not view_equal(declared, page.value(store, anchor.projection), OBJECT_PROJECTION_VIEW):
                            raise InventoryCapabilityError("Candidate conflicts with its declaration anchor.")
                    value = node.at(GraphPath.from_data(item["path"]), traverse_refs=True)
                    view_record = header
                    if item["projection"] == "creation":
                        value = value.creation_view()
                    elif item["projection"] == "object":
                        anchor = page.anchors.get((domain, value.object_id))
                        value = value.object_view() if anchor is None else page.value(store, anchor.projection)
                        if anchor is not None:
                            view_record = anchor.projection
                    elif item["projection"] == "saved":
                        require_saved(value)
                    owner = page.preparation(store, view_record)
                    try:
                        digest = owner.digest(value)
                    except ValueError:
                        from ..node_prepare import NodePreparation
                        owner = NodePreparation((value,), previous=owner)
                        digest = owner.digest(value)
                    if digest != item["digest"]:
                        raise InventoryCapabilityError("Candidate semantic digest disagrees with authority.")
                key = identity_key(value) if local else IdentityKey(item["kind"], item["digest"], value)
                evidence = IdentityEvidence(frozenset((query._source_evidence(store),)))
                page.validate()
                # The federator peeks at ordering without admitting or evaluating
                # a lookahead candidate. send(True) owns exact residual admission.
                if not (yield key, (value, evidence)):
                    raise RuntimeError("Discovery candidate requires explicit admission.")
                if admit:
                    capture.reserve_keys((key,))
                if local or item["stored"]:
                    capture._node_stored_keys.setdefault(domain, set()).add(key)
                match = not isinstance(query.source, RetainedRepoLookupSource) or item["stored"]
                for restriction in query.restrictions:
                    kind = restriction.kind
                    if kind == "kind":
                        match &= (restriction.value == "local_state" if local else item["kind"] == restriction.value)
                    elif kind == "stored":
                        if local:
                            match &= any(page.state_memberships.get((s.authority_fence_key(), value.key)) is not None for s in scope_stores[id(restriction)])
                        elif item["kind"] == "object_ref":
                            match &= any(object_holder(s, value, page, item, store) for s in scope_stores[id(restriction)])
                        else:
                            root_kind = "recipe" if item["kind"] == "cdef" else "saved"
                            digest = item["recipe_digest"] if root_kind == "recipe" else item["saved_digest"]
                            matched = False
                            for selected in scope_stores[id(restriction)]:
                                root = page.scoped_roots.get((selected.authority_fence_key(), root_kind, digest))
                                if root is not None:
                                    matched |= value.graph_equal(page.value(selected, root))
                            match &= matched
                    elif kind == "origin":
                        from ..node_views import CREATION_VIEW, view_equal
                        match &= local and view_equal(value.creation_cdef, restriction.value, CREATION_VIEW)
                    elif kind == "compatible":
                        try:
                            value.validate_for(restriction.value, codec=value.codec)
                        except (TypeError, ValueError):
                            match = False
                    elif kind == "state_key":
                        match &= local and value.key == restriction.value
                    elif kind == "available":
                        match &= local and page.available[value.key]
                    elif kind == "sel" and local:
                        match &= prepared[id(restriction)].matches(value, generator_budget=capture.witness_budget())
                    elif kind == "alias":
                        match &= not local and page.names[id(restriction)] == (header.root_kind, header.digest) and not item["path"]["segments"]
                    elif kind == "metadata":
                        match &= query._matches_metadata(value, restriction.value,
                                                         metadata_facts[id(restriction)], kind=item["kind"])
                    elif kind == "retained":
                        match &= local and page.grants.get(value.key, False)
                    elif kind == "sel" and restriction.prepared is not None:
                        match &= restriction.prepared.matches(value, generator_budget=capture.witness_budget(), prepared=owner)
                    else:
                        match &= query._matches(restriction, value, IdentityEvidence(frozenset((query._source_evidence(store),))), capture, capture.witness_budget())
                page.validate()
                yield (key, (value, evidence)) if match else None
            page.validate()
            topology()
            if not rows or direct:
                return
            if index is not None:
                last = rows[-1]
                data = None if local else json.loads(last[2])
                after = last[:2] if local else (data["key_kind"], data["digest"], *last[:2])

    streams, frontier = [], []
    try:
        for position, store in enumerate(producers.values()):
            iterator = stream(store)
            streams.append(iterator)
            first = next(iterator, None)
            if first is not None:
                heapq.heappush(frontier, ((first[0].kind, first[0].digest), position, first))
        if not local and not any(item.kind in {"stored", "alias", "metadata"} for item in query.restrictions):
            def cached_stream():
                for key in sorted(cached._entries, key=lambda key: key.sort_key):
                    value = cached._entries[key][0]
                    evidence = IdentityEvidence(frozenset((query._source_evidence(query.source.repo),)))
                    if not (yield key, (value, evidence)):
                        raise RuntimeError("Cache candidate requires explicit admission.")
                    if admit:
                        capture.reserve((value,))
                    match = all(query._matches(item, value, evidence, capture, capture.witness_budget()) for item in query.restrictions)
                    yield (key, (value, evidence)) if match else None
            iterator = cached_stream()
            position = len(streams)
            streams.append(iterator)
            first = next(iterator, None)
            if first is not None:
                heapq.heappush(frontier, ((first[0].kind, first[0].digest), position, first))
        while frontier:
            topology()
            _NodePage(stores, tokens, indexes).validate()
            bucket_id = frontier[0][0]
            bucket = {}
            while frontier and frontier[0][0] == bucket_id:
                _, position, (key, (value, evidence)) = heapq.heappop(frontier)
                qualified = streams[position].send(True)
                if qualified is not None:
                    key, (value, evidence) = qualified
                    old = bucket.get(key)
                    if old is not None and isinstance(value, LocalState) and old[0] != value:
                        raise QueryError("Local-state creation authority conflicts across Stores.")
                    bucket[key] = (value, evidence) if old is None else (old[0], old[1].merged_with(evidence))
                following = next(streams[position], None)
                if following is not None:
                    heapq.heappush(frontier, ((following[0].kind, following[0].digest), position, following))
            for key in sorted(bucket, key=lambda key: key.sort_key):
                if key in cached._entries:
                    value, evidence = bucket[key]
                    bucket[key] = value, evidence.merged_with(IdentityEvidence(frozenset((query._source_evidence(query.source.repo),))))
                yield key, bucket[key]
    finally:
        for iterator in streams:
            iterator.close()
