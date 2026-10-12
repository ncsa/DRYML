"""Private Query V3 producer inventory and coherent Store-cut capture.

This layer turns Repo, Store, and fixed members into detached identity evidence
without publishing a replacement query API. It snapshots Repo topology once per
capture and groups only identical authority-fence domains.
The terminal's capture also enforces outer scan/index controls and cumulative
identity admissions, retaining expenditure across replacement and retry without
changing authoritative records or introducing cross-terminal proof reuse.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

from ..object import Object
from ..definition import ConcreteDefinition
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
    and completed recovery scans separately from page fetch/decode counters.
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
        self.scan_warning_emitted = False
        self._facts: dict[str, CapturedStoreFacts] = {}
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

        from .model import QueryVerifyBudgetExceeded

        new = {identity_key(value) for value in values} - self._admitted
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
            from .query import _GeneratorWitnessBudget

            self._witness_budget = _GeneratorWitnessBudget(self.max_witness_limit)
        return self._witness_budget

    def _record_cut(self, key: str) -> None:
        """Account for each fenced read, including selective Store authority."""

        self.capture_rounds += 1
        if key not in self._cut_domains:
            self._cut_domains.add(key)
            self.source_cuts += 1

    def repo_stores(self, repo) -> tuple[Store, ...]:
        """Retain the first connected producer topology for this terminal."""

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
            return cached
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
                values.append(reference)
            receipt = value.last_state_ref
            if isinstance(receipt, StateRef) and receipt.object == reference:
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
