"""Private Query V3 producer inventory and coherent Store-cut capture.

This layer turns Repo, Store, and fixed members into detached identity evidence
without publishing a replacement query API. It snapshots Repo topology once per
capture and groups only identical authority-fence domains.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

from ..object import Object
from ..reference_values import ObjectRef, StateRef
from ..store.store import Store, StoreCapabilityError
from .authority import CapturedStoreFacts, IdentityValue, _derive_identities
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


class SourceCapture:
    """One terminal-local registry of detached Store facts and cache inventory.

    ``source_cuts`` counts distinct authority-fence domains, while
    ``capture_rounds`` and ``demand_recaptures`` distinguish monotonic metadata
    demand growth from source instability retries owned by later execution work.
    """

    def __init__(self):
        self._facts: dict[str, CapturedStoreFacts] = {}
        self._exact_states: dict[str, dict[str, StateRef | None]] = {}
        self._topologies: dict[int, tuple[Store, ...]] = {}
        self._cache_knowledge: dict[tuple[int, bool], IdentitySet] = {}
        self.active = False
        self.source_cuts = 0
        self.capture_rounds = 0
        self.demand_recaptures = 0
        self.instability_retries = 0

    def repo_stores(self, repo) -> tuple[Store, ...]:
        """Retain the first connected producer topology for this terminal."""

        key = id(repo)
        if key not in self._topologies:
            with repo.retain_topology():
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
        identity inventory was captured.
        """

        store = source.store if isinstance(source, StoreSource) else source
        if not isinstance(store, Store):
            raise TypeError("Store capture requires a Store or StoreSource.")
        key = store.authority_fence_key()
        cached = self._facts.get(key)
        if cached is not None and metadata_scopes <= cached.metadata_scopes:
            return cached
        self.capture_rounds += 1
        if cached is None:
            self.source_cuts += 1
        else:
            self.demand_recaptures += 1
        self._require_complete_inventory(store)
        requested_scopes = metadata_scopes | (cached.metadata_scopes if cached else frozenset())
        with store.authority_read_fence():
            main = store.read_main_ref()
            definitions = tuple(store.iter_definition_records())
            definition_by_digest = {record.digest: record.definition for record in definitions}
            if main is not None:
                main_definition = definition_by_digest.get(main.definition_digest)
                if main_definition is None:
                    raise InventoryCapabilityError("Main-reference authority targets a missing definition record.")
            else:
                main_definition = None
            facts = CapturedStoreFacts.build(
                definitions=definitions,
                stored_roots=store.iter_stored_root_records(),
                declarations=store.iter_declaration_records(),
                state_refs=store.iter_state_ref_records(),
                object_aliases=store.iter_object_alias_records(),
                state_aliases=store.iter_state_alias_records(),
                main_definition=main_definition,
                source=SourceEvidence.from_source(store.authority_fence_key()),
            )
            if requested_scopes:
                facts = CapturedStoreFacts.build(
                    definitions=facts.definitions,
                    stored_roots=facts.stored_roots,
                    declarations=facts.declarations,
                    state_refs=facts.state_refs,
                    object_aliases=facts.object_aliases,
                    state_aliases=facts.state_aliases,
                    main_definition=facts.main_definition,
                    source=facts.source,
                    metadata=self._capture_metadata(store, facts, requested_scopes),
                    metadata_scopes=requested_scopes,
                )
        exact = self._exact_states.pop(key, {})
        exact_changed = False
        if exact:
            current_states = {record.digest: record.state_ref for record in facts.state_refs}
            exact_changed = any(
                previous != current_states.get(digest) for digest, previous in exact.items()
            )
        self._facts[key] = facts
        if exact_changed and self.active:
            raise _SourceRecapture(True)
        if cached is not None and self.active:
            raise _SourceRecapture(_cut_invalidated(cached, facts))
        return facts

    def capture_stores(
        self, sources: Iterable[StoreSource | Store], *, metadata_scopes: frozenset[str] = frozenset()
    ) -> tuple[CapturedStoreFacts, ...]:
        """Capture each selected Store while preserving distinct transaction cuts."""

        return tuple(
            self.capture_store(source, metadata_scopes=metadata_scopes)
            for source in sources
        )

    def read_exact_state(self, source: StoreSource | Store, target: StateRef) -> StateRef | None:
        """Read one complete exact StateRef without requiring broad inventory.

        The digest-addressed record is independently authoritative for exact
        saved-state membership. This selective path intentionally does not use
        the capture registry or any record-family iterator.
        """

        store = source.store if isinstance(source, StoreSource) else source
        if not isinstance(store, Store):
            raise TypeError("Exact state lookup requires a Store or StoreSource.")
        if not isinstance(target, StateRef):
            raise TypeError("Exact state lookup requires a StateRef.")
        key = store.authority_fence_key()
        digest = target.digest()
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
        with retain_topology():
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
