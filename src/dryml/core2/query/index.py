from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any

from ..definition import ConcreteDefinition
from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from ..utils.stable_hash import stable_hash_function
from .fingerprint import canonical_class_key, target_fingerprint
from .model import (
    DefinitionId,
    DefinitionOccurrence,
    DefinitionRecord,
    FeatureRequirement,
    FeatureToken,
    OccurrenceKey,
    QueryIndexError,
    QueryStats,
    RefreshPolicy,
    StoreId,
)
from .path import Arg, DefinitionPath, Index, Key, Kwarg


class DefinitionCatalog:
    def __init__(self, repo):
        self.repo = repo
        self.lock = RLock()
        self.definitions_by_id: dict[DefinitionId, DefinitionRecord] = {}
        self.ids_by_cdef: dict[ConcreteDefinition, DefinitionId] = {}
        self.ids_by_stable_hash: dict[str, set[DefinitionId]] = defaultdict(set)
        self.replicas_by_definition: dict[DefinitionId, set[StoreId]] = defaultdict(set)
        self.stored_definitions_by_store: dict[StoreId, set[DefinitionId]] = defaultdict(set)
        self.occurrences_by_nested: dict[DefinitionId, set[OccurrenceKey]] = defaultdict(set)
        self.occurrences_by_owner: dict[DefinitionId, set[OccurrenceKey]] = defaultdict(set)
        self.occurrence_by_key: dict[OccurrenceKey, DefinitionOccurrence] = {}
        self.postings: dict[FeatureToken, dict[DefinitionId, int]] = defaultdict(dict)
        self.store_by_id: dict[StoreId, Any] = {}
        self.hydrated_stores: set[StoreId] = set()
        self.generation = 0

    def store_id(self, store) -> StoreId:
        sid = self._store_key(store)
        self.store_by_id.setdefault(sid, store)
        return sid

    def _store_key(self, store) -> StoreId:
        if hasattr(store, "catalog_key"):
            return store.catalog_key()
        return f"{type(store).__module__}.{type(store).__qualname__}:id:{id(store)}"

    def _unique_repo_stores(self) -> list[Any]:
        stores = []
        seen = set()
        for store in self.repo.stores:
            sid = self._store_key(store)
            if sid in seen:
                continue
            seen.add(sid)
            stores.append(store)
        return stores

    def sync_caches(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        with self.lock:
            ids: set[DefinitionId] = set()
            for cdef in self.repo.strong_obj_cache.keys():
                ids.add(self._register_definition_locked(cdef))
            if reuse_weak:
                for cdef in self.repo.weak_obj_cache.keys():
                    ids.add(self._register_definition_locked(cdef))
            return ids

    def refresh(self, policy: RefreshPolicy, *, stats: QueryStats | None = None) -> None:
        if policy is False:
            return
        if policy is True:
            self._rebuild_from_stores(stats=stats)
            return

        with self.lock:
            stores = self._unique_repo_stores()
            seen = set()
            unseen = []
            for store in stores:
                sid = self._store_key(store)
                if sid in seen or sid in self.hydrated_stores:
                    continue
                seen.add(sid)
                unseen.append(store)
        if not unseen:
            return
        self._hydrate_stores(unseen, stats=stats)

    def ensure_exact_stored(self, cdef: ConcreteDefinition, *, stats: QueryStats | None = None) -> bool:
        found = False
        for store in self._unique_repo_stores():
            persisted = store.read_definition(cdef) if hasattr(store, "read_definition") else None
            if persisted is not None and persisted == cdef:
                self.register_stored_graph(cdef, store)
                found = True
        if found and stats is not None:
            stats.fast_path = "exact-root-store-has"
        return found

    def register_cached(self, cdef: ConcreteDefinition) -> DefinitionId:
        with self.lock:
            return self._register_definition_locked(cdef)

    def register_stored_graph(self, cdef: ConcreteDefinition, store) -> DefinitionId:
        with self.lock:
            sid = self.store_id(store)
            did = self._register_definition_locked(cdef)
            self.replicas_by_definition[did].add(sid)
            self.stored_definitions_by_store[sid].add(did)
            self.repo.light_index.add(cdef)
            self._register_occurrences_locked(owner_id=did, root=cdef)
            self.generation += 1
            return did

    def cdef_id(self, cdef: ConcreteDefinition) -> DefinitionId | None:
        with self.lock:
            return self.ids_by_cdef.get(cdef)

    def record_for_cdef(self, cdef: ConcreteDefinition) -> DefinitionRecord | None:
        with self.lock:
            did = self.ids_by_cdef.get(cdef)
            return self.definitions_by_id.get(did) if did is not None else None

    def stored_ids(self) -> set[DefinitionId]:
        with self.lock:
            return {did for did, stores in self.replicas_by_definition.items() if stores}

    def known_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        cached = self.sync_caches(reuse_weak=reuse_weak)
        return self.stored_ids() | cached

    def cached_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        return self.sync_caches(reuse_weak=reuse_weak)

    def nested_ids(self) -> set[DefinitionId]:
        with self.lock:
            return set(self.occurrences_by_nested.keys())

    def all_occurrences(self) -> tuple[DefinitionOccurrence, ...]:
        with self.lock:
            return tuple(self.occurrence_by_key.values())

    def occurrences_for_nested_ids(self, ids: set[DefinitionId]) -> tuple[DefinitionOccurrence, ...]:
        with self.lock:
            out: list[DefinitionOccurrence] = []
            for did in ids:
                for key in self.occurrences_by_nested.get(did, ()):
                    out.append(self.occurrence_by_key[key])
            return tuple(out)

    def definition_for_id(self, did: DefinitionId) -> ConcreteDefinition:
        return self.definitions_by_id[did].cdef

    def ids_to_cdefs(self, ids: set[DefinitionId] | frozenset[DefinitionId]) -> tuple[ConcreteDefinition, ...]:
        with self.lock:
            return tuple(self.definitions_by_id[did].cdef for did in ids if did in self.definitions_by_id)

    def stores_for_cdef(self, cdef: ConcreteDefinition) -> tuple[Any, ...]:
        with self.lock:
            did = self.ids_by_cdef.get(cdef)
            if did is None:
                return ()
            store_ids = self.replicas_by_definition.get(did, set())
            seen: set[StoreId] = set()
            stores = []
            for store in self.repo.stores:
                sid = self.store_id(store)
                if sid in store_ids and sid not in seen:
                    seen.add(sid)
                    stores.append(store)
            extras = [self.store_by_id[sid] for sid in sorted(store_ids - seen)]
            return tuple(stores + extras)

    def candidate_ids(
            self,
            universe_ids: set[DefinitionId],
            requirements: tuple[FeatureRequirement, ...],
            *,
            stats: QueryStats | None = None) -> set[DefinitionId]:
        if not requirements:
            if stats is not None:
                stats.candidate_count = len(universe_ids)
            return set(universe_ids)

        with self.lock:
            posting_data = []
            for req in requirements:
                posting = self.postings.get(req.token, {})
                posting_data.append((req, posting, len(posting)))
            posting_data.sort(key=lambda item: item[2])

            if stats is not None:
                stats.selected_features = tuple(req for req, _, _ in posting_data)
                stats.posting_sizes = tuple(size for _, _, size in posting_data)

            candidates: set[DefinitionId] | None = None
            for req, posting, _ in posting_data:
                ids = {did for did, count in posting.items() if count >= req.count}
                if candidates is None:
                    candidates = ids
                else:
                    candidates &= ids
                if not candidates:
                    break

            if candidates is None:
                candidates = set(universe_ids)
            else:
                candidates &= universe_ids

            if stats is not None:
                stats.candidate_count = len(candidates)
            return candidates

    def _hydrate_stores(self, stores, *, stats: QueryStats | None = None) -> None:
        with self.lock:
            snapshot = self._snapshot_locked()
        collected = []
        try:
            for store in stores:
                sid = self._store_key(store)
                try:
                    cdefs = tuple(store.hydrate_index())
                except Exception as e:
                    raise QueryIndexError(f"Failed to hydrate query index from store {store!r}: {e}") from e
                for cdef in cdefs:
                    if not isinstance(cdef, ConcreteDefinition):
                        raise QueryIndexError(f"Store {store!r} yielded {type(cdef).__name__}, not ConcreteDefinition.")
                collected.append((store, sid, cdefs))

            for store, sid, cdefs in collected:
                with self.lock:
                    self.store_by_id.setdefault(sid, store)
                    self.hydrated_stores.add(sid)
                if stats is not None:
                    stats.store_scan_count += 1
                    stats.refresh_action = "hydrate"
                for cdef in cdefs:
                    self.register_stored_graph(cdef, store)
        except Exception:
            with self.lock:
                self._restore_locked(snapshot)
            raise

    def _rebuild_from_stores(self, *, stats: QueryStats | None = None) -> None:
        stores = self._unique_repo_stores()
        cached_cdefs = list(self.repo.strong_obj_cache.keys()) + list(self.repo.weak_obj_cache.keys())
        with self.lock:
            snapshot = self._snapshot_locked()

        collected = []
        try:
            for store in stores:
                sid = self._store_key(store)
                try:
                    cdefs = tuple(store.hydrate_index())
                except Exception as e:
                    raise QueryIndexError(f"Failed to hydrate query index from store {store!r}: {e}") from e
                for cdef in cdefs:
                    if not isinstance(cdef, ConcreteDefinition):
                        raise QueryIndexError(f"Store {store!r} yielded {type(cdef).__name__}, not ConcreteDefinition.")
                collected.append((store, sid, cdefs))

            with self.lock:
                self.definitions_by_id.clear()
                self.ids_by_cdef.clear()
                self.ids_by_stable_hash.clear()
                self.replicas_by_definition.clear()
                self.stored_definitions_by_store.clear()
                self.occurrences_by_nested.clear()
                self.occurrences_by_owner.clear()
                self.occurrence_by_key.clear()
                self.postings.clear()
                self.store_by_id.clear()
                self.hydrated_stores.clear()
                self.repo.light_index.clear()
                self.generation += 1

            for cdef in cached_cdefs:
                self.register_cached(cdef)
            for store, sid, cdefs in collected:
                with self.lock:
                    self.store_by_id.setdefault(sid, store)
                    self.hydrated_stores.add(sid)
                if stats is not None:
                    stats.store_scan_count += 1
                for cdef in cdefs:
                    self.register_stored_graph(cdef, store)
            if stats is not None:
                stats.refresh_action = "forced-refresh"
        except Exception:
            with self.lock:
                self._restore_locked(snapshot)
            raise

    def _register_definition_locked(self, cdef: ConcreteDefinition) -> DefinitionId:
        existing = self.ids_by_cdef.get(cdef)
        if existing is not None:
            return existing

        digest = cdef.stable_hash()
        for did in self.ids_by_stable_hash.get(digest, set()):
            record = self.definitions_by_id[did]
            if record.cdef == cdef:
                self.ids_by_cdef[cdef] = did
                return did

        if digest not in self.definitions_by_id:
            did = digest
        else:
            did = f"{digest}#{len(self.ids_by_stable_hash[digest])}"

        fingerprint = target_fingerprint(cdef)
        record = DefinitionRecord(
            definition_id=did,
            cdef=cdef,
            class_key=canonical_class_key(cdef.cls),
            fingerprint=fingerprint,
        )
        self.definitions_by_id[did] = record
        self.ids_by_cdef[cdef] = did
        self.ids_by_stable_hash[digest].add(did)
        for token, count in fingerprint.counts.items():
            self.postings[token][did] = count
        return did

    def _register_occurrences_locked(self, *, owner_id: DefinitionId, root: ConcreteDefinition) -> None:
        for idx, child in enumerate(root.args):
            self._walk_nested_locked(owner_id, child, DefinitionPath((Arg(idx),)), stack={owner_id})
        for key, child in root.kwargs.items():
            self._walk_nested_locked(owner_id, child, DefinitionPath((Kwarg(key),)), stack={owner_id})

    def _walk_nested_locked(
            self,
            owner_id: DefinitionId,
            value: Any,
            path: DefinitionPath,
            *,
            stack: set[DefinitionId] | None = None) -> None:
        if stack is None:
            stack = set()

        if isinstance(value, ConcreteDefinition):
            nested_id = self._register_definition_locked(value)
            key = (owner_id, path, nested_id)
            if key not in self.occurrence_by_key:
                owner_cdef = self.definitions_by_id[owner_id].cdef
                self.occurrence_by_key[key] = DefinitionOccurrence(owner_cdef, path, value)
                self.occurrences_by_nested[nested_id].add(key)
                self.occurrences_by_owner[owner_id].add(key)
            if nested_id in stack:
                return
            next_stack = set(stack)
            next_stack.add(nested_id)
            for idx, child in enumerate(value.args):
                self._walk_nested_locked(owner_id, child, path.child(Arg(idx)), stack=next_stack)
            for key_name, child in value.kwargs.items():
                self._walk_nested_locked(owner_id, child, path.child(Kwarg(key_name)), stack=next_stack)
            return

        if isinstance(value, (FrozenDict, dict)):
            for key, child in value.items():
                self._walk_nested_locked(owner_id, child, path.child(Key(key)), stack=stack)
            return

        if isinstance(value, (FrozenList, FrozenTuple, list, tuple)):
            for idx, child in enumerate(value):
                self._walk_nested_locked(owner_id, child, path.child(Index(idx)), stack=stack)
            return

        if isinstance(value, (FrozenSet, set, frozenset)):
            for idx, child in enumerate(_sorted_set_values(value)):
                self._walk_nested_locked(owner_id, child, path.child(Index(idx)), stack=stack)
            return

    def _snapshot_locked(self):
        return {
            "definitions_by_id": dict(self.definitions_by_id),
            "ids_by_cdef": dict(self.ids_by_cdef),
            "ids_by_stable_hash": defaultdict(set, {k: set(v) for k, v in self.ids_by_stable_hash.items()}),
            "replicas_by_definition": defaultdict(set, {k: set(v) for k, v in self.replicas_by_definition.items()}),
            "stored_definitions_by_store": defaultdict(set, {k: set(v) for k, v in self.stored_definitions_by_store.items()}),
            "occurrences_by_nested": defaultdict(set, {k: set(v) for k, v in self.occurrences_by_nested.items()}),
            "occurrences_by_owner": defaultdict(set, {k: set(v) for k, v in self.occurrences_by_owner.items()}),
            "occurrence_by_key": dict(self.occurrence_by_key),
            "postings": defaultdict(dict, {k: dict(v) for k, v in self.postings.items()}),
            "store_by_id": dict(self.store_by_id),
            "hydrated_stores": set(self.hydrated_stores),
            "generation": self.generation,
            "light_index": set(self.repo.light_index),
        }

    def _restore_locked(self, snapshot) -> None:
        self.definitions_by_id = dict(snapshot["definitions_by_id"])
        self.ids_by_cdef = dict(snapshot["ids_by_cdef"])
        self.ids_by_stable_hash = defaultdict(set, {k: set(v) for k, v in snapshot["ids_by_stable_hash"].items()})
        self.replicas_by_definition = defaultdict(set, {k: set(v) for k, v in snapshot["replicas_by_definition"].items()})
        self.stored_definitions_by_store = defaultdict(set, {k: set(v) for k, v in snapshot["stored_definitions_by_store"].items()})
        self.occurrences_by_nested = defaultdict(set, {k: set(v) for k, v in snapshot["occurrences_by_nested"].items()})
        self.occurrences_by_owner = defaultdict(set, {k: set(v) for k, v in snapshot["occurrences_by_owner"].items()})
        self.occurrence_by_key = dict(snapshot["occurrence_by_key"])
        self.postings = defaultdict(dict, {k: dict(v) for k, v in snapshot["postings"].items()})
        self.store_by_id = dict(snapshot["store_by_id"])
        self.hydrated_stores = set(snapshot["hydrated_stores"])
        self.generation = snapshot["generation"]
        self.repo.light_index.clear()
        self.repo.light_index.update(snapshot["light_index"])


def _sorted_set_values(values):
    try:
        return sorted(values, key=lambda value: (stable_hash_function(value), repr(value)))
    except TypeError as e:
        raise QueryIndexError(
            "Cannot index nested definitions inside a set containing non-stably-hashable values."
        ) from e
