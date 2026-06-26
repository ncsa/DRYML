from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from threading import RLock
from typing import Any

from ..cdef_graph import ConcreteDefinitionGraph
from ..definition import ConcreteDefinition
from .fingerprint import canonical_class_key, target_local_fingerprint
from .model import (
    DefinitionId,
    AllOccurrenceTraversalSnapshot,
    DefinitionEdgeRecord,
    DefinitionOccurrence,
    DefinitionRecord,
    EdgeKey,
    FeatureRequirement,
    FeatureToken,
    IndexWriteResult,
    OccurrenceTraversalSnapshot,
    OwnerProjection,
    QueryIndexError,
    QueryIndexUnavailable,
    QueryStats,
    RefreshPolicy,
    StoreId,
)
from .path import DefinitionPath
from .utils import cdef_equal


class _CatalogBuildRepo:
    def __init__(self, repo):
        self._repo = repo
        self.stores = repo.stores
        self.strong_obj_cache = repo.strong_obj_cache
        self.weak_obj_cache = repo.weak_obj_cache
        self.light_index: set[ConcreteDefinition] = set()


class MemoryDefinitionGraphReadView:
    def __init__(
            self,
            *,
            generation: int,
            definitions_by_id: dict[DefinitionId, DefinitionRecord],
            ids_by_cdef: dict[ConcreteDefinition, DefinitionId],
            ids_by_stable_hash: dict[str, frozenset[DefinitionId]],
            replicas_by_definition: dict[DefinitionId, frozenset[StoreId]],
            local_postings: dict[FeatureToken, dict[DefinitionId, int]],
            edge_by_key: dict[EdgeKey, DefinitionEdgeRecord],
            outgoing_edges: dict[DefinitionId, frozenset[EdgeKey]],
            incoming_edges: dict[DefinitionId, frozenset[EdgeKey]],
            child_by_parent_path: dict[tuple[DefinitionId, DefinitionPath], frozenset[DefinitionId]],
            parents_by_child_path: dict[tuple[DefinitionId, DefinitionPath], frozenset[DefinitionId]],
            strong_cached_ids: set[DefinitionId],
            weak_cached_ids: set[DefinitionId],
            source_key: str = "memory-catalog",
            repo=None,
            store_by_id: dict[StoreId, Any] | None = None):
        self.source_key = source_key
        self.generation = generation
        self._definitions_by_id = definitions_by_id
        self._ids_by_cdef = ids_by_cdef
        self._ids_by_stable_hash = ids_by_stable_hash
        self._replicas_by_definition = replicas_by_definition
        self._local_postings = local_postings
        self._edge_by_key = edge_by_key
        self._outgoing_edges = outgoing_edges
        self._incoming_edges = incoming_edges
        self._child_by_parent_path = child_by_parent_path
        self._parents_by_child_path = parents_by_child_path
        self._strong_cached_ids = frozenset(strong_cached_ids)
        self._weak_cached_ids = frozenset(weak_cached_ids)
        self._repo = repo
        self._store_by_id = {} if store_by_id is None else store_by_id
        self._active = True

    def close(self) -> None:
        self._active = False

    def _check_active(self) -> None:
        if not self._active:
            raise QueryIndexError("Catalog read view is only valid inside DefinitionCatalog.read_view().")

    def _private_data_error(self):
        raise AttributeError("MemoryDefinitionGraphReadView backing data is private; use read-view methods.")

    definitions_by_id = property(lambda self: self._private_data_error())
    ids_by_cdef = property(lambda self: self._private_data_error())
    ids_by_stable_hash = property(lambda self: self._private_data_error())
    replicas_by_definition = property(lambda self: self._private_data_error())
    local_postings = property(lambda self: self._private_data_error())
    edge_by_key = property(lambda self: self._private_data_error())
    outgoing_edges = property(lambda self: self._private_data_error())
    incoming_edges = property(lambda self: self._private_data_error())
    child_by_parent_path = property(lambda self: self._private_data_error())
    parents_by_child_path = property(lambda self: self._private_data_error())

    def all_definition_ids(self) -> set[DefinitionId]:
        self._check_active()
        return set(self._definitions_by_id)

    def cdef_id(self, cdef: ConcreteDefinition) -> DefinitionId | None:
        self._check_active()
        return self._ids_by_cdef.get(cdef)

    def definition_for_id(self, did: DefinitionId) -> ConcreteDefinition:
        self._check_active()
        return self._definitions_by_id[did].cdef

    def record_for_cdef(self, cdef: ConcreteDefinition) -> DefinitionRecord | None:
        self._check_active()
        did = self._ids_by_cdef.get(cdef)
        return self._definitions_by_id.get(did) if did is not None else None

    def ids_to_cdefs(self, ids: set[DefinitionId] | frozenset[DefinitionId]) -> tuple[ConcreteDefinition, ...]:
        self._check_active()
        return tuple(self._definitions_by_id[did].cdef for did in ids if did in self._definitions_by_id)

    def cdefs_by_id(self, ids: set[DefinitionId] | frozenset[DefinitionId]) -> dict[DefinitionId, ConcreteDefinition]:
        self._check_active()
        return {did: self._definitions_by_id[did].cdef for did in ids if did in self._definitions_by_id}

    def exact_ids(self, cdef: ConcreteDefinition) -> set[DefinitionId]:
        self._check_active()
        digest = cdef.stable_hash()
        return {
            did
            for did in self._ids_by_stable_hash.get(digest, frozenset())
            if cdef_equal(self._definitions_by_id[did].cdef, cdef)
        }

    def estimate_exact_ids(self, cdef: ConcreteDefinition) -> int:
        self._check_active()
        return len(self.exact_ids(cdef))

    def estimate_local_candidates(self, requirements: tuple[FeatureRequirement, ...]) -> int:
        self._check_active()
        if not requirements:
            return len(self._definitions_by_id)
        return min(len(self._local_postings.get(req.token, {})) for req in requirements)

    def local_candidates(
            self,
            requirements: tuple[FeatureRequirement, ...],
            *,
            within: set[DefinitionId] | None = None,
            domain=None,
            stats: QueryStats | None = None) -> set[DefinitionId]:
        self._check_active()
        return _candidate_ids_from_postings_data(
            self._definitions_by_id,
            within,
            requirements,
            self._local_postings,
            domain=domain,
            stats=stats,
        )

    def is_stored_id(self, did: DefinitionId) -> bool:
        self._check_active()
        return bool(self._replicas_by_definition.get(did))

    def filter_stored_ids(self, ids) -> set[DefinitionId]:
        self._check_active()
        return {did for did in ids if self.is_stored_id(did)}

    def all_stored_ids(self) -> set[DefinitionId]:
        self._check_active()
        return {did for did, stores in self._replicas_by_definition.items() if stores}

    def stored_ids_for_store_id(self, store_id: StoreId) -> set[DefinitionId]:
        self._check_active()
        return {
            did
            for did, stores in self._replicas_by_definition.items()
            if store_id in stores
        }

    def is_cached_id(self, did: DefinitionId, *, reuse_weak: bool = True) -> bool:
        self._check_active()
        if did in self._strong_cached_ids:
            return True
        return reuse_weak and did in self._weak_cached_ids

    def all_cached_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        self._check_active()
        ids = set(self._strong_cached_ids)
        if reuse_weak:
            ids.update(self._weak_cached_ids)
        return ids

    def all_known_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        self._check_active()
        return self.all_stored_ids() | self.all_cached_ids(reuse_weak=reuse_weak)

    def replica_map(self, ids: set[DefinitionId] | frozenset[DefinitionId]) -> dict[ConcreteDefinition, tuple[Any, ...]]:
        self._check_active()
        if self._repo is None:
            return {
                self._definitions_by_id[did].cdef: ()
                for did in ids
                if did in self._definitions_by_id
            }
        repo_store_by_id = {
            self._store_key(store): store
            for store in self._repo.stores
        }
        out: dict[ConcreteDefinition, tuple[Any, ...]] = {}
        for did in ids:
            record = self._definitions_by_id.get(did)
            if record is None:
                continue
            stores = []
            seen: set[StoreId] = set()
            for sid in sorted(self._replicas_by_definition.get(did, ())):
                store = repo_store_by_id.get(sid, self._store_by_id.get(sid))
                if store is not None and sid not in seen:
                    seen.add(sid)
                    stores.append(store)
            out[record.cdef] = tuple(stores)
        return out

    def _store_key(self, store) -> StoreId:
        if hasattr(store, "catalog_key"):
            return store.catalog_key()
        return f"{type(store).__module__}.{type(store).__qualname__}:id:{id(store)}"

    def nested_ids(self) -> set[DefinitionId]:
        self._check_active()
        nested: set[DefinitionId] = set()
        for owner_id in self.all_stored_ids():
            nested.update(self._descendant_ids(owner_id))
        return nested

    def filter_nested_ids(self, ids: set[DefinitionId]) -> set[DefinitionId]:
        self._check_active()
        return {did for did in ids if self.has_stored_ancestor(did)}

    def has_stored_ancestor(self, did: DefinitionId) -> bool:
        self._check_active()
        seen: set[DefinitionId] = set()
        stack = [self._edge_by_key[key].parent_id for key in self._incoming_edges.get(did, ())]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            if self.is_stored_id(cur):
                return True
            seen.add(cur)
            stack.extend(self._edge_by_key[key].parent_id for key in self._incoming_edges.get(cur, ()))
        return False

    def _owner_ids_for_nested_ids(self, ids: set[DefinitionId]) -> set[DefinitionId]:
        self._check_active()
        owners: set[DefinitionId] = set()
        seen: set[DefinitionId] = set(ids)
        stack = list(ids)
        while stack:
            cur = stack.pop()
            for edge_key in self._incoming_edges.get(cur, ()):
                parent_id = self._edge_by_key[edge_key].parent_id
                if self.is_stored_id(parent_id):
                    owners.add(parent_id)
                if parent_id not in seen:
                    seen.add(parent_id)
                    stack.append(parent_id)
        return owners

    def project_owners(self, ids: set[DefinitionId] | frozenset[DefinitionId]) -> OwnerProjection:
        self._check_active()
        owner_ids = self._owner_ids_for_nested_ids(set(ids))
        owners = tuple(
            self._definitions_by_id[owner_id].cdef
            for owner_id in owner_ids
            if owner_id in self._definitions_by_id
        )
        replicas = self.replica_map(owner_ids)
        return OwnerProjection(
            owner_ids=frozenset(owner_ids),
            cdefs=owners,
            replicas={owner: replicas.get(owner, ()) for owner in owners},
        )

    def parents(
            self,
            child_ids: set[DefinitionId],
            path: DefinitionPath,
            *,
            unordered: bool,
            within: set[DefinitionId] | frozenset[DefinitionId] | None = None) -> set[DefinitionId]:
        self._check_active()
        if not child_ids or (within is not None and not within):
            return set()
        out = set()
        if unordered:
            if within is None:
                for child_id in child_ids:
                    for edge_key in self._incoming_edges.get(child_id, ()):
                        record = self._edge_by_key[edge_key]
                        if record.path.startswith(path):
                            out.add(record.parent_id)
                return out
            child_filter = set(child_ids)
            for parent_id in within:
                for edge_key in self._outgoing_edges.get(parent_id, ()):
                    record = self._edge_by_key[edge_key]
                    if record.child_id in child_filter and record.path.startswith(path):
                        out.add(parent_id)
                        break
            return out
        for child_id in child_ids:
            parents = self._parents_by_child_path.get((child_id, path), frozenset())
            out.update(parents if within is None else parents & within)
        return out

    def children(
            self,
            parent_ids: set[DefinitionId],
            path: DefinitionPath,
            *,
            unordered: bool,
            within: set[DefinitionId] | frozenset[DefinitionId] | None = None) -> set[DefinitionId]:
        self._check_active()
        if not parent_ids or (within is not None and not within):
            return set()
        out = set()
        if unordered:
            for parent_id in parent_ids:
                for edge_key in self._outgoing_edges.get(parent_id, ()):
                    record = self._edge_by_key[edge_key]
                    if (within is None or record.child_id in within) and record.path.startswith(path):
                        out.add(record.child_id)
            return out
        for parent_id in parent_ids:
            children = self._child_by_parent_path.get((parent_id, path), frozenset())
            out.update(children if within is None else children & within)
        return out

    def occurrence_snapshot_for_nested_ids(self, target_ids: set[DefinitionId]) -> OccurrenceTraversalSnapshot:
        self._check_active()
        incoming: dict[DefinitionId, list[DefinitionEdgeRecord]] = defaultdict(list)
        cdefs: dict[DefinitionId, ConcreteDefinition] = {}
        stored_ids: set[DefinitionId] = set()
        seen: set[DefinitionId] = set()
        stack = [did for did in target_ids if did in self._definitions_by_id]
        for did in stack:
            cdefs[did] = self._definitions_by_id[did].cdef

        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for edge_key in self._incoming_edges.get(cur, ()):
                edge = self._edge_by_key[edge_key]
                incoming[edge.child_id].append(edge)
                cdefs.setdefault(edge.child_id, self._definitions_by_id[edge.child_id].cdef)
                cdefs.setdefault(edge.parent_id, self._definitions_by_id[edge.parent_id].cdef)
                if self.is_stored_id(edge.parent_id):
                    stored_ids.add(edge.parent_id)
                if edge.parent_id not in seen:
                    stack.append(edge.parent_id)

        return OccurrenceTraversalSnapshot(
            targets=set(target_ids),
            cdefs=cdefs,
            stored_ids=stored_ids,
            incoming={child_id: tuple(edges) for child_id, edges in incoming.items()},
            owner_replicas=self.replica_map(stored_ids),
        )

    def iter_all_occurrences(self, *, max_occurrences: int | None = None):
        self._check_active()
        return self.all_occurrence_snapshot().iter_occurrences(max_occurrences=max_occurrences)

    def all_occurrence_snapshot(self) -> AllOccurrenceTraversalSnapshot:
        self._check_active()
        return AllOccurrenceTraversalSnapshot(
            cdefs={did: record.cdef for did, record in self._definitions_by_id.items()},
            stored_ids=self.all_stored_ids(),
            outgoing={
                parent_id: tuple(self._edge_by_key[key] for key in keys)
                for parent_id, keys in self._outgoing_edges.items()
            },
        )

    def _descendant_ids(self, owner_id: DefinitionId) -> set[DefinitionId]:
        self._check_active()
        seen: set[DefinitionId] = set()
        stack = [self._edge_by_key[key].child_id for key in sorted(
            self._outgoing_edges.get(owner_id, ()),
            key=lambda key: str(self._edge_by_key[key].path),
        )]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            child_edges = sorted(
                self._outgoing_edges.get(cur, ()),
                key=lambda key: str(self._edge_by_key[key].path),
            )
            stack.extend(self._edge_by_key[key].child_id for key in reversed(child_edges))
        return seen


class DefinitionCatalog:
    def __init__(self, repo):
        self.repo = repo
        self.lock = RLock()
        self.definitions_by_id: dict[DefinitionId, DefinitionRecord] = {}
        self.ids_by_cdef: dict[ConcreteDefinition, DefinitionId] = {}
        self.ids_by_stable_hash: dict[str, set[DefinitionId]] = defaultdict(set)
        self.replicas_by_definition: dict[DefinitionId, set[StoreId]] = defaultdict(set)
        self.stored_definitions_by_store: dict[StoreId, set[DefinitionId]] = defaultdict(set)
        self.local_postings: dict[FeatureToken, dict[DefinitionId, int]] = defaultdict(dict)
        self.edge_by_key: dict[EdgeKey, DefinitionEdgeRecord] = {}
        self.outgoing_edges: dict[DefinitionId, set[EdgeKey]] = defaultdict(set)
        self.incoming_edges: dict[DefinitionId, set[EdgeKey]] = defaultdict(set)
        self.child_by_parent_path: dict[tuple[DefinitionId, DefinitionPath], set[DefinitionId]] = defaultdict(set)
        self.parents_by_child_path: dict[tuple[DefinitionId, DefinitionPath], set[DefinitionId]] = defaultdict(set)
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

    @contextmanager
    def read_view(self, *, include_cached: bool = True):
        with self.lock:
            view = self._read_view_locked(include_cached=include_cached)
            try:
                yield view
            finally:
                view.close()

    def _read_view_locked(self, *, include_cached: bool) -> MemoryDefinitionGraphReadView:
        strong_cached_ids, weak_cached_ids = self._cached_id_sets_locked() if include_cached else (set(), set())
        return MemoryDefinitionGraphReadView(
            generation=self.generation,
            definitions_by_id=self.definitions_by_id,
            ids_by_cdef=self.ids_by_cdef,
            ids_by_stable_hash=self.ids_by_stable_hash,
            replicas_by_definition=self.replicas_by_definition,
            local_postings=self.local_postings,
            edge_by_key=self.edge_by_key,
            outgoing_edges=self.outgoing_edges,
            incoming_edges=self.incoming_edges,
            child_by_parent_path=self.child_by_parent_path,
            parents_by_child_path=self.parents_by_child_path,
            strong_cached_ids=strong_cached_ids,
            weak_cached_ids=weak_cached_ids,
            source_key="memory-catalog",
            repo=self.repo,
            store_by_id=self.store_by_id,
        )

    @property
    def source_key(self) -> str:
        return "memory-catalog"

    def current_generation(self) -> int:
        return self.generation

    def _cached_id_sets_locked(self) -> tuple[set[DefinitionId], set[DefinitionId]]:
        strong_cached_ids = {
            self.ids_by_cdef[cdef]
            for cdef in self.repo.strong_obj_cache.keys()
            if cdef in self.ids_by_cdef
        }
        weak_cached_ids = {
            self.ids_by_cdef[cdef]
            for cdef in self.repo.weak_obj_cache.keys()
            if cdef in self.ids_by_cdef
        }
        return strong_cached_ids, weak_cached_ids

    def sync_caches(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        with self.lock:
            ids: set[DefinitionId] = set()
            cdefs = list(self.repo.strong_obj_cache.keys())
            if reuse_weak:
                cdefs.extend(self.repo.weak_obj_cache.keys())
            cdefs = list(dict.fromkeys(cdefs))
            if cdefs:
                graph = ConcreteDefinitionGraph.from_roots(cdefs)
                self._register_graph_structure_locked(graph)
                for cdef in graph.roots:
                    ids.add(self.ids_by_cdef[cdef])
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
                if not self._auto_hydrates_store(store):
                    continue
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
                self.register_stored(cdef, store)
                found = True
        if found and stats is not None:
            stats.fast_path = "exact-root-store-has"
        return found

    def register_cached(self, cdef: ConcreteDefinition) -> DefinitionId:
        with self.lock:
            graph = ConcreteDefinitionGraph.from_root(cdef)
            self._register_graph_structure_locked(graph)
            return self.ids_by_cdef[cdef]

    def register_graph(self, graph: ConcreteDefinitionGraph) -> tuple[DefinitionId, ...]:
        with self.lock:
            changed = self._register_graph_structure_locked(graph)
            ids = [self.ids_by_cdef[root] for root in graph.roots]
            if changed:
                self.generation += 1
            return tuple(ids)

    def register_stored(
            self,
            cdef: ConcreteDefinition,
            store,
            *,
            graph: ConcreteDefinitionGraph | None = None) -> DefinitionId:
        with self.lock:
            sid = self.store_id(store)
            if graph is None:
                graph = ConcreteDefinitionGraph.from_root(cdef)
            changed = self._register_graph_structure_locked(graph)
            did = self.ids_by_cdef[cdef]
            membership_changed = (
                sid not in self.replicas_by_definition.get(did, set())
                or did not in self.stored_definitions_by_store.get(sid, set())
                or cdef not in self.repo.light_index
            )
            self.replicas_by_definition[did].add(sid)
            self.stored_definitions_by_store[sid].add(did)
            self.repo.light_index.add(cdef)
            if membership_changed or changed:
                self.generation += 1
            return did

    def register_stored_root(self, cdef: ConcreteDefinition, store) -> DefinitionId:
        with self.lock:
            sid = self.store_id(store)
            before_count = len(self.definitions_by_id)
            did = self._register_definition_locked(cdef)
            membership_changed = (
                sid not in self.replicas_by_definition.get(did, set())
                or did not in self.stored_definitions_by_store.get(sid, set())
                or cdef not in self.repo.light_index
            )
            self.replicas_by_definition[did].add(sid)
            self.stored_definitions_by_store[sid].add(did)
            self.repo.light_index.add(cdef)
            if membership_changed or len(self.definitions_by_id) != before_count:
                self.generation += 1
            return did

    def register_stored_roots(
            self,
            cdefs,
            store,
            *,
            graph: ConcreteDefinitionGraph | None = None) -> tuple[DefinitionId, ...]:
        roots = tuple(dict.fromkeys(cdefs))
        if not roots:
            return ()
        if graph is None:
            graph = ConcreteDefinitionGraph.from_roots(roots)
        with self.lock:
            sid = self.store_id(store)
            changed = self._register_graph_structure_locked(graph)
            membership_changed = False
            ids = []
            for cdef in roots:
                did = self.ids_by_cdef[cdef]
                ids.append(did)
                if (
                        sid not in self.replicas_by_definition.get(did, set())
                        or did not in self.stored_definitions_by_store.get(sid, set())
                        or cdef not in self.repo.light_index):
                    membership_changed = True
                self.replicas_by_definition[did].add(sid)
                self.stored_definitions_by_store[sid].add(did)
                self.repo.light_index.add(cdef)
            if membership_changed or changed:
                self.generation += 1
            return tuple(ids)

    def remove_stored_roots(self, cdefs, store) -> IndexWriteResult:
        roots = tuple(dict.fromkeys(cdefs))
        if not roots:
            return IndexWriteResult(generation=self.current_generation(), changed=False)
        with self.lock:
            sid = self.store_id(store)
            removed = 0
            for cdef in roots:
                did = self._definition_id_for_exact_cdef_locked(cdef)
                if did is None:
                    continue
                if sid in self.replicas_by_definition.get(did, set()):
                    self.replicas_by_definition[did].remove(sid)
                    removed += 1
                if did in self.stored_definitions_by_store.get(sid, set()):
                    self.stored_definitions_by_store[sid].remove(did)
                if not self.replicas_by_definition.get(did):
                    self.repo.light_index.discard(self.definitions_by_id[did].cdef)
            if removed:
                self.generation += 1
            return IndexWriteResult(
                generation=self.generation,
                changed=bool(removed),
                roots_removed=removed,
            )

    def cdef_id(self, cdef: ConcreteDefinition) -> DefinitionId | None:
        with self.read_view(include_cached=False) as view:
            return view.cdef_id(cdef)

    def exact_ids(self, cdef: ConcreteDefinition) -> set[DefinitionId]:
        with self.read_view(include_cached=False) as view:
            return view.exact_ids(cdef)

    def record_for_cdef(self, cdef: ConcreteDefinition) -> DefinitionRecord | None:
        with self.read_view(include_cached=False) as view:
            return view.record_for_cdef(cdef)

    def all_stored_ids(self) -> set[DefinitionId]:
        with self.read_view(include_cached=False) as view:
            return view.all_stored_ids()

    def stored_count(self) -> int:
        return len(self.all_stored_ids())

    def is_stored_id(self, did: DefinitionId) -> bool:
        with self.read_view(include_cached=False) as view:
            return view.is_stored_id(did)

    def filter_stored_ids(self, ids) -> set[DefinitionId]:
        with self.read_view(include_cached=False) as view:
            return view.filter_stored_ids(ids)

    def all_known_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        self.sync_caches(reuse_weak=reuse_weak)
        with self.read_view(include_cached=True) as view:
            return view.all_known_ids(reuse_weak=reuse_weak)

    def known_count(self, *, reuse_weak: bool = True) -> int:
        return len(self.all_known_ids(reuse_weak=reuse_weak))

    def all_cached_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        self.sync_caches(reuse_weak=reuse_weak)
        with self.read_view(include_cached=True) as view:
            return view.all_cached_ids(reuse_weak=reuse_weak)

    def cached_count(self, *, reuse_weak: bool = True) -> int:
        return len(self.all_cached_ids(reuse_weak=reuse_weak))

    def is_cached_id(self, did: DefinitionId, *, reuse_weak: bool = True) -> bool:
        self.sync_caches(reuse_weak=reuse_weak)
        with self.read_view(include_cached=True) as view:
            return view.is_cached_id(did, reuse_weak=reuse_weak)

    def nested_ids(self) -> set[DefinitionId]:
        with self.read_view(include_cached=False) as view:
            return view.nested_ids()

    def all_occurrences(self) -> tuple[DefinitionOccurrence, ...]:
        return tuple(self.iter_all_occurrences())

    def iter_all_occurrences(self, *, max_occurrences: int | None = None):
        with self.read_view(include_cached=False) as snapshot:
            traversal = snapshot.all_occurrence_snapshot()
        return traversal.iter_occurrences(max_occurrences=max_occurrences)

    def occurrences_for_nested_ids(self, ids: set[DefinitionId]) -> tuple[DefinitionOccurrence, ...]:
        return tuple(self.iter_occurrences_for_nested_ids(ids))

    def iter_occurrences_for_nested_ids(
            self,
            ids: set[DefinitionId],
            *,
            max_occurrences: int | None = None):
        with self.read_view(include_cached=False) as snapshot:
            traversal = snapshot.occurrence_snapshot_for_nested_ids(ids)
        return traversal.iter_occurrences(max_occurrences=max_occurrences)

    def filter_nested_ids(self, ids: set[DefinitionId]) -> set[DefinitionId]:
        with self.read_view(include_cached=False) as view:
            return view.filter_nested_ids(ids)

    def has_stored_ancestor(self, did: DefinitionId) -> bool:
        with self.read_view(include_cached=False) as view:
            return view.has_stored_ancestor(did)

    def all_definition_ids(self) -> set[DefinitionId]:
        with self.read_view(include_cached=False) as view:
            return view.all_definition_ids()

    def definition_for_id(self, did: DefinitionId) -> ConcreteDefinition:
        with self.read_view(include_cached=False) as view:
            return view.definition_for_id(did)

    def ids_to_cdefs(self, ids: set[DefinitionId] | frozenset[DefinitionId]) -> tuple[ConcreteDefinition, ...]:
        with self.read_view(include_cached=False) as view:
            return view.ids_to_cdefs(ids)

    def stores_for_cdef(self, cdef: ConcreteDefinition) -> tuple[Any, ...]:
        with self.read_view(include_cached=False) as view:
            did = view.cdef_id(cdef)
            if did is None:
                return ()
            return view.replica_map({did}).get(cdef, ())

    def local_candidates(
            self,
            requirements: tuple[FeatureRequirement, ...],
            *,
            within: set[DefinitionId] | None = None,
            domain=None,
            stats: QueryStats | None = None) -> set[DefinitionId]:
        with self.read_view(include_cached=False) as view:
            return view.local_candidates(requirements, within=within, domain=domain, stats=stats)

    def estimate_local_candidates(self, requirements: tuple[FeatureRequirement, ...]) -> int:
        with self.read_view(include_cached=False) as view:
            return view.estimate_local_candidates(requirements)

    def estimate_exact_ids(self, cdef: ConcreteDefinition) -> int:
        with self.read_view(include_cached=False) as view:
            return view.estimate_exact_ids(cdef)

    def parents(
            self,
            child_ids: set[DefinitionId],
            path: DefinitionPath,
            *,
            unordered: bool,
            within: set[DefinitionId] | frozenset[DefinitionId] | None = None) -> set[DefinitionId]:
        with self.read_view(include_cached=False) as view:
            return view.parents(child_ids, path, unordered=unordered, within=within)

    def children(
            self,
            parent_ids: set[DefinitionId],
            path: DefinitionPath,
            *,
            unordered: bool,
            within: set[DefinitionId] | frozenset[DefinitionId] | None = None) -> set[DefinitionId]:
        with self.read_view(include_cached=False) as view:
            return view.children(parent_ids, path, unordered=unordered, within=within)

    def _hydrate_stores(self, stores, *, stats: QueryStats | None = None) -> None:
        build_repo = _CatalogBuildRepo(self.repo)
        staged = DefinitionCatalog(build_repo)
        store_scan_count = 0
        for store in stores:
            sid, cdefs = self._hydrate_store_definitions(store)
            staged.store_by_id.setdefault(sid, store)
            staged.hydrated_stores.add(sid)
            staged.register_stored_roots(cdefs, store)
            store_scan_count += 1

        with self.lock:
            self._merge_staged_locked(staged, light_index=build_repo.light_index)
        if stats is not None:
            stats.store_scan_count += store_scan_count
            stats.refresh_action = "hydrate"

    def _rebuild_from_stores(self, *, stats: QueryStats | None = None) -> None:
        with self.lock:
            stores = self._unique_repo_stores()
            cached_cdefs = list(self.repo.strong_obj_cache.keys()) + list(self.repo.weak_obj_cache.keys())
            build_repo = _CatalogBuildRepo(self.repo)
            replacement = DefinitionCatalog(build_repo)
            if cached_cdefs:
                replacement.register_graph(ConcreteDefinitionGraph.from_roots(cached_cdefs))
            store_scan_count = 0
            for store in stores:
                if self._store_uses_persistent_sqlite(store):
                    continue
                self._check_memory_hydration_allowed(store, explicit=True)
                sid, cdefs = self._hydrate_store_definitions(store, explicit=True)
                replacement.store_by_id.setdefault(sid, store)
                replacement.hydrated_stores.add(sid)
                replacement.register_stored_roots(cdefs, store)
                store_scan_count += 1
            self._replace_with_locked(replacement, light_index=build_repo.light_index)
        if stats is not None:
            stats.store_scan_count += store_scan_count
            stats.refresh_action = "forced-refresh"

    def _hydrate_store_definitions(self, store, *, explicit: bool = False) -> tuple[StoreId, tuple[ConcreteDefinition, ...]]:
        self._check_memory_hydration_allowed(store, explicit=explicit)
        sid = self._store_key(store)
        try:
            cdefs = tuple(store.hydrate_index())
        except Exception as e:
            raise QueryIndexError(f"Failed to hydrate query index from store {store!r}: {e}") from e
        for cdef in cdefs:
            if not isinstance(cdef, ConcreteDefinition):
                raise QueryIndexError(f"Store {store!r} yielded {type(cdef).__name__}, not ConcreteDefinition.")
        return sid, cdefs

    def _auto_hydrates_store(self, store) -> bool:
        policy = getattr(store, "query_index_policy", "memory")
        if policy == "none":
            return False
        if self._store_uses_persistent_sqlite(store):
            return False
        self._check_memory_hydration_allowed(store, explicit=False)
        return True

    def _check_memory_hydration_allowed(self, store, *, explicit: bool) -> None:
        policy = getattr(store, "query_index_policy", "memory")
        if policy == "sqlite":
            return
        if policy == "auto" and self._store_uses_persistent_sqlite(store):
            return
        if policy == "none" and not explicit:
            raise QueryIndexUnavailable(
                f"Store {store!r} is configured with query_index='none'; use refresh=True for an explicit scan."
            )

    def _store_uses_persistent_sqlite(self, store) -> bool:
        policy = getattr(store, "query_index_policy", "memory")
        if policy == "sqlite":
            return True
        if policy != "auto":
            return False
        try:
            from .sqlite import sqlite_available
        except Exception:
            return False
        return sqlite_available()

    def _replace_with_locked(self, replacement: "DefinitionCatalog", *, light_index: set[ConcreteDefinition]) -> None:
        self.definitions_by_id = replacement.definitions_by_id
        self.ids_by_cdef = replacement.ids_by_cdef
        self.ids_by_stable_hash = replacement.ids_by_stable_hash
        self.replicas_by_definition = replacement.replicas_by_definition
        self.stored_definitions_by_store = replacement.stored_definitions_by_store
        self.local_postings = replacement.local_postings
        self.edge_by_key = replacement.edge_by_key
        self.outgoing_edges = replacement.outgoing_edges
        self.incoming_edges = replacement.incoming_edges
        self.child_by_parent_path = replacement.child_by_parent_path
        self.parents_by_child_path = replacement.parents_by_child_path
        self.store_by_id = replacement.store_by_id
        self.hydrated_stores = replacement.hydrated_stores
        self.repo.light_index.clear()
        self.repo.light_index.update(light_index)
        self.generation += 1

    def _merge_staged_locked(self, staged: "DefinitionCatalog", *, light_index: set[ConcreteDefinition]) -> None:
        id_map: dict[DefinitionId, DefinitionId] = {}
        changed = False
        for staged_id, record in staged.definitions_by_id.items():
            live_id, did_change = self._register_definition_record_locked(record)
            id_map[staged_id] = live_id
            changed = changed or did_change

        for edge in staged.edge_by_key.values():
            parent_id = id_map[edge.parent_id]
            child_id = id_map[edge.child_id]
            key: EdgeKey = (parent_id, edge.path, child_id)
            if key not in self.edge_by_key:
                self.edge_by_key[key] = DefinitionEdgeRecord(key, parent_id, edge.path, child_id)
                self.outgoing_edges[parent_id].add(key)
                self.incoming_edges[child_id].add(key)
                self.child_by_parent_path[(parent_id, edge.path)].add(child_id)
                self.parents_by_child_path[(child_id, edge.path)].add(parent_id)
                changed = True

        for sid, store in staged.store_by_id.items():
            if sid not in self.store_by_id:
                self.store_by_id[sid] = store
                changed = True

        for sid in staged.hydrated_stores:
            if sid not in self.hydrated_stores:
                self.hydrated_stores.add(sid)
                changed = True

        for sid, staged_ids in staged.stored_definitions_by_store.items():
            for staged_id in staged_ids:
                live_id = id_map[staged_id]
                cdef = self.definitions_by_id[live_id].cdef
                if (
                        sid not in self.replicas_by_definition.get(live_id, set())
                        or live_id not in self.stored_definitions_by_store.get(sid, set())
                        or cdef not in self.repo.light_index):
                    changed = True
                self.replicas_by_definition[live_id].add(sid)
                self.stored_definitions_by_store[sid].add(live_id)

        before_light_count = len(self.repo.light_index)
        self.repo.light_index.update(light_index)
        changed = changed or len(self.repo.light_index) != before_light_count
        if changed:
            self.generation += 1

    def _register_definition_locked(self, cdef: ConcreteDefinition) -> DefinitionId:
        existing = self.ids_by_cdef.get(cdef)
        if existing is not None:
            return existing

        digest = cdef.stable_hash()
        for did in self.ids_by_stable_hash.get(digest, set()):
            record = self.definitions_by_id[did]
            if cdef_equal(record.cdef, cdef):
                self.ids_by_cdef[cdef] = did
                return did

        if digest not in self.definitions_by_id:
            did = digest
        else:
            did = f"{digest}#{len(self.ids_by_stable_hash[digest])}"

        local_fingerprint = target_local_fingerprint(cdef)
        record = DefinitionRecord(
            definition_id=did,
            cdef=cdef,
            class_key=canonical_class_key(cdef.cls),
            local_fingerprint=local_fingerprint,
        )
        self.definitions_by_id[did] = record
        self.ids_by_cdef[cdef] = did
        self.ids_by_stable_hash[digest].add(did)
        for token, count in local_fingerprint.counts.items():
            self.local_postings[token][did] = count
        return did

    def _register_definition_record_locked(self, record: DefinitionRecord) -> tuple[DefinitionId, bool]:
        cdef = record.cdef
        existing = self.ids_by_cdef.get(cdef)
        if existing is not None:
            return existing, False

        digest = cdef.stable_hash()
        for did in self.ids_by_stable_hash.get(digest, set()):
            existing_record = self.definitions_by_id[did]
            if cdef_equal(existing_record.cdef, cdef):
                self.ids_by_cdef[cdef] = did
                return did, False

        if digest not in self.definitions_by_id:
            did = digest
        else:
            did = f"{digest}#{len(self.ids_by_stable_hash[digest])}"

        self.definitions_by_id[did] = DefinitionRecord(
            definition_id=did,
            cdef=cdef,
            class_key=record.class_key,
            local_fingerprint=record.local_fingerprint,
        )
        self.ids_by_cdef[cdef] = did
        self.ids_by_stable_hash[digest].add(did)
        for token, count in record.local_fingerprint.counts.items():
            self.local_postings[token][did] = count
        return did, True

    def _definition_id_for_exact_cdef_locked(self, cdef: ConcreteDefinition) -> DefinitionId | None:
        existing = self.ids_by_cdef.get(cdef)
        if existing is not None:
            return existing
        digest = cdef.stable_hash()
        for did in self.ids_by_stable_hash.get(digest, set()):
            if cdef_equal(self.definitions_by_id[did].cdef, cdef):
                self.ids_by_cdef[cdef] = did
                return did
        return None

    def _register_graph_structure_locked(self, graph: ConcreteDefinitionGraph) -> bool:
        before = (len(self.definitions_by_id), len(self.edge_by_key))
        for node in graph.nodes():
            self._register_definition_locked(node.definition)
        for edge in graph.edges():
            parent_id = self.ids_by_cdef[edge.parent]
            child_id = self.ids_by_cdef[edge.child]
            key: EdgeKey = (parent_id, edge.path, child_id)
            if key not in self.edge_by_key:
                self.edge_by_key[key] = DefinitionEdgeRecord(key, parent_id, edge.path, child_id)
                self.outgoing_edges[parent_id].add(key)
                self.incoming_edges[child_id].add(key)
                self.child_by_parent_path[(parent_id, edge.path)].add(child_id)
                self.parents_by_child_path[(child_id, edge.path)].add(parent_id)
        return (len(self.definitions_by_id), len(self.edge_by_key)) != before

    def _cached_cdefs(self, *, reuse_weak: bool = True) -> tuple[ConcreteDefinition, ...]:
        cdefs = list(self.repo.strong_obj_cache.keys())
        if reuse_weak:
            cdefs.extend(self.repo.weak_obj_cache.keys())
        return tuple(dict.fromkeys(cdefs))

def _candidate_ids_from_postings_data(
        definitions_by_id: dict[DefinitionId, DefinitionRecord],
        universe_ids: set[DefinitionId] | None,
        requirements: tuple[FeatureRequirement, ...],
        postings: dict[FeatureToken, dict[DefinitionId, int]],
        *,
        domain=None,
        stats: QueryStats | None = None) -> set[DefinitionId]:
    if not requirements:
        if domain is not None:
            candidates = domain.all_ids()
            if stats is not None:
                stats.candidate_count = len(candidates)
            return candidates
        if stats is not None:
            stats.candidate_count = len(universe_ids) if universe_ids is not None else len(definitions_by_id)
        return set(universe_ids) if universe_ids is not None else set(definitions_by_id)

    posting_data = []
    for req in requirements:
        posting = postings.get(req.token, {})
        posting_data.append((req, posting, len(posting)))
    posting_data.sort(key=lambda item: item[2])

    if stats is not None:
        stats.selected_features = tuple(req for req, _, _ in posting_data)
        stats.posting_sizes = tuple(size for _, _, size in posting_data)

    anchor_req, anchor_posting, _ = posting_data[0]
    candidates = {
        did
        for did, count in anchor_posting.items()
        if count >= anchor_req.count
        and (universe_ids is None or did in universe_ids)
        and (domain is None or domain.contains(did))
    }

    for req, posting, _ in posting_data[1:]:
        candidates = {
            did
            for did in candidates
            if posting.get(did, 0) >= req.count
        }
        if not candidates:
            break

    if stats is not None:
        stats.candidate_count = len(candidates)
    return candidates
