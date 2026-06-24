from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any

from ..cdef_identity import same_cdef
from ..cdef_graph import ConcreteDefinitionGraph
from ..definition import ConcreteDefinition
from .fingerprint import canonical_class_key, target_local_fingerprint
from .model import (
    DefinitionId,
    DefinitionEdgeRecord,
    DefinitionOccurrence,
    DefinitionRecord,
    EdgeKey,
    FeatureRequirement,
    FeatureToken,
    QueryIndexError,
    QueryStats,
    RefreshPolicy,
    StoreId,
)
from .path import DefinitionPath, GraphPath


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

    def register_stored_roots(self, cdefs, store) -> tuple[DefinitionId, ...]:
        roots = tuple(dict.fromkeys(cdefs))
        if not roots:
            return ()
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

    def cdef_id(self, cdef: ConcreteDefinition) -> DefinitionId | None:
        with self.lock:
            return self.ids_by_cdef.get(cdef)

    def exact_ids(self, cdef: ConcreteDefinition) -> set[DefinitionId]:
        with self.lock:
            return self._exact_ids_locked(cdef)

    def record_for_cdef(self, cdef: ConcreteDefinition) -> DefinitionRecord | None:
        with self.lock:
            did = self.ids_by_cdef.get(cdef)
            return self.definitions_by_id.get(did) if did is not None else None

    def stored_ids(self) -> set[DefinitionId]:
        return self.all_stored_ids()

    def all_stored_ids(self) -> set[DefinitionId]:
        with self.lock:
            return {did for did, stores in self.replicas_by_definition.items() if stores}

    def stored_count(self) -> int:
        with self.lock:
            return sum(1 for stores in self.replicas_by_definition.values() if stores)

    def is_stored_id(self, did: DefinitionId) -> bool:
        with self.lock:
            return self._is_stored_id_locked(did)

    def filter_stored_ids(self, ids) -> set[DefinitionId]:
        with self.lock:
            return {did for did in ids if self._is_stored_id_locked(did)}

    def known_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        cached = self.sync_caches(reuse_weak=reuse_weak)
        return self.all_stored_ids() | cached

    def all_known_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        cached = self.sync_caches(reuse_weak=reuse_weak)
        return self.all_stored_ids() | cached

    def known_count(self, *, reuse_weak: bool = True) -> int:
        with self.lock:
            ids = {did for did, stores in self.replicas_by_definition.items() if stores}
            for cdef in self._cached_cdefs(reuse_weak=reuse_weak):
                did = self.ids_by_cdef.get(cdef)
                if did is not None:
                    ids.add(did)
            return len(ids)

    def cached_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        return self.sync_caches(reuse_weak=reuse_weak)

    def all_cached_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        return self.sync_caches(reuse_weak=reuse_weak)

    def cached_count(self, *, reuse_weak: bool = True) -> int:
        return len(self._cached_cdefs(reuse_weak=reuse_weak))

    def is_cached_id(self, did: DefinitionId, *, reuse_weak: bool = True) -> bool:
        with self.lock:
            record = self.definitions_by_id.get(did)
        if record is None:
            return False
        return self.repo.get_cached(record.cdef, reuse_weak=reuse_weak) is not None

    def nested_ids(self) -> set[DefinitionId]:
        with self.lock:
            nested: set[DefinitionId] = set()
            for owner_id in self._stored_ids_locked():
                nested.update(self._descendant_ids_locked(owner_id))
            return nested

    def all_occurrences(self) -> tuple[DefinitionOccurrence, ...]:
        return tuple(self.iter_all_occurrences())

    def iter_all_occurrences(self, *, max_occurrences: int | None = None):
        def generate():
            with self.lock:
                yield from self._iter_occurrences_locked(max_occurrences=max_occurrences)

        return generate()

    def occurrences_for_nested_ids(self, ids: set[DefinitionId]) -> tuple[DefinitionOccurrence, ...]:
        return tuple(self.iter_occurrences_for_nested_ids(ids))

    def iter_occurrences_for_nested_ids(
            self,
            ids: set[DefinitionId],
            *,
            max_occurrences: int | None = None):
        def generate():
            with self.lock:
                yield from self._iter_occurrences_for_nested_ids_locked(ids, max_occurrences=max_occurrences)

        return generate()

    def owner_ids_for_nested_ids(self, ids: set[DefinitionId]) -> set[DefinitionId]:
        with self.lock:
            owners: set[DefinitionId] = set()
            seen: set[DefinitionId] = set(ids)
            stack = list(ids)
            while stack:
                cur = stack.pop()
                for edge_key in self.incoming_edges.get(cur, ()):
                    parent_id = self.edge_by_key[edge_key].parent_id
                    if self._is_stored_id_locked(parent_id):
                        owners.add(parent_id)
                    if parent_id not in seen:
                        seen.add(parent_id)
                        stack.append(parent_id)
            return owners

    def filter_nested_ids(self, ids: set[DefinitionId]) -> set[DefinitionId]:
        with self.lock:
            return {
                did
                for did in ids
                if self._has_stored_ancestor_locked(did)
            }

    def has_stored_ancestor(self, did: DefinitionId) -> bool:
        with self.lock:
            return self._has_stored_ancestor_locked(did)

    def all_definition_ids(self) -> set[DefinitionId]:
        with self.lock:
            return set(self.definitions_by_id)

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
        with self.lock:
            return self._candidate_ids_from_postings(
                universe_ids,
                requirements,
                self.local_postings,
                stats=stats,
            )

    def local_candidate_ids(
            self,
            universe_ids: set[DefinitionId],
            requirements: tuple[FeatureRequirement, ...],
            *,
            stats: QueryStats | None = None) -> set[DefinitionId]:
        return self._candidate_ids_from_postings(
            universe_ids,
            requirements,
            self.local_postings,
            stats=stats,
        )

    def local_candidate_ids_unbounded(
            self,
            requirements: tuple[FeatureRequirement, ...],
            *,
            stats: QueryStats | None = None) -> set[DefinitionId]:
        return self._candidate_ids_from_postings(
            None,
            requirements,
            self.local_postings,
            stats=stats,
        )

    def candidate_ids_for_domain(
            self,
            domain,
            requirements: tuple[FeatureRequirement, ...],
            *,
            stats: QueryStats | None = None) -> set[DefinitionId]:
        with self.lock:
            return self._candidate_ids_from_postings(
                None,
                requirements,
                self.local_postings,
                domain=domain,
                stats=stats,
            )

    def estimate_local_candidates(self, requirements: tuple[FeatureRequirement, ...]) -> int:
        with self.lock:
            if not requirements:
                return len(self.definitions_by_id)
            return min(len(self.local_postings.get(req.token, {})) for req in requirements)

    def estimate_exact_ids(self, cdef: ConcreteDefinition) -> int:
        with self.lock:
            digest = cdef.stable_hash()
            return sum(
                1
                for did in self.ids_by_stable_hash.get(digest, set())
                if same_cdef(self.definitions_by_id[did].cdef, cdef)
            )

    def parent_ids_for_children(self, child_ids: set[DefinitionId], path: DefinitionPath, *, unordered: bool) -> set[DefinitionId]:
        with self.lock:
            if not child_ids:
                return set()
            out = set()
            if unordered:
                for child_id in child_ids:
                    for edge_key in self.incoming_edges.get(child_id, ()):
                        record = self.edge_by_key[edge_key]
                        if record.path.startswith(path):
                            out.add(record.parent_id)
                return out
            for child_id in child_ids:
                out.update(self.parents_by_child_path.get((child_id, path), set()))
            return out

    def child_ids_for_parents(self, parent_ids: set[DefinitionId], path: DefinitionPath, *, unordered: bool) -> set[DefinitionId]:
        with self.lock:
            if not parent_ids:
                return set()
            out = set()
            if unordered:
                for parent_id in parent_ids:
                    for edge_key in self.outgoing_edges.get(parent_id, ()):
                        record = self.edge_by_key[edge_key]
                        if record.path.startswith(path):
                            out.add(record.child_id)
                return out
            for parent_id in parent_ids:
                out.update(self.child_by_parent_path.get((parent_id, path), set()))
            return out

    def parent_ids_with_matching_child(
            self,
            parent_ids: set[DefinitionId],
            child_ids: set[DefinitionId],
            path: DefinitionPath,
            *,
            unordered: bool) -> set[DefinitionId]:
        with self.lock:
            if not parent_ids or not child_ids:
                return set()
            out = set()
            if unordered:
                for parent_id in parent_ids:
                    for edge_key in self.outgoing_edges.get(parent_id, ()):
                        record = self.edge_by_key[edge_key]
                        if record.child_id in child_ids and record.path.startswith(path):
                            out.add(parent_id)
                            break
                return out
            for parent_id in parent_ids:
                if self.child_by_parent_path.get((parent_id, path), set()) & child_ids:
                    out.add(parent_id)
            return out

    def child_ids_with_matching_parent(
            self,
            parent_ids: set[DefinitionId],
            child_ids: set[DefinitionId],
            path: DefinitionPath,
            *,
            unordered: bool) -> set[DefinitionId]:
        with self.lock:
            if not parent_ids or not child_ids:
                return set()
            out = set()
            if unordered:
                for child_id in child_ids:
                    for edge_key in self.incoming_edges.get(child_id, ()):
                        record = self.edge_by_key[edge_key]
                        if record.parent_id in parent_ids and record.path.startswith(path):
                            out.add(child_id)
                            break
                return out
            for child_id in child_ids:
                if self.parents_by_child_path.get((child_id, path), set()) & parent_ids:
                    out.add(child_id)
            return out

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
                self.register_stored_roots(cdefs, store)
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
                self.local_postings.clear()
                self.edge_by_key.clear()
                self.outgoing_edges.clear()
                self.incoming_edges.clear()
                self.child_by_parent_path.clear()
                self.parents_by_child_path.clear()
                self.store_by_id.clear()
                self.hydrated_stores.clear()
                self.repo.light_index.clear()
                self.generation += 1

            if cached_cdefs:
                self.register_graph(ConcreteDefinitionGraph.from_roots(cached_cdefs))
            for store, sid, cdefs in collected:
                with self.lock:
                    self.store_by_id.setdefault(sid, store)
                    self.hydrated_stores.add(sid)
                if stats is not None:
                    stats.store_scan_count += 1
                self.register_stored_roots(cdefs, store)
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
            if same_cdef(record.cdef, cdef):
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

    def _exact_ids_locked(self, cdef: ConcreteDefinition) -> set[DefinitionId]:
        digest = cdef.stable_hash()
        return {
            did
            for did in self.ids_by_stable_hash.get(digest, set())
            if same_cdef(self.definitions_by_id[did].cdef, cdef)
        }

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

    def _candidate_ids_from_postings(
            self,
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
                stats.candidate_count = len(universe_ids) if universe_ids is not None else len(self.definitions_by_id)
            return set(universe_ids) if universe_ids is not None else set(self.definitions_by_id)

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

    def _cached_cdefs(self, *, reuse_weak: bool = True) -> tuple[ConcreteDefinition, ...]:
        cdefs = list(self.repo.strong_obj_cache.keys())
        if reuse_weak:
            cdefs.extend(self.repo.weak_obj_cache.keys())
        return tuple(dict.fromkeys(cdefs))

    def _stored_ids_locked(self) -> set[DefinitionId]:
        return {did for did, stores in self.replicas_by_definition.items() if stores}

    def _is_stored_id_locked(self, did: DefinitionId) -> bool:
        return bool(self.replicas_by_definition.get(did))

    def _descendant_ids_locked(self, owner_id: DefinitionId) -> set[DefinitionId]:
        seen: set[DefinitionId] = set()
        stack = [self.edge_by_key[key].child_id for key in sorted(
            self.outgoing_edges.get(owner_id, ()),
            key=lambda key: str(self.edge_by_key[key].path),
        )]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            child_edges = sorted(
                self.outgoing_edges.get(cur, ()),
                key=lambda key: str(self.edge_by_key[key].path),
            )
            stack.extend(self.edge_by_key[key].child_id for key in reversed(child_edges))
        return seen

    def _has_stored_ancestor_locked(self, did: DefinitionId) -> bool:
        seen: set[DefinitionId] = set()
        stack = [self.edge_by_key[key].parent_id for key in self.incoming_edges.get(did, ())]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            if self._is_stored_id_locked(cur):
                return True
            seen.add(cur)
            stack.extend(self.edge_by_key[key].parent_id for key in self.incoming_edges.get(cur, ()))
        return False

    def _iter_occurrences_locked(
            self,
            *,
            target_ids: set[DefinitionId] | None = None,
            max_occurrences: int | None = None):
        yielded = 0
        if max_occurrences is not None and max_occurrences <= 0:
            return
        for owner_id in sorted(self._stored_ids_locked()):
            stack = []
            for key in sorted(self.outgoing_edges.get(owner_id, ()), key=lambda key: str(self.edge_by_key[key].path), reverse=True):
                edge = self.edge_by_key[key]
                stack.append((edge.child_id, edge.path))
            while stack:
                did, path = stack.pop()
                if target_ids is None or did in target_ids:
                    yielded += 1
                    yield self._occurrence_locked(owner_id, path, did)
                    if max_occurrences is not None and yielded >= max_occurrences:
                        return
                for key in sorted(self.outgoing_edges.get(did, ()), key=lambda key: str(self.edge_by_key[key].path), reverse=True):
                    edge = self.edge_by_key[key]
                    stack.append((edge.child_id, path.join(edge.path)))

    def _iter_occurrences_for_nested_ids_locked(
            self,
            target_ids: set[DefinitionId],
            *,
            max_occurrences: int | None = None):
        yielded = 0
        if max_occurrences is not None and max_occurrences <= 0:
            return
        for target_id in sorted(target_ids):
            if target_id not in self.definitions_by_id:
                continue
            stack = [(target_id, GraphPath())]
            while stack:
                cur_id, suffix = stack.pop()
                incoming = sorted(
                    self.incoming_edges.get(cur_id, ()),
                    key=lambda key: (self.edge_by_key[key].parent_id, str(self.edge_by_key[key].path)),
                    reverse=True,
                )
                for edge_key in incoming:
                    edge = self.edge_by_key[edge_key]
                    path = edge.path.join(suffix)
                    if self._is_stored_id_locked(edge.parent_id):
                        yielded += 1
                        yield self._occurrence_locked(edge.parent_id, path, target_id)
                        if max_occurrences is not None and yielded >= max_occurrences:
                            return
                    stack.append((edge.parent_id, path))

    def _occurrence_locked(
            self,
            owner_id: DefinitionId,
            path: DefinitionPath,
            definition_id: DefinitionId) -> DefinitionOccurrence:
        return DefinitionOccurrence(
            self.definitions_by_id[owner_id].cdef,
            path,
            self.definitions_by_id[definition_id].cdef,
        )

    def _snapshot_locked(self):
        return {
            "definitions_by_id": dict(self.definitions_by_id),
            "ids_by_cdef": dict(self.ids_by_cdef),
            "ids_by_stable_hash": defaultdict(set, {k: set(v) for k, v in self.ids_by_stable_hash.items()}),
            "replicas_by_definition": defaultdict(set, {k: set(v) for k, v in self.replicas_by_definition.items()}),
            "stored_definitions_by_store": defaultdict(set, {k: set(v) for k, v in self.stored_definitions_by_store.items()}),
            "local_postings": defaultdict(dict, {k: dict(v) for k, v in self.local_postings.items()}),
            "edge_by_key": dict(self.edge_by_key),
            "outgoing_edges": defaultdict(set, {k: set(v) for k, v in self.outgoing_edges.items()}),
            "incoming_edges": defaultdict(set, {k: set(v) for k, v in self.incoming_edges.items()}),
            "child_by_parent_path": defaultdict(set, {k: set(v) for k, v in self.child_by_parent_path.items()}),
            "parents_by_child_path": defaultdict(set, {k: set(v) for k, v in self.parents_by_child_path.items()}),
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
        self.local_postings = defaultdict(dict, {k: dict(v) for k, v in snapshot["local_postings"].items()})
        self.edge_by_key = dict(snapshot["edge_by_key"])
        self.outgoing_edges = defaultdict(set, {k: set(v) for k, v in snapshot["outgoing_edges"].items()})
        self.incoming_edges = defaultdict(set, {k: set(v) for k, v in snapshot["incoming_edges"].items()})
        self.child_by_parent_path = defaultdict(set, {k: set(v) for k, v in snapshot["child_by_parent_path"].items()})
        self.parents_by_child_path = defaultdict(set, {k: set(v) for k, v in snapshot["parents_by_child_path"].items()})
        self.store_by_id = dict(snapshot["store_by_id"])
        self.hydrated_stores = set(snapshot["hydrated_stores"])
        self.generation = snapshot["generation"]
        self.repo.light_index.clear()
        self.repo.light_index.update(snapshot["light_index"])
