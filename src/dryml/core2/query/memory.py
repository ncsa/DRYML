from __future__ import annotations

from contextlib import contextmanager

from ..cdef_graph import EdgeKind
from .index import DefinitionCatalog, MemoryDefinitionGraphReadView
from .model import IndexWriteResult, OccurrenceTraversalSnapshot, QueryIndexStatus, ValidationReport


class AggregateMemoryQueryIndex(DefinitionCatalog):
    """Current in-process aggregate memory query index.

    This remains the compatibility implementation for cached/known domains and for
    mixed backend repos until Store-local memory sources and a separate cache
    overlay are split out.
    """


class MemoryStoreQueryIndex:
    def __init__(self, catalog: DefinitionCatalog, store):
        self.catalog = catalog
        self.store = store
        self._source_key = catalog.store_id(store)

    @property
    def source_key(self) -> str:
        return self._source_key

    @contextmanager
    def read_view(self, *, include_cached: bool = False):
        with self.catalog.read_view(include_cached=include_cached) as view:
            yield MemoryStoreReadView(view, store=self.store, store_id=self.source_key)

    def current_generation(self) -> int:
        return self.catalog.current_generation()

    def register_stored_roots(self, graph, roots) -> IndexWriteResult:
        before = self.current_generation()
        roots = tuple(roots)
        ids = self.catalog.register_stored_roots(roots, self.store, graph=graph)
        generation = self.current_generation()
        return IndexWriteResult(
            generation=generation,
            changed=generation != before,
            roots_added=0 if generation == before else len(ids),
        )

    def remove_stored_roots(self, roots) -> IndexWriteResult:
        return self.catalog.remove_stored_roots(tuple(roots), self.store)

    def refresh(self, policy, *, stats=None) -> None:
        self.catalog.refresh(policy, stats=stats)

    def status(self) -> QueryIndexStatus:
        return QueryIndexStatus(
            backend="memory",
            store_key=self.source_key,
            generation=self.current_generation(),
            schema_version=None,
            semantic_versions={},
            state="ready",
        )

    def validate(self, *, thorough: bool = False) -> ValidationReport:
        return ValidationReport("memory", self.source_key, True)

    def ensure_exact_stored(self, cdef, *, stats=None) -> bool:
        return self.catalog.ensure_exact_stored(cdef, stats=stats)

    def sync_caches(self, *, reuse_weak: bool = True) -> None:
        self.catalog.sync_caches(reuse_weak=reuse_weak)

    def close(self) -> None:
        return None


class MemoryStoreReadView:
    def __init__(self, view: MemoryDefinitionGraphReadView, *, store, store_id: str):
        self._view = view
        self._store = store
        self._store_id = store_id
        self.source_key = store_id

    @property
    def generation(self) -> int:
        return self._view.generation

    def _stored_ids(self) -> set:
        return self._view.stored_ids_for_store_id(self._store_id)

    def all_definition_ids(self):
        return self._view.all_definition_ids()

    def estimate_exact_ids(self, cdef):
        return self._view.estimate_exact_ids(cdef)

    def estimate_local_candidates(self, requirements):
        return self._view.estimate_local_candidates(requirements)

    def exact_ids(self, cdef):
        return self._view.exact_ids(cdef)

    def local_candidates(self, requirements, *, within=None, domain=None, stats=None):
        return self._view.local_candidates(requirements, within=within, domain=domain, stats=stats)

    def parents(self, child_ids, path, *, unordered: bool, edge_kind: EdgeKind = EdgeKind.MATERIALIZE, within=None):
        return self._view.parents(child_ids, path, unordered=unordered, edge_kind=edge_kind, within=within)

    def children(self, parent_ids, path, *, unordered: bool, edge_kind: EdgeKind = EdgeKind.MATERIALIZE, within=None):
        return self._view.children(parent_ids, path, unordered=unordered, edge_kind=edge_kind, within=within)

    def is_stored_id(self, did) -> bool:
        return did in self._stored_ids()

    def filter_stored_ids(self, ids) -> set:
        return set(ids) & self._stored_ids()

    def all_stored_ids(self) -> set:
        return self._stored_ids()

    def is_cached_id(self, did, *, reuse_weak: bool = True) -> bool:
        return False

    def all_cached_ids(self, *, reuse_weak: bool = True) -> set:
        return set()

    def all_known_ids(self, *, reuse_weak: bool = True) -> set:
        return self.all_stored_ids()

    def nested_ids(self) -> set:
        nested = set()
        stack = list(self.all_stored_ids())
        seen = set(stack)
        while stack:
            parent_id = stack.pop()
            children = self._view.children({parent_id}, _ROOT_PATH, unordered=True, edge_kind=EdgeKind.MATERIALIZE)
            for child_id in children:
                if child_id in seen:
                    continue
                seen.add(child_id)
                nested.add(child_id)
                stack.append(child_id)
        return nested

    def filter_nested_ids(self, ids) -> set:
        return {did for did in ids if self.has_stored_ancestor(did)}

    def has_stored_ancestor(self, did) -> bool:
        stored = self.all_stored_ids()
        seen = {did}
        stack = [did]
        while stack:
            child_id = stack.pop()
            parents = self._view.parents({child_id}, _ROOT_PATH, unordered=True, edge_kind=EdgeKind.MATERIALIZE)
            for parent_id in parents:
                if parent_id in stored:
                    return True
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                stack.append(parent_id)
        return False

    def cdefs_by_id(self, ids):
        return self._view.cdefs_by_id(ids)

    def replica_map(self, ids):
        cdefs = self._view.cdefs_by_id(self.filter_stored_ids(ids))
        return {cdef: (self._store,) for cdef in cdefs.values()}

    def project_owners(self, ids):
        snapshot = self.occurrence_snapshot_for_nested_ids(ids)
        return snapshot.project_owners(set(ids))

    def occurrence_snapshot_for_nested_ids(self, target_ids):
        base = self._view.occurrence_snapshot_for_nested_ids(target_ids)
        stored_ids = self.all_stored_ids()
        owner_replicas = self.replica_map(stored_ids)
        return OccurrenceTraversalSnapshot(
            targets=set(target_ids),
            cdefs=base.cdefs,
            stored_ids=stored_ids,
            incoming=base.incoming,
            owner_replicas=owner_replicas,
            copy_data=False,
        )


from .path import DefinitionPath

_ROOT_PATH = DefinitionPath()


__all__ = ["AggregateMemoryQueryIndex", "DefinitionCatalog", "MemoryDefinitionGraphReadView", "MemoryStoreQueryIndex", "MemoryStoreReadView"]
