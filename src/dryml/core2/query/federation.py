from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..definition import ConcreteDefinition
from .domain import NestedDomain, StoredDomain
from .graph_plan import graph_candidate_ids
from .model import DefinitionOccurrence, QueryIndexError, QueryIndexStatus, QueryIndexUnavailable, QueryStats, RefreshPolicy
from .selector_graph import compile_selector_graph


CACHE_SOURCE_KEY = "repo-cache"


@dataclass(frozen=True, slots=True)
class StoreIndexBinding:
    store: Any
    source_key: str
    priority: int
    index: Any | None = None


@dataclass(frozen=True, slots=True)
class RepoGenerationVector:
    generations: Mapping[str, int]


class RepoQueryIndex:
    def __init__(self, repo):
        self.repo = repo
        self._opened_indexes: dict[str, Any] = {}
        self._bindings: tuple[StoreIndexBinding, ...] = ()
        self.refresh_bindings()

    @property
    def store_bindings(self) -> tuple[StoreIndexBinding, ...]:
        return self._bindings

    def refresh_bindings(self) -> tuple[StoreIndexBinding, ...]:
        bindings: list[StoreIndexBinding] = []
        seen: set[str] = set()
        for priority, store in enumerate(self.repo.stores):
            source_key = _store_source_key(store)
            if source_key in seen:
                continue
            seen.add(source_key)
            bindings.append(StoreIndexBinding(
                store=store,
                source_key=source_key,
                priority=priority,
                index=self._opened_indexes.get(source_key),
            ))
        self._bindings = tuple(bindings)
        return self._bindings

    def sources_for_domain(self, domain: str) -> tuple[StoreIndexBinding | str, ...]:
        self.refresh_bindings()
        if domain == "cached":
            return (CACHE_SOURCE_KEY,)
        if domain == "known":
            return (*self._bindings, CACHE_SOURCE_KEY)
        if domain in {"stored", "nested"}:
            return self._bindings
        return self._bindings

    def can_execute_query_domain(self, domain: str) -> bool:
        if domain not in {"stored", "nested"}:
            return False
        self.refresh_bindings()
        if not self._bindings:
            return False
        return all(getattr(binding.store, "query_index_policy", None) == "sqlite" for binding in self._bindings)

    def execute_definition_domain(self, query):
        stats = QueryStats(refresh_action="federated")
        selector_graph = compile_selector_graph(query.selector, class_match=query.class_match_policy)
        exact_root = query.selector if isinstance(query.selector, ConcreteDefinition) else None
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        replicas: dict[ConcreteDefinition, list[Any]] = {}

        for binding in self._executable_bindings("stored"):
            index = self.open_store_index(binding)
            index.refresh(query.refresh_policy, stats=stats)
            with index.read_view(include_cached=False) as snapshot:
                domain = StoredDomain(snapshot)
                if exact_root is not None:
                    candidate_ids = domain.filter(snapshot.exact_ids(exact_root))
                    stats.candidate_count += len(candidate_ids)
                elif selector_graph is not None:
                    candidate_ids = graph_candidate_ids(snapshot, selector_graph, domain, stats=stats)
                else:
                    candidate_ids = domain.all_ids()
                    stats.candidate_count += len(candidate_ids)
                cdefs_by_id = snapshot.cdefs_by_id(candidate_ids)
            matches = query._verify_cdefs(tuple(cdefs_by_id.values()), stats=stats)
            for cdef in matches:
                canonical = _canonical_cdef_key(merged, cdef)
                replicas.setdefault(canonical, []).append(binding.store)

        out = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        stats.result_count = len(out)
        stats.universe_size = None
        return out, stats, {cdef: tuple(replicas.get(cdef, ())) for cdef in out}

    def execute_nested_definitions(self, query):
        stats = QueryStats(refresh_action="federated")
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        for binding in self._executable_bindings("nested"):
            cdefs_by_id, _ = self._capture_nested_candidates(query, binding, stats)
            matches, _ = query._verify_cdefs_by_id(cdefs_by_id, stats=stats)
            for cdef in matches:
                _canonical_cdef_key(merged, cdef)
        out = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        stats.result_count = len(out)
        stats.universe_size = None
        return out, stats

    def execute_nested_owners(self, query):
        stats = QueryStats(refresh_action="federated")
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        replicas: dict[ConcreteDefinition, list[Any]] = {}
        for binding in self._executable_bindings("nested"):
            for _ in range(3):
                cdefs_by_id, generation = self._capture_nested_candidates(query, binding, stats)
                _, match_ids = query._verify_cdefs_by_id(cdefs_by_id, stats=stats)
                index = self.open_store_index(binding)
                with index.read_view(include_cached=False) as snapshot:
                    if snapshot.generation != generation:
                        continue
                    projection = snapshot.project_owners(match_ids)
                    break
            else:
                raise QueryIndexError("Catalog generation changed repeatedly during nested owner query.")
            for owner in projection.cdefs:
                canonical = _canonical_cdef_key(merged, owner)
                replicas.setdefault(canonical, []).append(binding.store)
        out = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        stats.result_count = len(out)
        stats.universe_size = None
        return out, stats, {cdef: tuple(replicas.get(cdef, ())) for cdef in out}

    def execute_nested_occurrences(self, query):
        stats = QueryStats(refresh_action="federated")
        occurrences: list[DefinitionOccurrence] = []
        owner_replicas: dict[ConcreteDefinition, list[Any]] = {}
        for binding in self._executable_bindings("nested"):
            for _ in range(3):
                cdefs_by_id, generation = self._capture_nested_candidates(query, binding, stats)
                _, match_ids = query._verify_cdefs_by_id(cdefs_by_id, stats=stats)
                index = self.open_store_index(binding)
                with index.read_view(include_cached=False) as snapshot:
                    if snapshot.generation != generation:
                        continue
                    traversal = snapshot.occurrence_snapshot_for_nested_ids(match_ids)
                    break
            else:
                raise QueryIndexError("Catalog generation changed repeatedly during nested occurrence query.")
            for occ in traversal.iter_occurrences(max_occurrences=query.occurrence_limit):
                occurrences.append(occ)
                owner_replicas.setdefault(occ.owner, []).append(binding.store)
                if query.occurrence_limit is not None and len(occurrences) >= query.occurrence_limit:
                    break
            if query.occurrence_limit is not None and len(occurrences) >= query.occurrence_limit:
                break
        stats.result_count = len(occurrences)
        stats.universe_size = None
        return tuple(occurrences), stats, {owner: tuple(_dedupe_stores(stores)) for owner, stores in owner_replicas.items()}

    def open_store_index(self, binding: StoreIndexBinding):
        existing = self._opened_indexes.get(binding.source_key)
        if existing is not None:
            return existing
        opener = getattr(binding.store, "open_query_index", None)
        if opener is None:
            return None
        index = opener()
        if index is not None:
            self._opened_indexes[binding.source_key] = index
            self.refresh_bindings()
        return index

    def index_status(self, store=None) -> tuple[QueryIndexStatus, ...]:
        bindings = self._bindings_for_store(store)
        statuses: list[QueryIndexStatus] = []
        for binding in bindings:
            statuses.append(self._status_for_binding(binding))
        return tuple(statuses)

    def generation_vector(self, *, include_store_indexes: bool = False) -> RepoGenerationVector:
        generations = {CACHE_SOURCE_KEY: self.repo._query_catalog.generation}
        if include_store_indexes:
            for status in self.index_status():
                if status.generation is not None:
                    generations[status.store_key] = status.generation
        return RepoGenerationVector(generations=generations)

    def refresh(self, policy: RefreshPolicy, *, stats: QueryStats | None = None) -> None:
        self.refresh_bindings()
        for binding in self._bindings:
            if getattr(binding.store, "query_index_policy", None) in {"memory", "none"}:
                continue
            index = self.open_store_index(binding)
            if index is None:
                continue
            refresh = getattr(index, "refresh", None)
            if refresh is None:
                continue
            try:
                refresh(policy, stats=stats)
            except QueryIndexUnavailable:
                continue

    def rebuild(self, store=None) -> None:
        for binding in self._bindings_for_store(store):
            if getattr(binding.store, "query_index_policy", None) in {"memory", "none"}:
                continue
            index = self.open_store_index(binding)
            if index is None:
                continue
            rebuild = getattr(index, "rebuild", None)
            if rebuild is None:
                continue
            rebuild()

    def register_saved_graph(self, graph, roots_by_store: Mapping[Any, Sequence[Any]]) -> None:
        self.refresh_bindings()
        binding_by_key = {binding.source_key: binding for binding in self._bindings}
        for store, roots in roots_by_store.items():
            roots = tuple(roots)
            if not roots:
                continue
            source_key = _store_source_key(store)
            binding = binding_by_key.get(source_key)
            if binding is None:
                continue
            if getattr(store, "query_index_policy", None) in {"auto", "memory", "none"}:
                continue
            index = self.open_store_index(binding)
            if index is None:
                continue
            register = getattr(index, "register_stored_roots", None)
            if register is None:
                continue
            try:
                register(graph, roots)
            except QueryIndexUnavailable:
                continue
            except Exception:
                marker = getattr(store, "mark_query_index_dirty", None)
                if marker is not None:
                    marker()
                raise

    def close(self) -> None:
        for index in tuple(self._opened_indexes.values()):
            close = getattr(index, "close", None)
            if close is not None:
                close()
        self._opened_indexes.clear()
        self.refresh_bindings()

    def _executable_bindings(self, domain: str) -> tuple[StoreIndexBinding, ...]:
        if not self.can_execute_query_domain(domain):
            raise QueryIndexUnavailable(f"Repo query federation cannot execute domain {domain!r} for current Store backends.")
        return self._bindings

    def _capture_nested_candidates(self, query, binding: StoreIndexBinding, stats: QueryStats):
        index = self.open_store_index(binding)
        index.refresh(query.refresh_policy, stats=stats)
        selector_graph = compile_selector_graph(query.selector, class_match=query.class_match_policy)
        with index.read_view(include_cached=False) as snapshot:
            if selector_graph is not None:
                candidate_ids = graph_candidate_ids(snapshot, selector_graph, None, stats=stats)
                candidate_ids = snapshot.filter_nested_ids(candidate_ids)
                stats.candidate_count += len(candidate_ids)
            else:
                domain = NestedDomain(snapshot)
                candidate_ids = domain.all_ids()
                stats.candidate_count += len(candidate_ids)
            cdefs_by_id = snapshot.cdefs_by_id(candidate_ids)
            generation = snapshot.generation
        return cdefs_by_id, generation

    def _bindings_for_store(self, store) -> tuple[StoreIndexBinding, ...]:
        self.refresh_bindings()
        if store is None:
            return self._bindings
        source_key = _store_source_key(store)
        return tuple(binding for binding in self._bindings if binding.source_key == source_key)

    def _status_for_binding(self, binding: StoreIndexBinding) -> QueryIndexStatus:
        policy = getattr(binding.store, "query_index_policy", "memory")
        if policy == "none":
            return QueryIndexStatus(
                backend="none",
                store_key=binding.source_key,
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="disabled",
            )
        if policy == "memory":
            return QueryIndexStatus(
                backend="memory",
                store_key=binding.source_key,
                generation=self.repo._query_catalog.generation,
                schema_version=None,
                semantic_versions={},
                state="ready",
            )
        try:
            index = self.open_store_index(binding)
        except QueryIndexUnavailable:
            return QueryIndexStatus(
                backend=str(policy),
                store_key=binding.source_key,
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="unavailable",
            )
        if index is None:
            return QueryIndexStatus(
                backend="memory",
                store_key=binding.source_key,
                generation=self.repo._query_catalog.generation,
                schema_version=None,
                semantic_versions={},
                state="ready",
            )
        status = getattr(index, "status", None)
        if status is None:
            return QueryIndexStatus(
                backend=type(index).__name__,
                store_key=binding.source_key,
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="ready",
            )
        return status()


def _store_source_key(store) -> str:
    catalog_key = getattr(store, "catalog_key", None)
    if catalog_key is not None:
        return catalog_key()
    return f"{type(store).__module__}.{type(store).__qualname__}:id:{id(store)}"


def _canonical_cdef_key(merged: dict[ConcreteDefinition, ConcreteDefinition], cdef: ConcreteDefinition) -> ConcreteDefinition:
    existing = merged.get(cdef)
    if existing is not None:
        return existing
    merged[cdef] = cdef
    return cdef


def _dedupe_stores(stores: Sequence[Any]) -> tuple[Any, ...]:
    out = []
    seen = set()
    for store in stores:
        key = _store_source_key(store)
        if key in seen:
            continue
        seen.add(key)
        out.append(store)
    return tuple(out)
