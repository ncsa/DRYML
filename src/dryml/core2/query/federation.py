from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..definition import ConcreteDefinition
from .domain import NestedDomain, StoredDomain
from .graph_plan import graph_candidate_ids
from .memory import MemoryStoreQueryIndex
from .model import QueryIndexError, QueryIndexStatus, QueryIndexUnavailable, QueryStats, RefreshPolicy, SourceQueryPlan, ValidationIssue, ValidationReport
from .selector_graph import compile_selector_graph
from .utils import chunked


CACHE_SOURCE_KEY = "repo-cache"


@dataclass(frozen=True, slots=True)
class StoreIndexBinding:
    store: Any
    source_key: str
    priority: int
    index: Any | None = None


@dataclass(frozen=True, slots=True)
class IndexGenerationVector:
    """Generation vector for a repo query-index snapshot."""

    generations: Mapping[str, int]


RepoGenerationVector = IndexGenerationVector


class RepoQueryIndex:
    def __init__(self, repo):
        self.repo = repo
        self._opened_indexes: dict[str, Any] = {}
        self._memory_indexes: dict[str, MemoryStoreQueryIndex] = {}
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
        has_non_memory_source = False
        for binding in self._bindings:
            policy = getattr(binding.store, "query_index_policy", "memory")
            if policy == "none":
                return False
            if policy == "memory":
                continue
            try:
                index = self.open_store_index(binding)
            except QueryIndexUnavailable:
                if policy == "auto":
                    continue
                raise
            if index is not None:
                has_non_memory_source = True
                continue
            if policy != "auto":
                return False
        if not has_non_memory_source:
            return False
        return True

    def execute_definition_domain(self, query, *, stop_after: int | None = None):
        stats = QueryStats(refresh_action="federated")
        selector_graph = compile_selector_graph(query.selector, class_match=query.class_match_policy)
        exact_root = query.selector if isinstance(query.selector, ConcreteDefinition) else None
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        replicas: dict[ConcreteDefinition, list[Any]] = {}
        source_plans: list[SourceQueryPlan] = []
        generations: dict[str, int] = {}

        for binding in self._executable_bindings("stored"):
            try:
                index = self._source_index_for_binding(binding)
                self._refresh_definition_source(index, query, exact_root, stats)
                before_verified = stats.verified_count
                source_matches = 0
                with index.read_view(include_cached=False) as snapshot:
                    candidate_ids = self._definition_candidate_ids(snapshot, query, selector_graph, exact_root, stats)
                    generation = snapshot.generation
                    generations[binding.source_key] = generation
                    if stop_after is None:
                        cdefs_by_id = snapshot.cdefs_by_id(candidate_ids)
                    else:
                        cdefs_by_id = {}
                        for batch in chunked(sorted(candidate_ids), 256):
                            batch_cdefs = snapshot.cdefs_by_id(batch)
                            batch_matches = query._verify_cdefs(tuple(batch_cdefs.values()), stats=stats)
                            source_matches += len(batch_matches)
                            for cdef in batch_matches:
                                canonical = _canonical_cdef_key(merged, cdef)
                                replicas.setdefault(canonical, []).append(binding.store)
                                if len(merged) >= stop_after:
                                    break
                            if len(merged) >= stop_after:
                                break
                matches = () if stop_after is not None else query._verify_cdefs(tuple(cdefs_by_id.values()), stats=stats)
                if stop_after is None:
                    source_matches = len(matches)
            except Exception as exc:
                raise _source_query_error(binding, exc) from exc
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(index),
                generation=generation,
                candidate_count=len(candidate_ids),
                verified_count=stats.verified_count - before_verified,
                result_count=source_matches,
                refresh_action=stats.refresh_action,
            ))
            for cdef in matches:
                canonical = _canonical_cdef_key(merged, cdef)
                replicas.setdefault(canonical, []).append(binding.store)
            if stop_after is not None and len(merged) >= stop_after:
                break

        out = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        stats.result_count = len(out)
        stats.universe_size = None
        stats.generation_vector = generations
        stats.source_plans = tuple(source_plans)
        return out, stats, {cdef: tuple(replicas.get(cdef, ())) for cdef in out}

    def count_definition_domain(self, query, *, stop_after: int | None = None) -> tuple[int, QueryStats]:
        stats = QueryStats(refresh_action="federated")
        selector_graph = compile_selector_graph(query.selector, class_match=query.class_match_policy)
        exact_root = query.selector if isinstance(query.selector, ConcreteDefinition) else None
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        source_plans: list[SourceQueryPlan] = []
        generations: dict[str, int] = {}

        for binding in self._executable_bindings("stored"):
            try:
                index = self._source_index_for_binding(binding)
                self._refresh_definition_source(index, query, exact_root, stats)
                source_matches = 0
                before_verified = stats.verified_count
                with index.read_view(include_cached=False) as snapshot:
                    candidate_ids = self._definition_candidate_ids(snapshot, query, selector_graph, exact_root, stats)
                    generation = snapshot.generation
                    generations[binding.source_key] = generation
                    for batch in chunked(sorted(candidate_ids), 256):
                        cdefs_by_id = snapshot.cdefs_by_id(batch)
                        matches = query._verify_cdefs(tuple(cdefs_by_id.values()), stats=stats)
                        source_matches += len(matches)
                        for cdef in matches:
                            _canonical_cdef_key(merged, cdef)
                            if stop_after is not None and len(merged) >= stop_after:
                                source_plans.append(SourceQueryPlan(
                                    source_key=binding.source_key,
                                    backend=_source_backend(index),
                                    generation=generation,
                                    candidate_count=len(candidate_ids),
                                    verified_count=stats.verified_count - before_verified,
                                    result_count=source_matches,
                                    refresh_action=stats.refresh_action,
                                ))
                                stats.result_count = len(merged)
                                stats.universe_size = None
                                stats.generation_vector = generations
                                stats.source_plans = tuple(source_plans)
                                return len(merged), stats
            except Exception as exc:
                raise _source_query_error(binding, exc) from exc
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(index),
                generation=generation,
                candidate_count=len(candidate_ids),
                verified_count=stats.verified_count - before_verified,
                result_count=source_matches,
                refresh_action=stats.refresh_action,
            ))

        stats.result_count = len(merged)
        stats.universe_size = None
        stats.generation_vector = generations
        stats.source_plans = tuple(source_plans)
        return len(merged), stats

    def explain_definition_domain(self, query) -> QueryStats:
        stats = QueryStats(refresh_action="federated-plan")
        selector_graph = compile_selector_graph(query.selector, class_match=query.class_match_policy)
        exact_root = query.selector if isinstance(query.selector, ConcreteDefinition) else None
        source_plans: list[SourceQueryPlan] = []
        generations: dict[str, int] = {}
        for binding in self._executable_bindings("stored"):
            try:
                index = self._source_index_for_binding(binding)
                self._refresh_definition_source(index, query, exact_root, stats)
                with index.read_view(include_cached=False) as snapshot:
                    before_candidates = stats.candidate_count
                    candidate_ids = self._definition_candidate_ids(snapshot, query, selector_graph, exact_root, stats)
                    generation = snapshot.generation
                    generations[binding.source_key] = generation
            except Exception as exc:
                raise _source_query_error(binding, exc) from exc
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(index),
                generation=generation,
                candidate_count=stats.candidate_count - before_candidates if stats.candidate_count >= before_candidates else len(candidate_ids),
                verified_count=0,
                result_count=None,
                refresh_action=stats.refresh_action,
            ))
        stats.universe_size = None
        stats.result_count = None
        stats.generation_vector = generations
        stats.source_plans = tuple(source_plans)
        return stats

    def execute_nested_definitions(self, query, *, stop_after: int | None = None):
        stats = QueryStats(refresh_action="federated")
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        source_plans: list[SourceQueryPlan] = []
        generations: dict[str, int] = {}
        for binding in self._executable_bindings("nested"):
            before_candidates = stats.candidate_count
            before_verified = stats.verified_count
            cdefs_by_id, generation = self._capture_nested_candidates(query, binding, stats)
            generations[binding.source_key] = generation
            matches, _ = query._verify_cdefs_by_id(cdefs_by_id, stats=stats)
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(self._source_index_for_binding(binding)),
                generation=generation,
                candidate_count=stats.candidate_count - before_candidates,
                verified_count=stats.verified_count - before_verified,
                result_count=len(matches),
                refresh_action=stats.refresh_action,
            ))
            for cdef in matches:
                _canonical_cdef_key(merged, cdef)
                if stop_after is not None and len(merged) >= stop_after:
                    break
            if stop_after is not None and len(merged) >= stop_after:
                break
        out = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        stats.result_count = len(out)
        stats.universe_size = None
        stats.generation_vector = generations
        stats.source_plans = tuple(source_plans)
        return out, stats

    def execute_nested_owners(self, query, *, stop_after: int | None = None):
        stats = QueryStats(refresh_action="federated")
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        replicas: dict[ConcreteDefinition, list[Any]] = {}
        source_plans: list[SourceQueryPlan] = []
        generations: dict[str, int] = {}
        for binding in self._executable_bindings("nested"):
            before_candidates = stats.candidate_count
            before_verified = stats.verified_count
            for _ in range(3):
                cdefs_by_id, generation = self._capture_nested_candidates(query, binding, stats)
                _, match_ids = query._verify_cdefs_by_id(cdefs_by_id, stats=stats)
                index = self._source_index_for_binding(binding)
                with index.read_view(include_cached=False) as snapshot:
                    if snapshot.generation != generation:
                        continue
                    projection = snapshot.project_owners(match_ids)
                    break
            else:
                raise QueryIndexError("Catalog generation changed repeatedly during nested owner query.")
            generations[binding.source_key] = generation
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(self._source_index_for_binding(binding)),
                generation=generation,
                candidate_count=stats.candidate_count - before_candidates,
                verified_count=stats.verified_count - before_verified,
                result_count=len(projection.cdefs),
                refresh_action=stats.refresh_action,
            ))
            for owner in projection.cdefs:
                canonical = _canonical_cdef_key(merged, owner)
                replicas.setdefault(canonical, []).append(binding.store)
                if stop_after is not None and len(merged) >= stop_after:
                    break
            if stop_after is not None and len(merged) >= stop_after:
                break
        out = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        stats.result_count = len(out)
        stats.universe_size = None
        stats.generation_vector = generations
        stats.source_plans = tuple(source_plans)
        return out, stats, {cdef: tuple(replicas.get(cdef, ())) for cdef in out}

    def execute_nested_occurrences(self, query):
        stats = QueryStats(refresh_action="federated")
        traversals = []
        owner_replicas: dict[ConcreteDefinition, list[Any]] = {}
        source_plans: list[SourceQueryPlan] = []
        generations: dict[str, int] = {}
        for binding in self._executable_bindings("nested"):
            before_candidates = stats.candidate_count
            before_verified = stats.verified_count
            for _ in range(3):
                cdefs_by_id, generation = self._capture_nested_candidates(query, binding, stats)
                _, match_ids = query._verify_cdefs_by_id(cdefs_by_id, stats=stats)
                index = self._source_index_for_binding(binding)
                with index.read_view(include_cached=False) as snapshot:
                    if snapshot.generation != generation:
                        continue
                    traversal = snapshot.occurrence_snapshot_for_nested_ids(match_ids)
                    break
            else:
                raise QueryIndexError("Catalog generation changed repeatedly during nested occurrence query.")
            traversals.append(traversal)
            for owner_id in traversal.stored_ids:
                owner = traversal.cdefs.get(owner_id)
                if owner is not None:
                    owner_replicas.setdefault(owner, []).append(binding.store)
            generations[binding.source_key] = generation
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(self._source_index_for_binding(binding)),
                generation=generation,
                candidate_count=stats.candidate_count - before_candidates,
                verified_count=stats.verified_count - before_verified,
                result_count=None,
                refresh_action=stats.refresh_action,
            ))
        stats.result_count = None
        stats.universe_size = None
        stats.generation_vector = generations
        stats.source_plans = tuple(source_plans)

        def occurrence_factory():
            yielded = 0
            seen = set()
            for traversal in traversals:
                for occurrence in traversal.iter_occurrences():
                    key = (occurrence.owner, occurrence.path, occurrence.definition)
                    if key in seen:
                        continue
                    seen.add(key)
                    yielded += 1
                    yield occurrence
                    if query.occurrence_limit is not None and yielded >= query.occurrence_limit:
                        return

        return occurrence_factory, stats, {owner: tuple(_dedupe_stores(stores)) for owner, stores in owner_replicas.items()}

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

    def _refresh_definition_source(self, index, query, exact_root: ConcreteDefinition | None, stats: QueryStats) -> None:
        if query.refresh_policy is True:
            index.refresh(True, stats=stats)
            return
        if exact_root is not None and query.refresh_policy is not False:
            ensure = getattr(index, "ensure_exact_stored", None)
            if ensure is not None:
                ensure(exact_root, stats=stats)
                return
        index.refresh(query.refresh_policy, stats=stats)

    def _source_index_for_binding(self, binding: StoreIndexBinding):
        policy = getattr(binding.store, "query_index_policy", "memory")
        if policy == "none":
            return None
        if policy == "memory":
            return self._memory_index_for_binding(binding)
        try:
            index = self.open_store_index(binding)
        except QueryIndexUnavailable:
            if policy == "auto":
                return self._memory_index_for_binding(binding)
            raise
        if index is not None:
            return index
        if policy == "auto":
            return self._memory_index_for_binding(binding)
        return None

    def _memory_index_for_binding(self, binding: StoreIndexBinding):
        existing = self._memory_indexes.get(binding.source_key)
        if existing is not None:
            return existing
        index = MemoryStoreQueryIndex(self.repo._query_catalog, binding.store)
        self._memory_indexes[binding.source_key] = index
        return index

    def index_status(self, store=None) -> tuple[QueryIndexStatus, ...]:
        bindings = self._bindings_for_store(store)
        statuses: list[QueryIndexStatus] = []
        for binding in bindings:
            statuses.append(self._status_for_binding(binding))
        return tuple(statuses)

    def validate(self, store=None, *, thorough: bool = False) -> tuple[ValidationReport, ...]:
        reports = []
        for binding in self._bindings_for_store(store):
            policy = getattr(binding.store, "query_index_policy", "memory")
            if policy == "none":
                reports.append(ValidationReport("none", binding.source_key, True))
                continue
            if policy == "memory":
                reports.append(ValidationReport("memory", binding.source_key, True))
                continue
            try:
                index = self.open_store_index(binding)
            except QueryIndexUnavailable as exc:
                reports.append(ValidationReport(str(policy), binding.source_key, False, (ValidationIssue("error", str(exc)),)))
                continue
            if index is None:
                reports.append(ValidationReport("memory", binding.source_key, True))
                continue
            validate = getattr(index, "validate", None)
            if validate is None:
                reports.append(ValidationReport(type(index).__name__, binding.source_key, True))
                continue
            reports.append(validate(thorough=thorough))
        return tuple(reports)

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
            if getattr(store, "query_index_policy", None) in {"memory", "none"}:
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
        self._memory_indexes.clear()
        self.refresh_bindings()

    def _executable_bindings(self, domain: str) -> tuple[StoreIndexBinding, ...]:
        if not self.can_execute_query_domain(domain):
            raise QueryIndexUnavailable(f"Repo query federation cannot execute domain {domain!r} for current Store backends.")
        return self._bindings

    def _capture_nested_candidates(self, query, binding: StoreIndexBinding, stats: QueryStats):
        index = self._source_index_for_binding(binding)
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

    def _definition_candidate_ids(self, snapshot, query, selector_graph, exact_root, stats: QueryStats):
        domain = StoredDomain(snapshot)
        if exact_root is not None:
            candidate_ids = domain.filter(snapshot.exact_ids(exact_root))
            stats.candidate_count += len(candidate_ids)
            return candidate_ids
        if selector_graph is not None:
            return graph_candidate_ids(snapshot, selector_graph, domain, stats=stats)
        candidate_ids = domain.all_ids()
        stats.candidate_count += len(candidate_ids)
        return candidate_ids

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


def _source_backend(index) -> str:
    if index is None:
        return "none"
    name = type(index).__name__
    if name == "SQLiteStoreQueryIndex":
        return "sqlite"
    if name == "MemoryStoreQueryIndex":
        return "memory"
    return name


def _source_query_error(binding: StoreIndexBinding, exc: BaseException) -> QueryIndexError:
    if isinstance(exc, QueryIndexError):
        return QueryIndexError(f"Query source {binding.source_key!r} failed: {exc}")
    return QueryIndexError(f"Query source {binding.source_key!r} failed with {type(exc).__name__}: {exc}")
