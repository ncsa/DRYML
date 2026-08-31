from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from ..definition import ConcreteDefinition
from .domain import KnownDomain, NestedDomain, StoredDomain
from .graph_plan import graph_candidate_ids
from .lowering import CollectSink, CountSink, LoweringDiagnostics
from .memory import MemoryStoreQueryIndex
from .model import OwnerProjection, QueryIndexError, QueryIndexGenerationChanged, QueryIndexStatus, QueryIndexUnavailable, QueryStats, QueryVerifyBudgetExceeded, QueryWouldScanError, RefreshPolicy, SourceQueryPlan, ValidationIssue, ValidationReport
from .selector_graph import compile_selector_graph
from .utils import chunked


CACHE_SOURCE_KEY = "repo-cache"
_QUERY_BACKED_RESULT_THRESHOLD = 10_000
_QUERY_BACKED_PAGE_SIZE = 512


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

        if exact_root is not None:
            return self._execute_exact_stored_definition(query, exact_root, stats, stop_after=stop_after)

        for binding in self._executable_bindings("stored"):
            try:
                index = self._source_index_for_binding(binding)
                self._refresh_definition_source(index, query, exact_root, stats)
                before_verified = stats.verified_count
                source_matches = 0
                if _source_supports_lowering(index):
                    matches, generation, candidate_count, source_matches = self._execute_lowered_definition_source(
                        index,
                        query,
                        selector_graph,
                        stats,
                        stop_after=stop_after,
                        existing=merged,
                    )
                    generations[binding.source_key] = generation
                elif stop_after is None:
                    with index.read_view(include_cached=False) as snapshot:
                        candidate_ids = self._definition_candidate_ids(snapshot, query, selector_graph, exact_root, stats)
                        generation = snapshot.generation
                        generations[binding.source_key] = generation
                        cdefs_by_id = snapshot.cdefs_by_id(candidate_ids)
                        candidate_count = len(candidate_ids)
                    matches = query._verify_cdefs(tuple(cdefs_by_id.values()), stats=stats)
                    source_matches = len(matches)
                else:
                    matches = ()
                    for _ in range(3):
                        source_merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
                        source_replicas: dict[ConcreteDefinition, list[Any]] = {}
                        source_matches = 0
                        generation_changed = False
                        with index.read_view(include_cached=False) as snapshot:
                            candidate_ids = self._definition_candidate_ids(snapshot, query, selector_graph, exact_root, stats)
                            generation = snapshot.generation
                            generations[binding.source_key] = generation
                        for batch in chunked(sorted(candidate_ids), 256):
                            with index.read_view(include_cached=False) as snapshot:
                                if snapshot.generation != generation:
                                    generation_changed = True
                                    break
                                batch_cdefs = tuple(snapshot.cdefs_by_id(batch).values())
                            batch_matches = query._verify_cdefs(batch_cdefs, stats=stats)
                            source_matches += len(batch_matches)
                            for cdef in batch_matches:
                                source_canonical = _canonical_cdef_key(source_merged, cdef)
                                source_replicas.setdefault(source_canonical, []).append(binding.store)
                                if _combined_cdef_count(merged, source_merged) >= stop_after:
                                    break
                            if _combined_cdef_count(merged, source_merged) >= stop_after:
                                break
                        if generation_changed:
                            continue
                        for cdef in source_merged.values():
                            canonical = _canonical_cdef_key(merged, cdef)
                            replicas.setdefault(canonical, []).extend(source_replicas.get(cdef, ()))
                        break
                    else:
                        raise QueryIndexError("Catalog generation changed repeatedly during terminal stored query.")
                    candidate_count = len(candidate_ids)
            except Exception as exc:
                raise _source_query_error(binding, exc) from exc
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(index),
                generation=generation,
                candidate_count=candidate_count,
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

    def _execute_exact_stored_definition(
            self,
            query,
            exact_root: ConcreteDefinition,
            stats: QueryStats,
            *,
            stop_after: int | None):
        source_plans: list[SourceQueryPlan] = []
        generations: dict[str, int] = {}
        replicas = []
        for binding in self._executable_bindings("stored"):
            index = self._source_index_for_binding(binding)
            self._refresh_definition_source(index, query, exact_root, stats)
            with index.read_view(include_cached=False) as snapshot:
                exact_relation = getattr(snapshot, "relation_exact_stored", None)
                exact_count = getattr(snapshot, "relation_exact_safe_count", None)
                if exact_relation is not None and exact_count is not None:
                    relation = exact_relation(exact_root)
                    source_match = exact_count(relation) or 0
                else:
                    source_match = 1 if snapshot.filter_stored_ids(snapshot.exact_ids(exact_root)) else 0
                generation = snapshot.generation
            generations[binding.source_key] = generation
            if source_match:
                replicas.append(binding.store)
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(index),
                generation=generation,
                candidate_count=source_match,
                verified_count=0,
                result_count=source_match,
                refresh_action=stats.refresh_action,
            ))
            if stop_after is not None and replicas:
                break
        out = (exact_root,) if replicas else ()
        stats.result_count = len(out)
        stats.universe_size = None
        stats.generation_vector = generations
        stats.source_plans = tuple(source_plans)
        stats.lowering_strategy = "exact-safe-definition"
        stats.lowering_diagnostics = {
            "strategy": "exact-safe-definition",
            "exact_safe": True,
            "terminal_stop_reason": "exact-safe-match" if stop_after is not None and replicas else None,
        }
        return out, stats, {exact_root: tuple(_dedupe_stores(replicas))} if replicas else {}

    def query_backed_definition_result_set(self, query):
        if query.domain != "stored" or query.universe is not None:
            return None
        for binding in self._executable_bindings("stored"):
            if not _source_supports_lowering(self._source_index_for_binding(binding)):
                return None
        stats = self.explain_definition_domain(query)
        estimated = sum(plan.candidate_count for plan in stats.source_plans)
        if estimated <= _QUERY_BACKED_RESULT_THRESHOLD:
            return None

        from .result import QueryBackedDefinitionResultSet

        generation_vector = dict(stats.generation_vector or {})
        explanation = stats.explanation(domain=query.domain or "stored", refresh=query.refresh_policy)

        def page_factory():
            return self._iter_query_backed_lowered_definition_results(
                query,
                generation_vector=generation_vector,
                page_size=_QUERY_BACKED_PAGE_SIZE,
            )

        return QueryBackedDefinitionResultSet(
            self.repo,
            page_factory,
            materializable=True,
            domain="stored",
            explanation=explanation,
        )

    def _iter_query_backed_lowered_definition_results(
            self,
            query,
            *,
            generation_vector: Mapping[str, int],
            page_size: int = _QUERY_BACKED_PAGE_SIZE):
        selector_graph = compile_selector_graph(query.selector, class_match=query.class_match_policy)
        for binding in self._executable_bindings("stored"):
            index = self._source_index_for_binding(binding)
            expected_generation = generation_vector.get(binding.source_key)
            after = None
            diagnostics = LoweringDiagnostics()
            with index.read_view(include_cached=False) as snapshot:
                if expected_generation is not None and snapshot.generation != expected_generation:
                    raise QueryIndexGenerationChanged("Query-backed ResultSet source generation changed.")
                self._lower_stored_relation(
                    snapshot,
                    selector_graph,
                    terminal="page",
                    scan_policy=query.lowering_scan_policy,
                    diagnostics=diagnostics,
                )
                generation = snapshot.generation
            while True:
                with index.read_view(include_cached=False) as snapshot:
                    if snapshot.generation != generation:
                        raise QueryIndexGenerationChanged("Query-backed ResultSet source generation changed.")
                    relation, _ = self._lower_stored_relation(
                        snapshot,
                        selector_graph,
                        terminal="page",
                        scan_policy=query.lowering_scan_policy,
                        diagnostics=LoweringDiagnostics(),
                    )
                    batches = snapshot.iter_relation_cdef_batches(relation, after=after, batch_size=page_size)
                    batch = next(batches, None)
                    relation_diagnostics = getattr(snapshot, "relation_diagnostics", None)
                    if relation_diagnostics is not None:
                        diagnostics.copy_from(relation_diagnostics(relation))
                if batch is None:
                    break
                page_stats = QueryStats()
                matches = query._verify_cdefs(batch.cdefs, stats=page_stats)
                for cdef in matches:
                    yield cdef, (binding.store,)
                after = batch.next_cursor

    def count_definition_domain(self, query, *, stop_after: int | None = None) -> tuple[int, QueryStats]:
        last_generation_error = None
        for _ in range(3):
            try:
                return self._count_definition_domain_once(query, stop_after=stop_after)
            except QueryIndexGenerationChanged as exc:
                last_generation_error = exc
        raise QueryIndexError("Catalog generation changed repeatedly during terminal count query.") from last_generation_error

    def _count_definition_domain_once(self, query, *, stop_after: int | None = None) -> tuple[int, QueryStats]:
        stats = QueryStats(refresh_action="federated")
        selector_graph = compile_selector_graph(query.selector, class_match=query.class_match_policy)
        exact_root = query.selector if isinstance(query.selector, ConcreteDefinition) else None
        if exact_root is not None and query.domain == "stored":
            return self._count_exact_stored_definition(query, exact_root, stats, stop_after=stop_after)
        merged = _CDefDedupeCounter()
        source_plans: list[SourceQueryPlan] = []
        generations: dict[str, int] = {}

        for binding in self._executable_bindings("stored"):
            try:
                index = self._source_index_for_binding(binding)
                self._refresh_definition_source(index, query, exact_root, stats)
                source_matches = 0
                before_verified = stats.verified_count
                if _source_supports_lowering(index):
                    merged.register_source(
                        binding.source_key,
                        self._count_witness_loader(index, binding.source_key),
                    )
                    source_counter, generation, candidate_count, source_matches = self._count_lowered_definition_source(
                        index,
                        query,
                        selector_graph,
                        stats,
                        stop_after=stop_after,
                        existing=merged,
                    )
                    generations[binding.source_key] = generation
                    merged.merge_from(source_counter)
                else:
                    for _ in range(3):
                        source_counter = _CDefDedupeCounter()
                        loader = self._count_witness_loader(index, binding.source_key)
                        source_counter.register_source(binding.source_key, loader)
                        merged.register_source(binding.source_key, loader)
                        new_global_matches = 0
                        source_matches = 0
                        generation_changed = False
                        with index.read_view(include_cached=False) as snapshot:
                            candidate_ids = self._definition_candidate_ids(snapshot, query, selector_graph, exact_root, stats)
                            generation = snapshot.generation
                            generations[binding.source_key] = generation
                            candidate_count = len(candidate_ids)
                        for batch in chunked(sorted(candidate_ids), 256):
                            with index.read_view(include_cached=False) as snapshot:
                                if snapshot.generation != generation:
                                    generation_changed = True
                                    break
                                cdefs_by_id = snapshot.cdefs_by_id(batch)
                            for did in batch:
                                cdef = cdefs_by_id.get(did)
                                if cdef is None:
                                    continue
                                matches = query._verify_cdefs((cdef,), stats=stats)
                                if not matches:
                                    continue
                                match = matches[0]
                                if not source_counter.accept(
                                        match,
                                        source_key=binding.source_key,
                                        generation=generation,
                                        definition_id=did):
                                    continue
                                source_matches += 1
                                if not merged.seen(match):
                                    new_global_matches += 1
                                if stop_after is not None and merged.count + new_global_matches >= stop_after:
                                    break
                            if stop_after is not None and merged.count + new_global_matches >= stop_after:
                                break
                        if generation_changed:
                            continue
                        merged.merge_from(source_counter)
                        break
                    else:
                        raise QueryIndexError("Catalog generation changed repeatedly during terminal count query.")
                if stop_after is not None and merged.count >= stop_after:
                    source_plans.append(SourceQueryPlan(
                        source_key=binding.source_key,
                        backend=_source_backend(index),
                        generation=generation,
                        candidate_count=candidate_count,
                        verified_count=stats.verified_count - before_verified,
                        result_count=source_matches,
                        refresh_action=stats.refresh_action,
                    ))
                    stats.result_count = merged.count
                    stats.universe_size = None
                    stats.generation_vector = generations
                    stats.source_plans = tuple(source_plans)
                    return merged.count, stats
            except QueryIndexGenerationChanged:
                raise
            except Exception as exc:
                raise _source_query_error(binding, exc) from exc
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(index),
                generation=generation,
                candidate_count=candidate_count,
                verified_count=stats.verified_count - before_verified,
                result_count=source_matches,
                refresh_action=stats.refresh_action,
            ))

        stats.result_count = merged.count
        stats.universe_size = None
        stats.generation_vector = generations
        stats.source_plans = tuple(source_plans)
        return merged.count, stats

    def _count_exact_stored_definition(
            self,
            query,
            exact_root: ConcreteDefinition,
            stats: QueryStats,
            *,
            stop_after: int | None) -> tuple[int, QueryStats]:
        source_plans: list[SourceQueryPlan] = []
        generations: dict[str, int] = {}
        found = False
        for binding in self._executable_bindings("stored"):
            index = self._source_index_for_binding(binding)
            self._refresh_definition_source(index, query, exact_root, stats)
            with index.read_view(include_cached=False) as snapshot:
                exact_relation = getattr(snapshot, "relation_exact_stored", None)
                exact_count = getattr(snapshot, "relation_exact_safe_count", None)
                if exact_relation is not None and exact_count is not None:
                    relation = exact_relation(exact_root)
                    source_match = exact_count(relation) or 0
                else:
                    source_match = 1 if snapshot.filter_stored_ids(snapshot.exact_ids(exact_root)) else 0
                generation = snapshot.generation
            generations[binding.source_key] = generation
            found = found or bool(source_match)
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(index),
                generation=generation,
                candidate_count=source_match,
                verified_count=0,
                result_count=source_match,
                refresh_action=stats.refresh_action,
            ))
            if stop_after is not None and found:
                break
        stats.result_count = 1 if found else 0
        stats.universe_size = None
        stats.generation_vector = generations
        stats.source_plans = tuple(source_plans)
        stats.lowering_strategy = "exact-safe-count"
        stats.lowering_diagnostics = {
            "strategy": "exact-safe-count",
            "exact_safe": True,
            "terminal_stop_reason": "first-match" if stop_after is not None and found else None,
        }
        return stats.result_count, stats

    def explain_definition_domain(self, query, *, sql: bool = False) -> QueryStats:
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
                    if getattr(snapshot, "supports_lowering", False):
                        diagnostics = LoweringDiagnostics()
                        plan = snapshot.lower_selector_graph(
                            selector_graph,
                            StoredDomain(snapshot),
                            terminal="explain",
                            scan_policy=query.lowering_scan_policy,
                            diagnostics=diagnostics,
                        )
                        explain = getattr(snapshot, "explain_lowered_plan", None)
                        if sql and explain is not None:
                            explain(plan)
                        candidate_ids = ()
                        stats.lowering_strategy = diagnostics.strategy
                        stats.scan_required = stats.scan_required or diagnostics.scan_required
                        stats.scan_reason = stats.scan_reason or diagnostics.scan_reason
                        stats.lowering_diagnostics = diagnostics.as_dict()
                        candidate_delta = plan.estimated_size or 0
                    else:
                        candidate_ids = self._definition_candidate_ids(snapshot, query, selector_graph, exact_root, stats)
                        candidate_delta = stats.candidate_count - before_candidates if stats.candidate_count >= before_candidates else len(candidate_ids)
                    generation = snapshot.generation
                    generations[binding.source_key] = generation
            except Exception as exc:
                raise _source_query_error(binding, exc) from exc
            if stats.candidate_count == before_candidates:
                stats.candidate_count += candidate_delta
            source_plans.append(SourceQueryPlan(
                source_key=binding.source_key,
                backend=_source_backend(index),
                generation=generation,
                candidate_count=candidate_delta,
                verified_count=0,
                result_count=None,
                refresh_action=stats.refresh_action,
            ))
        stats.universe_size = None
        stats.result_count = None
        stats.generation_vector = generations
        stats.source_plans = tuple(source_plans)
        return stats

    def _execute_lowered_definition_source(
            self,
            index,
            query,
            selector_graph,
            stats: QueryStats,
            *,
            stop_after: int | None,
            existing: Mapping[ConcreteDefinition, ConcreteDefinition]):
        for _ in range(3):
            sink = CollectSink(stop_after=stop_after)
            source_merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
            generation_changed = False
            after = None
            diagnostics = LoweringDiagnostics()
            with index.read_view(include_cached=False) as snapshot:
                self._lower_stored_relation(
                    snapshot,
                    selector_graph,
                    terminal="collect",
                    scan_policy=query.lowering_scan_policy,
                    diagnostics=diagnostics,
                )
                generation = snapshot.generation
            while True:
                with index.read_view(include_cached=False) as snapshot:
                    if snapshot.generation != generation:
                        generation_changed = True
                        break
                    page_diagnostics = LoweringDiagnostics()
                    relation, _ = self._lower_stored_relation(
                        snapshot,
                        selector_graph,
                        terminal="collect",
                        scan_policy=query.lowering_scan_policy,
                        diagnostics=page_diagnostics,
                    )
                    batches = snapshot.iter_relation_cdef_batches(
                        relation,
                        after=after,
                        batch_size=_terminal_batch_size(stop_after),
                    )
                    batch = next(batches, None)
                    relation_diagnostics = getattr(snapshot, "relation_diagnostics", None)
                    if relation_diagnostics is not None:
                        page_diagnostics.copy_from(relation_diagnostics(relation))
                    _merge_page_diagnostics(diagnostics, page_diagnostics)
                if batch is None:
                    break
                for cdef in batch.cdefs:
                    before_verified = stats.verified_count
                    matches = query._verify_cdefs((cdef,), stats=stats)
                    diagnostics.python_verifications += stats.verified_count - before_verified
                    if matches:
                        _canonical_cdef_key(source_merged, matches[0])
                        sink.accept(matches[0])
                    if sink.done or (stop_after is not None and _combined_cdef_count(existing, source_merged) >= stop_after):
                        break
                if sink.done or (stop_after is not None and _combined_cdef_count(existing, source_merged) >= stop_after):
                    diagnostics.terminal_stop_reason = sink.stop_reason or "collect-limit"
                    break
                after = batch.next_cursor
            if generation_changed:
                continue
            stats.lowering_strategy = diagnostics.strategy
            stats.scan_required = stats.scan_required or diagnostics.scan_required
            stats.scan_reason = stats.scan_reason or diagnostics.scan_reason
            stats.candidate_rows_read += diagnostics.candidate_rows_read
            stats.cdef_blobs_decoded += diagnostics.cdef_blobs_decoded
            stats.pages_fetched += diagnostics.pages_fetched
            stats.terminal_stop_reason = stats.terminal_stop_reason or diagnostics.terminal_stop_reason
            stats.lowering_diagnostics = diagnostics.as_dict()
            matches = tuple(source_merged.values())
            return matches, generation, diagnostics.candidate_rows_read, len(matches)
        raise QueryIndexError("Catalog generation changed repeatedly during lowered stored query.")

    def _count_lowered_definition_source(
            self,
            index,
            query,
            selector_graph,
            stats: QueryStats,
            *,
            stop_after: int | None,
            existing: "_CDefDedupeCounter"):
        for _ in range(3):
            sink = CountSink(stop_after=stop_after)
            source_counter = _CDefDedupeCounter()
            source_counter.register_source(index.source_key, self._count_witness_loader(index, index.source_key))
            new_global_matches = 0
            generation_changed = False
            after = None
            diagnostics = LoweringDiagnostics()
            with index.read_view(include_cached=False) as snapshot:
                self._lower_stored_relation(
                    snapshot,
                    selector_graph,
                    terminal="count",
                    scan_policy=query.lowering_scan_policy,
                    diagnostics=diagnostics,
                )
                generation = snapshot.generation
            while True:
                with index.read_view(include_cached=False) as snapshot:
                    if snapshot.generation != generation:
                        generation_changed = True
                        break
                    page_diagnostics = LoweringDiagnostics()
                    relation, _ = self._lower_stored_relation(
                        snapshot,
                        selector_graph,
                        terminal="count",
                        scan_policy=query.lowering_scan_policy,
                        diagnostics=page_diagnostics,
                    )
                    batches = snapshot.iter_relation_cdef_batches(
                        relation,
                        after=after,
                        batch_size=_terminal_batch_size(stop_after),
                    )
                    batch = next(batches, None)
                    relation_diagnostics = getattr(snapshot, "relation_diagnostics", None)
                    if relation_diagnostics is not None:
                        page_diagnostics.copy_from(relation_diagnostics(relation))
                    _merge_page_diagnostics(diagnostics, page_diagnostics)
                if batch is None:
                    break
                for did, cdef in zip(batch.ids, batch.cdefs):
                    before_verified = stats.verified_count
                    matches = query._verify_cdefs((cdef,), stats=stats)
                    diagnostics.python_verifications += stats.verified_count - before_verified
                    if matches:
                        match = matches[0]
                        if source_counter.accept(
                                match,
                                source_key=index.source_key,
                                generation=generation,
                                definition_id=did):
                            sink.accept(match)
                            if not existing.seen(match):
                                new_global_matches += 1
                    if stop_after is not None and existing.count + new_global_matches >= stop_after:
                        break
                if stop_after is not None and existing.count + new_global_matches >= stop_after:
                    diagnostics.terminal_stop_reason = sink.stop_reason or "count-limit"
                    break
                after = batch.next_cursor
            if generation_changed:
                continue
            stats.lowering_strategy = diagnostics.strategy
            stats.scan_required = stats.scan_required or diagnostics.scan_required
            stats.scan_reason = stats.scan_reason or diagnostics.scan_reason
            stats.candidate_rows_read += diagnostics.candidate_rows_read
            stats.cdef_blobs_decoded += diagnostics.cdef_blobs_decoded
            stats.pages_fetched += diagnostics.pages_fetched
            stats.count_witness_reloads += source_counter.witness_reload_count
            stats.count_collision_buckets += source_counter.collision_bucket_count
            diagnostics.count_witness_reloads = source_counter.witness_reload_count
            diagnostics.count_collision_buckets = source_counter.collision_bucket_count
            stats.terminal_stop_reason = stats.terminal_stop_reason or diagnostics.terminal_stop_reason
            stats.lowering_diagnostics = diagnostics.as_dict()
            return source_counter, generation, diagnostics.candidate_rows_read, source_counter.count
        raise QueryIndexError("Catalog generation changed repeatedly during lowered count query.")

    def _count_witness_loader(self, index, source_key: str) -> Callable[[int, int], ConcreteDefinition]:
        def load(generation: int, definition_id: int) -> ConcreteDefinition:
            with index.read_view(include_cached=False) as snapshot:
                if snapshot.generation != generation:
                    raise QueryIndexGenerationChanged(
                        f"Count dedupe witness source '{source_key}' generation changed."
                    )
                cdef = snapshot.cdefs_by_id((definition_id,)).get(definition_id)
                if cdef is None:
                    raise QueryIndexError(
                        f"Count dedupe witness {definition_id} is missing from source '{source_key}'."
                    )
                return cdef

        return load

    def execute_nested_definitions(self, query, *, stop_after: int | None = None):
        stats = QueryStats(refresh_action="federated")
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        source_plans: list[SourceQueryPlan] = []
        generations: dict[str, int] = {}
        for binding in self._executable_bindings("nested"):
            before_candidates = stats.candidate_count
            before_verified = stats.verified_count
            index = self._source_index_for_binding(binding)
            if _source_supports_lowering(index):
                matches, _, generation, candidate_count = self._capture_lowered_nested_matches(
                    query,
                    binding,
                    stats,
                    stop_after=stop_after,
                )
                stats.candidate_count += candidate_count
            else:
                cdefs_by_id, generation = self._capture_nested_candidates(query, binding, stats)
                matches, _ = query._verify_cdefs_by_id(cdefs_by_id, stats=stats)
            generations[binding.source_key] = generation
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
        owner_relation_ops = 0
        for binding in self._executable_bindings("nested"):
            before_candidates = stats.candidate_count
            before_verified = stats.verified_count
            for _ in range(3):
                index = self._source_index_for_binding(binding)
                using_lowering = _source_supports_lowering(index)
                if using_lowering:
                    _, match_ids, generation, candidate_count = self._capture_lowered_nested_matches(query, binding, stats)
                    stats.candidate_count += candidate_count
                else:
                    cdefs_by_id, generation = self._capture_nested_candidates(query, binding, stats)
                    _, match_ids = query._verify_cdefs_by_id(cdefs_by_id, stats=stats)
                with index.read_view(include_cached=False) as snapshot:
                    if snapshot.generation != generation:
                        continue
                    if using_lowering and hasattr(snapshot, "relation_from_ids") and hasattr(snapshot, "relation_project_owners"):
                        target_relation = snapshot.relation_from_ids(
                            match_ids,
                            domain="nested",
                            debug_label="verified-nested-targets",
                        )
                        owner_relation = snapshot.relation_project_owners(target_relation)
                        owner_batches = tuple(snapshot.iter_relation_cdef_batches(owner_relation, batch_size=256))
                        owner_ids = frozenset(did for batch in owner_batches for did in batch.ids)
                        owner_cdefs = tuple(cdef for batch in owner_batches for cdef in batch.cdefs)
                        projection = OwnerProjection(
                            owner_ids=owner_ids,
                            cdefs=owner_cdefs,
                            replicas={owner: (binding.store,) for owner in owner_cdefs},
                        )
                        owner_relation_ops += 1
                    else:
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
        stats.lowering_diagnostics = {
            **(stats.lowering_diagnostics or {}),
            "owners_found": len(out),
            "owner_projection_relation_ops": owner_relation_ops,
        }
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
                index = self._source_index_for_binding(binding)
                if _source_supports_lowering(index):
                    _, match_ids, generation, candidate_count = self._capture_lowered_nested_matches(query, binding, stats)
                    stats.candidate_count += candidate_count
                else:
                    cdefs_by_id, generation = self._capture_nested_candidates(query, binding, stats)
                    _, match_ids = query._verify_cdefs_by_id(cdefs_by_id, stats=stats)
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
        stats.lowering_diagnostics = {
            **(stats.lowering_diagnostics or {}),
            "occurrence_nested_targets": sum(len(traversal.targets) for traversal in traversals),
            "occurrence_nodes_captured": sum(len(traversal.cdefs) for traversal in traversals),
            "occurrence_incoming_edges_captured": sum(
                len(edges)
                for traversal in traversals
                for edges in traversal.incoming.values()
            ),
            "occurrence_owners_found": sum(len(traversal.stored_ids) for traversal in traversals),
            "occurrence_path_limit": query.occurrence_limit,
            "occurrence_paths_emitted": None,
            "occurrence_path_limit_reached": None,
        }

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

    def _capture_lowered_nested_matches(self, query, binding: StoreIndexBinding, stats: QueryStats, *, stop_after: int | None = None):
        index = self._source_index_for_binding(binding)
        index.refresh(query.refresh_policy, stats=stats)
        selector_graph = compile_selector_graph(query.selector, class_match=query.class_match_policy)
        for _ in range(3):
            merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
            match_ids: set[Any] = set()
            generation_changed = False
            after = None
            diagnostics = LoweringDiagnostics()
            with index.read_view(include_cached=False) as snapshot:
                self._lower_relation_for_domain(
                    snapshot,
                    selector_graph,
                    NestedDomain(snapshot),
                    terminal="collect",
                    scan_policy=query.lowering_scan_policy,
                    diagnostics=diagnostics,
                )
                generation = snapshot.generation
            while True:
                with index.read_view(include_cached=False) as snapshot:
                    if snapshot.generation != generation:
                        generation_changed = True
                        break
                    page_diagnostics = LoweringDiagnostics()
                    relation, _ = self._lower_relation_for_domain(
                        snapshot,
                        selector_graph,
                        NestedDomain(snapshot),
                        terminal="collect",
                        scan_policy=query.lowering_scan_policy,
                        diagnostics=page_diagnostics,
                    )
                    batches = snapshot.iter_relation_cdef_batches(relation, after=after, batch_size=256)
                    batch = next(batches, None)
                    relation_diagnostics = getattr(snapshot, "relation_diagnostics", None)
                    if relation_diagnostics is not None:
                        page_diagnostics.copy_from(relation_diagnostics(relation))
                    _merge_page_diagnostics(diagnostics, page_diagnostics)
                if batch is None:
                    break
                for did, cdef in zip(batch.ids, batch.cdefs):
                    before_verified = stats.verified_count
                    matches = query._verify_cdefs((cdef,), stats=stats)
                    diagnostics.python_verifications += stats.verified_count - before_verified
                    if not matches:
                        continue
                    _canonical_cdef_key(merged, matches[0])
                    match_ids.add(did)
                    if stop_after is not None and len(merged) >= stop_after:
                        break
                if stop_after is not None and len(merged) >= stop_after:
                    diagnostics.terminal_stop_reason = "collect-limit"
                    break
                after = batch.next_cursor
            if generation_changed:
                continue
            stats.lowering_strategy = diagnostics.strategy
            stats.scan_required = stats.scan_required or diagnostics.scan_required
            stats.scan_reason = stats.scan_reason or diagnostics.scan_reason
            stats.candidate_rows_read += diagnostics.candidate_rows_read
            stats.cdef_blobs_decoded += diagnostics.cdef_blobs_decoded
            stats.pages_fetched += diagnostics.pages_fetched
            stats.terminal_stop_reason = stats.terminal_stop_reason or diagnostics.terminal_stop_reason
            stats.lowering_diagnostics = diagnostics.as_dict()
            return tuple(merged.values()), match_ids, generation, diagnostics.candidate_rows_read
        raise QueryIndexError("Catalog generation changed repeatedly during lowered nested query.")

    def _lower_stored_relation(
            self,
            snapshot,
            selector_graph,
            *,
            terminal,
            scan_policy,
            diagnostics: LoweringDiagnostics,
            use_count: int = 1,
            recursive: bool = False):
        return self._lower_relation_for_domain(
            snapshot,
            selector_graph,
            StoredDomain(snapshot),
            terminal=terminal,
            scan_policy=scan_policy,
            diagnostics=diagnostics,
            use_count=use_count,
            recursive=recursive,
        )

    def _lower_relation_for_domain(
            self,
            snapshot,
            selector_graph,
            domain,
            *,
            terminal,
            scan_policy,
            diagnostics: LoweringDiagnostics,
            use_count: int = 1,
            recursive: bool = False):
        plan = snapshot.lower_selector_graph(
            selector_graph,
            KnownDomain(snapshot),
            terminal=terminal,
            scan_policy=scan_policy,
            diagnostics=diagnostics,
        )
        relation = snapshot.relation_filter_domain(plan.relation(), domain)
        optimize = getattr(snapshot, "relation_optimize", None)
        if optimize is not None:
            relation = optimize(relation, terminal=terminal, use_count=use_count, recursive=recursive)
        relation_diagnostics = getattr(snapshot, "relation_diagnostics", None)
        if relation_diagnostics is not None:
            diagnostics.copy_from(relation_diagnostics(relation))
        return relation, plan

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

    def register_saved_graph(
            self, graph, roots_by_store: Mapping[Any, Sequence[Any]],
            state_refs_by_store: Mapping[Any, Sequence[Any]] | None = None) -> None:
        """Register one completed authoritative save with derived Store indexes.

        Args:
            graph: Complete query graph published by the save.
            roots_by_store: Stored roots grouped by owning Store.
            state_refs_by_store: Newly published StateRefs grouped by Store for
                incremental advisory-reference registration.

        Raises:
            QueryIndexError: If a configured derived index rejects registration.

        Side Effects:
            Updates available derived indexes or leaves their durable dirty
            markers in place when registration fails.
        """
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
            register_saved = getattr(index, "register_saved_graph", None)
            register = register_saved or getattr(index, "register_stored_roots", None)
            if register is None:
                continue
            try:
                if register_saved is not None:
                    references = () if state_refs_by_store is None else tuple(
                        state_refs_by_store.get(store, ())
                    )
                    register(graph, roots, references)
                else:
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


@dataclass(frozen=True, slots=True)
class _CDefWitnessRef:
    source_key: str
    generation: int
    definition_id: int


class _CDefDedupeCounter:
    """Collision-safe count state with lazy CDef witness recovery.

    First observations retain only a stable hash and a backend witness ref. Full
    CDefs are loaded and retained only when a stable hash repeats and exact
    equality must distinguish duplicates from real hash collisions.
    """

    def __init__(self) -> None:
        self._witnesses: dict[str, _CDefWitnessRef] = {}
        self._buckets: dict[str, list[ConcreteDefinition]] = {}
        self._bucket_refs: dict[str, list[_CDefWitnessRef]] = {}
        self._loaders: dict[str, Callable[[int, int], ConcreteDefinition]] = {}
        self.count = 0
        self.witness_reload_count = 0
        self.collision_bucket_count = 0

    def register_source(self, source_key: str, loader: Callable[[int, int], ConcreteDefinition]) -> None:
        self._loaders[source_key] = loader

    def seen(self, cdef: ConcreteDefinition) -> bool:
        stable_hash = cdef.stable_hash()
        if stable_hash not in self._witnesses and stable_hash not in self._buckets:
            return False
        bucket = self._ensure_bucket(stable_hash)
        return any(existing == cdef for existing in bucket)

    def accept(
            self,
            cdef: ConcreteDefinition,
            *,
            source_key: str,
            generation: int,
            definition_id: int) -> bool:
        ref = _CDefWitnessRef(source_key, generation, definition_id)
        stable_hash = cdef.stable_hash()
        if stable_hash not in self._witnesses and stable_hash not in self._buckets:
            self._witnesses[stable_hash] = ref
            self.count += 1
            return True
        return self._accept_repeated_hash(cdef, ref)

    def merge_from(self, other: "_CDefDedupeCounter") -> None:
        self._loaders.update(other._loaders)
        for stable_hash, ref in other._witnesses.items():
            if stable_hash in other._buckets:
                continue
            if stable_hash not in self._witnesses and stable_hash not in self._buckets:
                self._witnesses[stable_hash] = ref
                self.count += 1
            else:
                self._accept_repeated_hash(other._load_ref(ref), ref)
        for stable_hash, bucket in other._buckets.items():
            refs = other._bucket_refs[stable_hash]
            for cdef, ref in zip(bucket, refs):
                if stable_hash not in self._witnesses and stable_hash not in self._buckets:
                    self._witnesses[stable_hash] = ref
                    self.count += 1
                else:
                    self._accept_repeated_hash(cdef, ref)

    def _accept_repeated_hash(self, cdef: ConcreteDefinition, ref: _CDefWitnessRef) -> bool:
        stable_hash = cdef.stable_hash()
        bucket = self._ensure_bucket(stable_hash)
        if any(existing == cdef for existing in bucket):
            return False
        bucket.append(cdef)
        self._bucket_refs[stable_hash].append(ref)
        self.count += 1
        return True

    def _ensure_bucket(self, stable_hash: str) -> list[ConcreteDefinition]:
        bucket = self._buckets.get(stable_hash)
        if bucket is not None:
            return bucket
        ref = self._witnesses[stable_hash]
        bucket = [self._load_ref(ref)]
        self._buckets[stable_hash] = bucket
        self._bucket_refs[stable_hash] = [ref]
        self.collision_bucket_count += 1
        return bucket

    def _load_ref(self, ref: _CDefWitnessRef) -> ConcreteDefinition:
        loader = self._loaders.get(ref.source_key)
        if loader is None:
            raise QueryIndexError(f"No count dedupe witness loader for source '{ref.source_key}'.")
        self.witness_reload_count += 1
        return loader(ref.generation, ref.definition_id)


def _canonical_cdef_key(merged: dict[ConcreteDefinition, ConcreteDefinition], cdef: ConcreteDefinition) -> ConcreteDefinition:
    existing = merged.get(cdef)
    if existing is not None:
        return existing
    merged[cdef] = cdef
    return cdef


def _combined_cdef_count(
    merged: Mapping[ConcreteDefinition, ConcreteDefinition],
    additions: Mapping[ConcreteDefinition, ConcreteDefinition],
) -> int:
    count = len(merged)
    for cdef in additions:
        if cdef not in merged:
            count += 1
    return count


def _merge_page_diagnostics(total: LoweringDiagnostics, page: LoweringDiagnostics) -> None:
    total.sql_statements_executed += page.sql_statements_executed
    total.candidate_rows_read += page.candidate_rows_read
    total.cdef_blobs_decoded += page.cdef_blobs_decoded
    total.pages_fetched += page.pages_fetched
    total.relations_created += page.relations_created
    total.relations_dropped += page.relations_dropped
    total.temp_rows_inserted += page.temp_rows_inserted
    total.inline_relations = tuple(dict.fromkeys((*total.inline_relations, *page.inline_relations)))
    total.materialized_relations = tuple(dict.fromkeys((*total.materialized_relations, *page.materialized_relations)))
    if total.logical_plan is None:
        total.logical_plan = page.logical_plan
    if total.physical_plan is None:
        total.physical_plan = page.physical_plan


def _terminal_batch_size(stop_after: int | None) -> int:
    if stop_after is None:
        return 256
    return max(1, min(256, stop_after))


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


def _source_supports_lowering(index) -> bool:
    if index is None:
        return False
    try:
        with index.read_view(include_cached=False) as snapshot:
            return bool(getattr(snapshot, "supports_lowering", False))
    except Exception:
        return False


def _source_query_error(binding: StoreIndexBinding, exc: BaseException) -> QueryIndexError:
    if isinstance(exc, (QueryWouldScanError, QueryVerifyBudgetExceeded)):
        return exc
    if isinstance(exc, QueryIndexError):
        return QueryIndexError(f"Query source {binding.source_key!r} failed: {exc}")
    return QueryIndexError(f"Query source {binding.source_key!r} failed with {type(exc).__name__}: {exc}")
