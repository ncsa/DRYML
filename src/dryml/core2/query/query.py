from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from ..canonical import matching_container_family
from ..arg_roles import apply_definition_arg_roles
from ..cdef_identity import V2_IDENTITY_VERSION
from ..definition import ConcreteDefinition, Definition, categorical_definition, selector_match
from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from ..links import DefLink
from ..object import Object
from ..params import Par
from ..quoted import QuotedDef, SelectorSpec
from ..selector import Selector
from ..symbol import maybe_symbol_ref, resolve_symbol
from ..utils.types import is_nonclass_callable
from .graph_plan import graph_candidate_ids
from .lowering import ScanPolicy
from .domain import CachedDomain, KnownDomain, NestedDomain, StoredDomain
from .model import (
    ClassMatchPolicy,
    DefinitionId,
    DefinitionOccurrence,
    QueryDomain,
    QueryDomainError,
    QueryExplanation,
    QueryCardinalityError,
    QueryIndexError,
    QueryIndexUnavailable,
    QueryVerifyBudgetExceeded,
    QueryProjection,
    SourceQueryPlan,
    QueryStats,
    RefreshPolicy,
    ResultUniverse,
)
from .path import DefinitionPath, DefinitionPathLike, QueryPathError, get_subtree, normalize_path, replace_subtree
from .result import DefinitionResultSet, ObjectResultSet, OccurrenceResultSet
from .selector_graph import compile_selector_graph
from .utils import cdef_equal


@dataclass(frozen=True, slots=True)
class CapturedNestedCandidates:
    generation: int
    cdefs_by_id: dict[DefinitionId, ConcreteDefinition]
    stats: QueryStats


class _QueryGenerationChanged(Exception):
    pass


_MAX_NESTED_QUERY_RETRIES = 3


@dataclass(frozen=True, slots=True)
class DefinitionQuery:
    repo: Any
    original: Definition | ConcreteDefinition | None
    selector: Definition | ConcreteDefinition | None
    domain: QueryDomain | None = None
    projection: QueryProjection | None = None
    class_match_policy: ClassMatchPolicy = "selector"
    strict_policy: bool = False
    refresh_policy: RefreshPolicy = "auto"
    reuse_weak_policy: bool = True
    universe: ResultUniverse | None = None
    occurrence_limit: int | None = None
    scan_policy_mode: str = "allow"
    max_verify_limit: int | None = None

    @classmethod
    def from_source(
            cls,
            repo,
            source=None,
            *,
            domain: QueryDomain | None = None,
            universe: ResultUniverse | None = None) -> "DefinitionQuery":
        if isinstance(source, Selector):
            root = source.root
            return cls(
                repo=repo,
                original=root,
                selector=root,
                domain=domain,
                universe=universe,
                strict_policy=source.strict,
                class_match_policy=source.cls_policy,
            )
        original = _snapshot_source(source)
        return cls(repo=repo, original=original, selector=original, domain=domain, universe=universe)

    def categorical(
            self,
            *,
            path: DefinitionPathLike = "$",
            recursive: bool = False) -> "DefinitionQuery":
        if self.selector is None:
            raise QueryPathError("Cannot apply categorical() to an unconstrained query.")
        norm = normalize_path(path)
        subtree = get_subtree(self.selector, norm)
        projected = categorical_definition(subtree, recursive=recursive)
        return replace(self, selector=replace_subtree(self.selector, norm, projected))

    def restore(self, *, path: DefinitionPathLike = "$") -> "DefinitionQuery":
        if self.original is None or self.selector is None:
            raise QueryPathError("Cannot restore() on an unconstrained query.")
        norm = normalize_path(path)
        subtree = get_subtree(self.original, self._original_path(norm))
        replacement = deepcopy(subtree) if isinstance(subtree, Definition) else subtree
        return replace(self, selector=replace_subtree(self.selector, norm, replacement))

    def exact(
            self,
            definition: ConcreteDefinition | Object | None = None,
            *,
            path: DefinitionPathLike = "$") -> "DefinitionQuery":
        if self.selector is None:
            raise QueryPathError("Cannot apply exact() to an unconstrained query.")
        norm = normalize_path(path)
        if definition is None:
            if self.original is None:
                raise QueryPathError(f"Cannot infer exact subtree at {norm!s}; query has no original source.")
            definition = get_subtree(self.original, self._original_path(norm))
        if isinstance(definition, Object):
            definition = definition.definition
        if not isinstance(definition, ConcreteDefinition):
            raise TypeError(f"Exact constraint at {norm!s} requires a ConcreteDefinition, got {type(definition).__name__}.")
        return replace(self, selector=replace_subtree(self.selector, norm, definition))

    def _original_path(self, path: DefinitionPath) -> DefinitionPath:
        """Map selector spelling to an exact V2 source path when available."""

        if not isinstance(self.original, ConcreteDefinition) or self.original.identity_version != V2_IDENTITY_VERSION:
            return path
        from .selector_graph import _semantic_selector_path

        semantic = _semantic_selector_path(self.selector, path)
        return path if semantic is None else semantic

    def class_match(self, policy: ClassMatchPolicy) -> "DefinitionQuery":
        if policy not in {"selector", "exact"}:
            raise ValueError("class_match policy must be 'selector' or 'exact'.")
        return replace(self, class_match_policy=policy)

    def strict(self, enabled: bool = True) -> "DefinitionQuery":
        return replace(self, strict_policy=bool(enabled))

    def refresh(self, policy: RefreshPolicy = "auto") -> "DefinitionQuery":
        if policy not in {False, "auto", True}:
            raise ValueError("refresh policy must be False, 'auto', or True.")
        return replace(self, refresh_policy=policy)

    def reuse_weak(self, enabled: bool = True) -> "DefinitionQuery":
        return replace(self, reuse_weak_policy=bool(enabled))

    def stored(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        self._check_universe_domain_switch("stored")
        q = replace(self, domain="stored", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def cached(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        self._check_universe_domain_switch("cached")
        q = replace(self, domain="cached", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def known(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        self._check_universe_domain_switch("known")
        q = replace(self, domain="known", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def nested(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        self._check_universe_domain_switch("nested")
        q = replace(self, domain="nested", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def _check_universe_domain_switch(self, requested: str) -> None:
        if self.universe is None:
            return
        if requested != self.universe.domain:
            raise QueryDomainError(
                f"Cannot switch a fixed ResultSet universe from domain {self.universe.domain!r} to {requested!r}."
            )

    def definitions(self) -> "DefinitionQuery":
        if self.domain != "nested":
            return self
        return replace(self, projection="definitions")

    def owners(self) -> "DefinitionQuery":
        if self.domain != "nested":
            raise QueryDomainError("owners() is only valid for nested queries.")
        return replace(self, projection="owners")

    def max_occurrences(self, limit: int | None) -> "DefinitionQuery":
        if limit is not None and limit < 0:
            raise ValueError("max_occurrences limit must be non-negative or None.")
        # This bounds path enumeration. Capturing the candidate ancestor subgraph is terminal-specific.
        return replace(self, occurrence_limit=limit)

    def scan_policy(self, policy: str) -> "DefinitionQuery":
        if policy not in {"allow", "warn", "forbid"}:
            raise ValueError("scan policy must be 'allow', 'warn', or 'forbid'.")
        return replace(self, scan_policy_mode=policy)

    def require_indexed(self) -> "DefinitionQuery":
        return self.scan_policy("forbid")

    def max_verify(self, limit: int | None) -> "DefinitionQuery":
        if limit is not None and limit < 0:
            raise ValueError("max_verify limit must be non-negative or None.")
        return replace(self, max_verify_limit=limit)

    @property
    def lowering_scan_policy(self) -> ScanPolicy:
        return ScanPolicy(self.scan_policy_mode, self.max_verify_limit)

    def execute(self):
        self._require_domain()
        if self.domain == "nested":
            if self.universe is None and self.projection == "definitions":
                cdefs, stats = self._execute_nested_definitions()
                explanation = stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)
                return DefinitionResultSet(
                    self.repo,
                    cdefs,
                    materializable=False,
                    domain="nested-definitions",
                    explanation=explanation,
                    replicas={},
                )
            if self.universe is None and self.projection == "owners":
                cdefs, stats, replicas = self._execute_nested_owners()
                explanation = stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)
                return DefinitionResultSet(
                    self.repo,
                    cdefs,
                    materializable=True,
                    domain="owners",
                    explanation=explanation,
                    replicas=replicas,
                )
            occs, stats, owner_replicas = self._execute_nested_occurrences()
            explanation = stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)
            if callable(occs):
                raw = OccurrenceResultSet(
                    self.repo,
                    occurrence_factory=occs,
                    explanation=explanation,
                    owner_replicas=owner_replicas,
                )
            else:
                raw = OccurrenceResultSet(self.repo, occs, explanation=explanation, owner_replicas=owner_replicas)
            if self.projection == "definitions":
                return raw.definitions()
            if self.projection == "owners":
                return raw.owners()
            return raw

        if self.universe is None and self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            query_backed = getattr(self.repo._query_index, "query_backed_definition_result_set", None)
            if query_backed is not None:
                result_set = query_backed(self)
                if result_set is not None:
                    return result_set

        cdefs, stats, query_replicas = self._execute_definition_domain()
        explanation = stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)
        materializable = True
        domain = self.domain or "stored"
        if self.universe is not None:
            materializable = self.universe.materializable
            domain = self.universe.domain
        replicas = {} if query_replicas is None else query_replicas
        if self.universe is not None and self.universe.replicas is not None:
            replicas = {cdef: self.universe.replicas.get(cdef, ()) for cdef in cdefs}
        return DefinitionResultSet(
            self.repo,
            cdefs,
            materializable=materializable,
            domain=domain,
            explanation=explanation,
            replicas=replicas,
        )

    def defs(self):
        result = self.execute()
        if isinstance(result, DefinitionResultSet):
            return result
        raise QueryDomainError("Raw nested queries return occurrences; use .definitions().defs() or .owners().defs().")

    def objects(self, **load_options) -> ObjectResultSet:
        result = self.execute()
        if isinstance(result, OccurrenceResultSet):
            raise QueryDomainError("Raw nested occurrences cannot be materialized directly; use .owners().objects().")
        return result.objects(**load_options)

    def count(self) -> int:
        return self._execute_count()

    def exists(self) -> bool:
        if self.universe is None and self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            count, _ = self.repo._query_index.count_definition_domain(self, stop_after=1)
            return count > 0
        if self.universe is None and self.domain == "known" and self.repo._query_index.can_execute_query_domain("stored"):
            return bool(self._execute_federated_known_domain(stop_after=1)[0])
        if self.universe is None and self.domain == "nested":
            return bool(self._execute_terminal_items(stop_after=1))
        return self._execute_count() > 0

    def one(self):
        items = self._execute_terminal_items(stop_after=2)
        if len(items) != 1:
            label = "occurrence" if self.domain == "nested" and self.projection is None else "result"
            raise QueryCardinalityError(f"Expected exactly one {label}, found {len(items)}.")
        return items[0]

    def one_or_none(self):
        items = self._execute_terminal_items(stop_after=2)
        if len(items) > 1:
            label = "occurrence" if self.domain == "nested" and self.projection is None else "result"
            raise QueryCardinalityError(f"Expected zero or one {label}, found {len(items)}.")
        return items[0] if items else None

    def explain(self, *, analyze: bool = False, sql: bool = False) -> QueryExplanation:
        return self._execute_explanation(analyze=analyze, sql=sql)

    def _execute_count(self) -> int:
        self._require_domain()
        if self.universe is None and self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            count, _ = self.repo._query_index.count_definition_domain(self)
            return count
        if self.universe is None and self.domain == "known" and self.repo._query_index.can_execute_query_domain("stored"):
            cdefs, _, _ = self._execute_federated_known_domain()
            return len(cdefs)
        if self.domain == "nested":
            if self.universe is None and self.projection == "definitions":
                if self.repo._query_index.can_execute_query_domain("nested"):
                    cdefs, _ = self.repo._query_index.execute_nested_definitions(self)
                else:
                    cdefs, _ = self._execute_nested_definitions()
                return len(cdefs)
            if self.universe is None and self.projection == "owners":
                if self.repo._query_index.can_execute_query_domain("nested"):
                    cdefs, _, _ = self.repo._query_index.execute_nested_owners(self)
                else:
                    cdefs, _, _ = self._execute_nested_owners()
                return len(cdefs)
            occurrences, _, _ = self._execute_nested_occurrences()
            if callable(occurrences):
                return sum(1 for _ in occurrences())
            return len(occurrences)

        cdefs, _, _ = self._execute_definition_domain()
        return len(cdefs)

    def _execute_explanation(self, *, analyze: bool = False, sql: bool = False) -> QueryExplanation:
        self._require_domain()
        if analyze:
            result = self.execute()
            explanation = result.explanation
            if explanation is None:
                return QueryStats(result_count=len(result)).explanation(domain=self._domain_label(), refresh=self.refresh_policy)
            return explanation
        if self.domain == "nested":
            if self.universe is None and self.projection == "definitions":
                _, stats = self._execute_nested_definitions()
            elif self.universe is None and self.projection == "owners":
                _, stats, _ = self._execute_nested_owners()
            else:
                _, stats, _ = self._execute_nested_occurrences()
            return stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)

        if self.universe is None and self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            stats = self.repo._query_index.explain_definition_domain(self, sql=sql)
            return stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)

        _, stats, _ = self._execute_definition_domain()
        return stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)

    def _require_domain(self) -> None:
        if self.domain is None:
            raise QueryDomainError("Select a query domain with stored(), cached(), known(), or nested() before executing.")

    def _domain_label(self) -> str:
        if self.domain == "nested" and self.projection is not None:
            return f"nested-{self.projection}"
        return self.domain or "unset"

    def _execute_definition_domain(self):
        stats = QueryStats()
        if self.universe is not None:
            if self.universe.kind != "definitions":
                raise QueryDomainError("A definition terminal cannot execute over an occurrence universe.")
            stats.universe_size = len(self.universe.definitions)
            matches = self._verify_cdefs(tuple(self.universe.definitions), stats=stats)
            stats.result_count = len(matches)
            replicas = {}
            if self.universe.replicas is not None:
                replicas = {cdef: self.universe.replicas.get(cdef, ()) for cdef in matches}
            return matches, stats, replicas

        if self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            return self.repo._query_index.execute_definition_domain(self)
        if self.domain == "known" and self.repo._query_index.can_execute_query_domain("stored"):
            return self._execute_federated_known_domain()

        catalog = self.repo._query_catalog
        exact_root = self.selector if isinstance(self.selector, ConcreteDefinition) else None
        if self.refresh_policy is True:
            catalog.refresh(True, stats=stats)
        elif exact_root is not None and self.domain in {"stored", "known"} and self.refresh_policy is not False:
            catalog.ensure_exact_stored(exact_root, stats=stats)
        elif self.domain in {"stored", "known"}:
            catalog.refresh(self.refresh_policy, stats=stats)

        live_domain = self._definition_domain(catalog)
        live_domain.prepare(stats=stats)
        with catalog.read_view(include_cached=self.domain in {"cached", "known"}) as snapshot:
            domain = live_domain.with_catalog(snapshot)
            if exact_root is not None and self.domain in {"stored", "cached", "known"}:
                candidate_ids = domain.filter(snapshot.exact_ids(exact_root))
                stats.universe_size = len(candidate_ids)
                stats.candidate_count = len(candidate_ids)
            else:
                stats.universe_size = domain.estimate_size()
                selector_graph = compile_selector_graph(self.selector, class_match=self.class_match_policy)
                if selector_graph is not None:
                    candidate_ids = graph_candidate_ids(snapshot, selector_graph, domain, stats=stats)
                else:
                    candidate_ids = domain.all_ids()
                    stats.candidate_count = len(candidate_ids)
                    stats.universe_size = len(candidate_ids)
            cdefs_by_id = snapshot.cdefs_by_id(candidate_ids)
            replicas = snapshot.replica_map(candidate_ids)
        cdefs = tuple(cdefs_by_id.values())
        matches = self._verify_cdefs(cdefs, stats=stats)
        stats.result_count = len(matches)
        replicas = {cdef: replicas.get(cdef, ()) for cdef in matches}
        return matches, stats, replicas

    def _execute_federated_known_domain(self, *, stop_after: int | None = None):
        from .federation import CACHE_SOURCE_KEY

        stored_query = replace(self, domain="stored")
        stored_cdefs, stored_stats, stored_replicas = self.repo._query_index.execute_definition_domain(stored_query, stop_after=stop_after)

        if stop_after is not None and len(stored_cdefs) >= stop_after:
            stats = QueryStats(refresh_action="federated-known")
            stats.store_scan_count = stored_stats.store_scan_count
            stats.candidate_count = stored_stats.candidate_count
            stats.verified_count = stored_stats.verified_count
            stats.result_count = len(stored_cdefs)
            stats.universe_size = None
            stats.generation_vector = dict(stored_stats.generation_vector or {})
            stats.source_plans = stored_stats.source_plans
            return stored_cdefs, stats, stored_replicas

        cached_query = replace(self, domain="cached")
        cached_cdefs, cached_stats, cached_replicas = cached_query._execute_definition_domain()
        cache_generation = self.repo._query_catalog.current_generation()

        merged = {cdef: cdef for cdef in stored_cdefs}
        for cdef in cached_cdefs:
            merged.setdefault(cdef, cdef)
            if stop_after is not None and len(merged) >= stop_after:
                break

        out = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        replicas = {}
        for cdef in out:
            if cdef in stored_replicas:
                replicas[cdef] = stored_replicas[cdef]
            else:
                replicas[cdef] = cached_replicas.get(cdef, ())

        stats = QueryStats(refresh_action="federated-known")
        stats.store_scan_count = stored_stats.store_scan_count + cached_stats.store_scan_count
        stats.candidate_count = stored_stats.candidate_count + cached_stats.candidate_count
        stats.verified_count = stored_stats.verified_count + cached_stats.verified_count
        stats.result_count = len(out)
        stats.universe_size = None
        stats.generation_vector = dict(stored_stats.generation_vector or {})
        stats.generation_vector[CACHE_SOURCE_KEY] = cache_generation
        stats.source_plans = (*stored_stats.source_plans, SourceQueryPlan(
            source_key=CACHE_SOURCE_KEY,
            backend="memory-cache",
            generation=cache_generation,
            candidate_count=cached_stats.candidate_count,
            verified_count=cached_stats.verified_count,
            result_count=cached_stats.result_count,
            refresh_action=cached_stats.refresh_action,
        ))
        return out, stats, replicas

    def _execute_terminal_items(self, *, stop_after: int):
        self._require_domain()
        if self.domain == "nested":
            if self.universe is None and self.projection == "definitions" and self.repo._query_index.can_execute_query_domain("nested"):
                return self.repo._query_index.execute_nested_definitions(self, stop_after=stop_after)[0]
            if self.universe is None and self.projection == "owners" and self.repo._query_index.can_execute_query_domain("nested"):
                return self.repo._query_index.execute_nested_owners(self, stop_after=stop_after)[0]
            if self.universe is None and self.projection is None:
                limit = stop_after if self.occurrence_limit is None else min(self.occurrence_limit, stop_after)
                occurrences, _, _ = replace(self, occurrence_limit=limit)._execute_nested_occurrences()
                if callable(occurrences):
                    return tuple(item for _, item in zip(range(stop_after), occurrences()))
                return tuple(occurrences[:stop_after])
            result = self.execute()
            return tuple(item for _, item in zip(range(stop_after), result))

        if self.universe is None and self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            return self.repo._query_index.execute_definition_domain(self, stop_after=stop_after)[0]
        if self.universe is None and self.domain == "known" and self.repo._query_index.can_execute_query_domain("stored"):
            return self._execute_federated_known_domain(stop_after=stop_after)[0]
        cdefs, _, _ = self._execute_definition_domain()
        return tuple(cdefs[:stop_after])

    def _definition_domain(self, catalog):
        if self.domain == "stored":
            return StoredDomain(catalog)
        if self.domain == "cached":
            return CachedDomain(catalog, reuse_weak=self.reuse_weak_policy)
        if self.domain == "known":
            return KnownDomain(catalog, reuse_weak=self.reuse_weak_policy)
        raise QueryDomainError(f"Unsupported definition domain {self.domain!r}.")

    def _execute_nested_occurrences(self):
        if self.universe is not None:
            stats = QueryStats()
            if self.universe.kind != "occurrences":
                raise QueryDomainError("A nested query cannot execute over a definition universe.")
            stats.universe_size = len(self.universe.occurrences)
            verified_nested = self._verify_cdefs(
                tuple({occ.definition for occ in self.universe.occurrences}),
                stats=stats,
            )
            verified = set(verified_nested)
            out = tuple(occ for occ in self.universe.occurrences if occ.definition in verified)
            if self.occurrence_limit is not None:
                out = out[:self.occurrence_limit]
            stats.result_count = len(out)
            return out, stats, self.universe.replicas

        if self.repo._query_index.can_execute_query_domain("nested"):
            return self.repo._query_index.execute_nested_occurrences(self)

        catalog = self.repo._query_catalog
        for _ in range(_MAX_NESTED_QUERY_RETRIES):
            stats = QueryStats()
            captured = self._capture_nested_candidates(catalog, stats)
            _, match_ids = self._verify_cdefs_by_id(captured.cdefs_by_id, stats=stats)
            try:
                traversal = self._capture_occurrence_traversal(catalog, match_ids, captured.generation)
                break
            except _QueryGenerationChanged:
                continue
        else:
            raise QueryIndexError("Catalog generation changed repeatedly during nested occurrence query.")

        def occurrence_factory():
            return traversal.iter_occurrences(max_occurrences=self.occurrence_limit)

        return occurrence_factory, stats, traversal.owner_replicas

    def _execute_nested_definitions(self) -> tuple[tuple[ConcreteDefinition, ...], QueryStats]:
        if self.universe is None and self.repo._query_index.can_execute_query_domain("nested"):
            return self.repo._query_index.execute_nested_definitions(self)
        matches, _, stats, _ = self._execute_nested_definition_matches()
        stats.result_count = len(matches)
        return matches, stats

    def _execute_nested_owners(self):
        if self.universe is None and self.repo._query_index.can_execute_query_domain("nested"):
            return self.repo._query_index.execute_nested_owners(self)
        catalog = self.repo._query_catalog
        for _ in range(_MAX_NESTED_QUERY_RETRIES):
            matches, match_ids, stats, generation = self._execute_nested_definition_matches()
            try:
                projection = self._project_owners(catalog, match_ids, generation)
                break
            except _QueryGenerationChanged:
                continue
        else:
            raise QueryIndexError("Catalog generation changed repeatedly during nested owner query.")
        owners = projection.cdefs
        owner_replicas = projection.replicas
        stats.result_count = len(owners)
        owners = tuple(sorted(owners, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        return owners, stats, {cdef: owner_replicas.get(cdef, ()) for cdef in owners}

    def _execute_nested_definition_matches(self):
        stats = QueryStats()
        catalog = self.repo._query_catalog
        captured = self._capture_nested_candidates(catalog, stats)
        matches, match_ids = self._verify_cdefs_by_id(captured.cdefs_by_id, stats=stats)
        return matches, match_ids, stats, captured.generation

    def _capture_nested_candidates(self, catalog, stats: QueryStats) -> CapturedNestedCandidates:
        catalog.refresh(self.refresh_policy, stats=stats)
        with catalog.read_view(include_cached=False) as snapshot:
            selector_graph = compile_selector_graph(self.selector, class_match=self.class_match_policy)
            if selector_graph is not None:
                candidate_ids = graph_candidate_ids(snapshot, selector_graph, None, stats=stats)
                candidate_ids = snapshot.filter_nested_ids(candidate_ids)
                stats.candidate_count = len(candidate_ids)
            else:
                domain = NestedDomain(snapshot)
                candidate_ids = domain.all_ids()
                stats.candidate_count = len(candidate_ids)
                stats.universe_size = None
            cdefs_by_id = snapshot.cdefs_by_id(candidate_ids)
            generation = snapshot.generation
        return CapturedNestedCandidates(generation=generation, cdefs_by_id=cdefs_by_id, stats=stats)

    def _capture_occurrence_traversal(
            self,
            catalog,
            ids: set[DefinitionId] | frozenset[DefinitionId],
            generation: int):
        with catalog.read_view(include_cached=False) as snapshot:
            if snapshot.generation != generation:
                raise _QueryGenerationChanged
            return snapshot.occurrence_snapshot_for_nested_ids(set(ids))

    def _project_owners(
            self,
            catalog,
            ids: set[DefinitionId] | frozenset[DefinitionId],
            generation: int):
        with catalog.read_view(include_cached=False) as snapshot:
            if snapshot.generation != generation:
                raise _QueryGenerationChanged
            return snapshot.project_owners(set(ids))

    def _verify_cdefs_by_id(
            self,
            cdefs_by_id: dict[DefinitionId, ConcreteDefinition],
            *,
            stats: QueryStats) -> tuple[tuple[ConcreteDefinition, ...], set[DefinitionId]]:
        matches = self._verify_cdefs(tuple(cdefs_by_id.values()), stats=stats)
        match_set = set(matches)
        match_ids = {did for did, cdef in cdefs_by_id.items() if cdef in match_set}
        return matches, match_ids

    def _verify_cdefs(
            self,
            cdefs: tuple[ConcreteDefinition, ...],
            *,
            stats: QueryStats) -> tuple[ConcreteDefinition, ...]:
        if self.selector is None:
            stats.verified_count += len(cdefs)
            stats.python_verifications += len(cdefs)
            if self.max_verify_limit is not None and stats.verified_count > self.max_verify_limit:
                raise QueryVerifyBudgetExceeded(
                    f"Query exceeded max_verify budget {self.max_verify_limit}: verified {stats.verified_count} CDefs."
                )
            return tuple(sorted(cdefs, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))

        out: list[ConcreteDefinition] = []
        for cdef in cdefs:
            stats.verified_count += 1
            stats.python_verifications += 1
            if self.max_verify_limit is not None and stats.verified_count > self.max_verify_limit:
                raise QueryVerifyBudgetExceeded(
                    f"Query exceeded max_verify budget {self.max_verify_limit}: verified {stats.verified_count} CDefs."
                )
            if not _structural_match(
                    self.selector,
                    cdef,
                    strict=self.strict_policy,
                    class_match=self.class_match_policy):
                continue
            out.append(cdef)
        return tuple(sorted(out, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))


def _snapshot_source(source):
    if source is None:
        return None
    if isinstance(source, Object):
        return source.definition
    if isinstance(source, ConcreteDefinition):
        return source
    if isinstance(source, Selector):
        return source.root
    if isinstance(source, Definition):
        return deepcopy(source)
    raise TypeError(f"Query source must be Selector, Definition, ConcreteDefinition, Object, or None, not {type(source).__name__}.")


def _structural_match(selector, cdef: ConcreteDefinition, *, strict: bool, class_match: ClassMatchPolicy) -> bool:
    if not _query_match(selector, cdef, strict=strict, class_match=class_match):
        return False
    return True


def _query_match(selector, target, *, strict: bool, class_match: ClassMatchPolicy) -> bool:
    """Query-layer verifier: selector semantics plus exact ConcreteDefinition anchors."""
    if isinstance(selector, Object):
        selector = selector.definition
    if isinstance(target, Object):
        target = target.definition
    if isinstance(selector, Selector):
        selector = selector.root

    if isinstance(selector, ConcreteDefinition):
            return isinstance(target, ConcreteDefinition) and cdef_equal(selector, target)

    if isinstance(selector, DefLink):
        from ..cdef_graph import EdgeKind
        if selector.kind is EdgeKind.MATERIALIZE:
            target_value = target.target if isinstance(target, DefLink) and target.kind is EdgeKind.MATERIALIZE else target
            return _query_match(selector.target, target_value, strict=strict, class_match=class_match)
        if selector.kind is EdgeKind.REF:
            if not isinstance(target, DefLink) or target.kind is not EdgeKind.REF:
                return False
            return _query_match(selector.target, target.target, strict=strict, class_match=class_match)
        return False

    if isinstance(selector, (QuotedDef, SelectorSpec)):
        if not isinstance(target, (QuotedDef, SelectorSpec, Selector, Definition)):
            return False
        sel_value = selector.value if isinstance(selector, QuotedDef) else selector.selector
        tgt_value = target.value if isinstance(target, QuotedDef) else target.selector if isinstance(target, SelectorSpec) else target
        if isinstance(sel_value, Selector):
            sel_value = sel_value.root
        if isinstance(tgt_value, Selector):
            tgt_value = tgt_value.root
        return _query_match(sel_value, tgt_value, strict=strict, class_match=class_match)

    if isinstance(selector, Par):
        return selector.matches(target, present=True)

    if isinstance(selector, Definition):
        selector = apply_definition_arg_roles(selector)
        if not isinstance(target, (Definition, ConcreteDefinition)):
            return False
        if selector.cls is not None:
            if not _query_match_class(selector.cls, target.cls, strict=strict, class_match=class_match):
                return False
        if isinstance(target, ConcreteDefinition) and target.identity_version == V2_IDENTITY_VERSION:
            # V2 identities persist semantic names rather than a particular
            # positional/keyword call spelling.  Partial binding deliberately
            # omits defaults, so only supplied parameters constrain matching.
            try:
                selector_parameters = selector.parameters
            except TypeError:
                # Missing/Present selectors may intentionally mention a
                # parameter absent from the live constructor. Bind the known
                # portion, then retain those structural absence constraints.
                from ..arg_roles import apply_bound_arg_roles
                from ..bound_args import _constructor_signature, bind_partial_arguments

                if selector.cls is None or not isinstance(selector.cls, type):
                    raise
                signature = _constructor_signature(selector.cls)
                known_kwargs = {
                    name: value
                    for name, value in selector.kwargs.items()
                    if name in signature.parameters
                }
                unknown_kwargs = {
                    name: value
                    for name, value in selector.kwargs.items()
                    if name not in signature.parameters
                }
                args = () if selector.args is None else tuple(selector.args)
                bound = bind_partial_arguments(selector.cls, args, known_kwargs)
                bound = apply_bound_arg_roles(selector.cls, bound)
                selector_parameters = dict(bound.items())
                selector_parameters.update(unknown_kwargs)
            for name, child in selector_parameters.items():
                if name not in target.parameters:
                    if isinstance(child, Par) and child.matches(None, present=False):
                        continue
                    return False
                if not _query_match(child, target.parameters[name], strict=strict, class_match=class_match):
                    return False
            return True
        if selector.args is not None:
            if target.args is None:
                return False
            if not _query_match(selector.args, target.args, strict=strict, class_match=class_match):
                return False
        for key, child in selector.kwargs.items():
            if key not in target.kwargs:
                if isinstance(child, Par) and child.matches(None, present=False):
                    continue
                return False
            if not _query_match(child, target.kwargs[key], strict=strict, class_match=class_match):
                return False
        return True

    if isinstance(selector, (dict, FrozenDict)):
        if not isinstance(target, (dict, FrozenDict)):
            return False
        for key, child in selector.items():
            if key not in target:
                if isinstance(child, Par) and child.matches(None, present=False):
                    continue
                return False
            if not _query_match(child, target[key], strict=strict, class_match=class_match):
                return False
        return True

    family = matching_container_family(selector, target)
    if family in {"list", "tuple"}:
        if len(selector) != len(target):
            return False
        return all(
            _query_match(sel_child, tgt_child, strict=strict, class_match=class_match)
            for sel_child, tgt_child in zip(selector, target)
        )

    if family == "set":
        return _unordered_match(selector, target, lambda sel_child, tgt_child: _query_match(
            sel_child,
            tgt_child,
            strict=strict,
            class_match=class_match,
        ))

    return _query_match_leaf(selector, target, strict=strict, class_match=class_match)


def _query_match_class(selector, target, *, strict: bool, class_match: ClassMatchPolicy) -> bool:
    selector_ref = maybe_symbol_ref(selector, functions=False)
    target_ref = maybe_symbol_ref(target, functions=False)
    if selector_ref is not None and target_ref is not None:
        if selector_ref == target_ref:
            return True
        if strict or class_match == "exact":
            return False
        try:
            selector_obj = resolve_symbol(selector_ref)
            target_obj = resolve_symbol(target_ref)
        except Exception:
            return False
        if isinstance(selector_obj, type) and isinstance(target_obj, type):
            return issubclass(target_obj, selector_obj)
        return False

    if isinstance(selector, type) and isinstance(target, type):
        if strict or class_match == "exact":
            return selector is target
        return issubclass(target, selector)

    return _query_match_leaf(selector, target, strict=strict, class_match=class_match)


def _query_match_leaf(selector, target, *, strict: bool, class_match: ClassMatchPolicy) -> bool:
    if is_nonclass_callable(selector):
        if strict:
            raise TypeError("Callable selectors are not allowed in strict query matching.")
        return bool(selector(target))

    selector_ref = maybe_symbol_ref(selector, functions=False)
    target_ref = maybe_symbol_ref(target, functions=False)
    if selector_ref is not None and target_ref is not None:
        if selector_ref == target_ref:
            return True
        if strict or class_match == "exact":
            return False
        try:
            selector_obj = resolve_symbol(selector_ref)
            target_obj = resolve_symbol(target_ref)
        except Exception:
            return False
        if isinstance(selector_obj, type) and isinstance(target_obj, type):
            return issubclass(target_obj, selector_obj)
        return False

    try:
        return selector_match(selector, target, strict=strict)
    except TypeError:
        return False


def _unordered_match(selector_values, target_values, edge_predicate) -> bool:
    selector_list = list(selector_values)
    target_list = list(target_values)
    if len(selector_list) != len(target_list):
        return False

    edges = [
        [idx for idx, tgt in enumerate(target_list) if edge_predicate(sel, tgt)]
        for sel in selector_list
    ]
    order = sorted(range(len(selector_list)), key=lambda idx: len(edges[idx]))
    matched_to_selector: dict[int, int] = {}

    def augment(sel_idx: int, seen: set[int]) -> bool:
        for tgt_idx in edges[sel_idx]:
            if tgt_idx in seen:
                continue
            seen.add(tgt_idx)
            if tgt_idx not in matched_to_selector or augment(matched_to_selector[tgt_idx], seen):
                matched_to_selector[tgt_idx] = sel_idx
                return True
        return False

    for sel_idx in order:
        if not augment(sel_idx, set()):
            return False
    return True
