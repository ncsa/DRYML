from __future__ import annotations

from collections.abc import Collection
from contextlib import AbstractContextManager
from typing import Any, Protocol

from ..definition import ConcreteDefinition
from .domain import DefinitionDomain
from .lowering import CandidateBatch, CandidateRelation, LoweredQueryPlan, LoweringDiagnostics, PagedResultCursor, QueryTerminal, ScanPolicy
from .model import DefinitionId, FeatureRequirement, IndexWriteResult, QueryIndexStatus, QueryStats, RefreshPolicy, ValidationReport
from .path import DefinitionPath


class DefinitionGraphIndex(Protocol):
    def all_definition_ids(self) -> set[DefinitionId]:
        ...

    def estimate_exact_ids(self, cdef: ConcreteDefinition) -> int:
        ...

    def estimate_local_candidates(self, requirements: tuple[FeatureRequirement, ...]) -> int:
        ...

    def exact_ids(self, cdef: ConcreteDefinition) -> set[DefinitionId]:
        ...

    def local_candidates(
            self,
            requirements: tuple[FeatureRequirement, ...],
            *,
            within: set[DefinitionId] | None = None,
            domain: DefinitionDomain | None = None,
            stats: QueryStats | None = None) -> set[DefinitionId]:
        ...

    def parents(
            self,
            child_ids: set[DefinitionId],
            path: DefinitionPath,
            *,
            unordered: bool,
            within: set[DefinitionId] | frozenset[DefinitionId] | None = None) -> set[DefinitionId]:
        ...

    def children(
            self,
            parent_ids: set[DefinitionId],
            path: DefinitionPath,
            *,
            unordered: bool,
            within: set[DefinitionId] | frozenset[DefinitionId] | None = None) -> set[DefinitionId]:
        ...


class QueryIndexReadView(DefinitionGraphIndex, Protocol):
    @property
    def source_key(self) -> str:
        ...

    @property
    def generation(self) -> int:
        ...

    def is_stored_id(self, did: DefinitionId) -> bool:
        ...

    def filter_stored_ids(self, ids: Collection[DefinitionId]) -> set[DefinitionId]:
        ...

    def all_stored_ids(self) -> set[DefinitionId]:
        ...

    def is_cached_id(self, did: DefinitionId, *, reuse_weak: bool = True) -> bool:
        ...

    def all_cached_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        ...

    def all_known_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        ...

    def nested_ids(self) -> set[DefinitionId]:
        ...

    def filter_nested_ids(self, ids: Collection[DefinitionId]) -> set[DefinitionId]:
        ...

    def has_stored_ancestor(self, did: DefinitionId) -> bool:
        ...

    def cdefs_by_id(self, ids: Collection[DefinitionId]) -> dict[DefinitionId, ConcreteDefinition]:
        ...

    def replica_map(self, ids: Collection[DefinitionId]) -> dict[ConcreteDefinition, tuple[Any, ...]]:
        ...

    def project_owners(self, ids: Collection[DefinitionId]) -> Any:
        ...

    def occurrence_snapshot_for_nested_ids(self, target_ids: Collection[DefinitionId]) -> Any:
        ...

    @property
    def supports_lowering(self) -> bool:
        ...

    def lower_selector_graph(
            self,
            selector_graph,
            domain: DefinitionDomain,
            *,
            terminal: QueryTerminal,
            scan_policy: ScanPolicy,
            diagnostics: LoweringDiagnostics | None = None,
            within_relation: str | None = None) -> LoweredQueryPlan:
        ...

    def iter_candidate_cdef_batches(
            self,
            plan: LoweredQueryPlan,
            *,
            after: PagedResultCursor | None = None,
            batch_size: int) -> Any:
        ...

    def iter_relation_cdef_batches(
            self,
            relation: CandidateRelation,
            *,
            after: PagedResultCursor | None = None,
            batch_size: int) -> Any:
        ...

    def relation_exact_stored(self, cdef: ConcreteDefinition) -> CandidateRelation:
        ...

    def relation_filter_domain(
            self,
            relation: CandidateRelation,
            domain: DefinitionDomain) -> CandidateRelation:
        ...

    def relation_parents(
            self,
            relation: CandidateRelation,
            path: DefinitionPath,
            *,
            unordered: bool = False) -> CandidateRelation:
        ...

    def relation_children(
            self,
            relation: CandidateRelation,
            path: DefinitionPath,
            *,
            unordered: bool = False) -> CandidateRelation:
        ...

    def relation_semijoin_child_exists(
            self,
            parent_relation: CandidateRelation,
            child_relation: CandidateRelation,
            path: DefinitionPath,
            *,
            unordered: bool = False) -> CandidateRelation:
        ...

    def relation_count_estimate(self, relation: CandidateRelation) -> int | None:
        ...

    def relation_exact_safe_count(self, relation: CandidateRelation) -> int | None:
        ...


class StoreQueryIndex(Protocol):
    @property
    def source_key(self) -> str:
        ...

    def read_view(self, *, include_cached: bool = True) -> AbstractContextManager[QueryIndexReadView]:
        ...

    def current_generation(self) -> int:
        ...

    def register_stored_roots(self, graph, roots: Collection[ConcreteDefinition]) -> IndexWriteResult:
        ...

    def remove_stored_roots(self, roots: Collection[ConcreteDefinition]) -> IndexWriteResult:
        ...

    def refresh(self, policy: RefreshPolicy, *, stats: QueryStats | None = None) -> None:
        ...

    def status(self) -> QueryIndexStatus:
        ...

    def validate(self, *, thorough: bool = False) -> ValidationReport:
        ...

    def ensure_exact_stored(self, cdef: ConcreteDefinition, *, stats: QueryStats | None = None) -> bool:
        ...

    def sync_caches(self, *, reuse_weak: bool = True) -> None:
        ...

    def close(self) -> None:
        ...
