from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .path import DefinitionPath


RefreshPolicy = Literal[False, "auto", True]
ClassMatchPolicy = Literal["selector", "exact"]
QueryDomain = Literal["stored", "cached", "known", "nested"]
QueryProjection = Literal["definitions", "owners"]


class QueryError(Exception):
    pass


class QueryDomainError(QueryError):
    pass


class QueryCardinalityError(QueryError):
    pass


class QueryIndexError(QueryError):
    pass


@dataclass(frozen=True, slots=True)
class ExactSubtreeConstraint:
    path: DefinitionPath
    cdef: Any


@dataclass(frozen=True, slots=True)
class FeatureToken:
    kind: str
    path: DefinitionPath | None
    payload: Any = None


@dataclass(frozen=True, slots=True)
class FeatureRequirement:
    token: FeatureToken
    count: int = 1


FINGERPRINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DefinitionFingerprint:
    counts: dict[FeatureToken, int]
    schema_version: int = FINGERPRINT_SCHEMA_VERSION


DefinitionId = str
StoreId = str
OccurrenceKey = tuple[DefinitionId, DefinitionPath, DefinitionId]


@dataclass(frozen=True, slots=True)
class DefinitionRecord:
    definition_id: DefinitionId
    cdef: Any
    class_key: str
    fingerprint: DefinitionFingerprint


@dataclass(frozen=True, slots=True)
class StoredReplica:
    definition_id: DefinitionId
    store_id: StoreId


@dataclass(frozen=True, slots=True)
class DefinitionOccurrence:
    owner: Any
    path: DefinitionPath
    definition: Any


@dataclass(frozen=True, slots=True)
class QueryExplanation:
    domain: str
    refresh: RefreshPolicy
    refresh_action: str = "none"
    store_scan_count: int = 0
    universe_size: int = 0
    selected_features: tuple[FeatureRequirement, ...] = ()
    posting_sizes: tuple[int, ...] = ()
    candidate_count: int = 0
    verified_count: int = 0
    result_count: int = 0
    fast_path: str | None = None

    def format(self) -> str:
        lines = [
            f"domain: {self.domain}",
            f"refresh: {self.refresh!r}",
            f"refresh action: {self.refresh_action}",
            f"store scans: {self.store_scan_count}",
            f"universe size: {self.universe_size}",
            f"features: {len(self.selected_features)}",
            f"candidates: {self.candidate_count}",
            f"verified: {self.verified_count}",
            f"results: {self.result_count}",
        ]
        if self.fast_path is not None:
            lines.append(f"fast path: {self.fast_path}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True, slots=True)
class ResultUniverse:
    kind: Literal["definitions", "occurrences"]
    definitions: tuple[Any, ...] = ()
    occurrences: tuple[DefinitionOccurrence, ...] = ()
    materializable: bool = False
    domain: str = "stored"


@dataclass(slots=True)
class QueryStats:
    refresh_action: str = "none"
    store_scan_count: int = 0
    universe_size: int = 0
    selected_features: tuple[FeatureRequirement, ...] = ()
    posting_sizes: tuple[int, ...] = ()
    candidate_count: int = 0
    verified_count: int = 0
    result_count: int = 0
    fast_path: str | None = None

    def explanation(self, *, domain: str, refresh: RefreshPolicy) -> QueryExplanation:
        return QueryExplanation(
            domain=domain,
            refresh=refresh,
            refresh_action=self.refresh_action,
            store_scan_count=self.store_scan_count,
            universe_size=self.universe_size,
            selected_features=self.selected_features,
            posting_sizes=self.posting_sizes,
            candidate_count=self.candidate_count,
            verified_count=self.verified_count,
            result_count=self.result_count,
            fast_path=self.fast_path,
        )
