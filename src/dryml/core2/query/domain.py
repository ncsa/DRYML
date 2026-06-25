from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from .model import DefinitionId, QueryStats


class DefinitionDomain(Protocol):
    name: str

    def prepare(self, *, stats: QueryStats | None = None) -> None:
        ...

    def contains(self, definition_id: DefinitionId) -> bool:
        ...

    def filter(self, definition_ids: Iterable[DefinitionId]) -> set[DefinitionId]:
        ...

    def estimate_size(self) -> int | None:
        ...

    def all_ids(self) -> set[DefinitionId]:
        ...

    def with_catalog(self, catalog: Any) -> "DefinitionDomain":
        ...


@dataclass(frozen=True, slots=True)
class StoredDomain:
    catalog: Any
    name: str = "stored"

    def prepare(self, *, stats: QueryStats | None = None) -> None:
        return None

    def contains(self, definition_id: DefinitionId) -> bool:
        return self.catalog.is_stored_id(definition_id)

    def filter(self, definition_ids: Iterable[DefinitionId]) -> set[DefinitionId]:
        return self.catalog.filter_stored_ids(definition_ids)

    def estimate_size(self) -> int | None:
        return None

    def all_ids(self) -> set[DefinitionId]:
        return self.catalog.all_stored_ids()

    def with_catalog(self, catalog: Any) -> "StoredDomain":
        return StoredDomain(catalog)


@dataclass(frozen=True, slots=True)
class CachedDomain:
    catalog: Any
    reuse_weak: bool = True
    name: str = "cached"

    def prepare(self, *, stats: QueryStats | None = None) -> None:
        self.catalog.sync_caches(reuse_weak=self.reuse_weak)

    def contains(self, definition_id: DefinitionId) -> bool:
        return self.catalog.is_cached_id(definition_id, reuse_weak=self.reuse_weak)

    def filter(self, definition_ids: Iterable[DefinitionId]) -> set[DefinitionId]:
        return {did for did in definition_ids if self.contains(did)}

    def estimate_size(self) -> int | None:
        return None

    def all_ids(self) -> set[DefinitionId]:
        return self.catalog.all_cached_ids(reuse_weak=self.reuse_weak)

    def with_catalog(self, catalog: Any) -> "CachedDomain":
        return CachedDomain(catalog, reuse_weak=self.reuse_weak)


@dataclass(frozen=True, slots=True)
class KnownDomain:
    catalog: Any
    reuse_weak: bool = True
    name: str = "known"

    def prepare(self, *, stats: QueryStats | None = None) -> None:
        self.catalog.sync_caches(reuse_weak=self.reuse_weak)

    def contains(self, definition_id: DefinitionId) -> bool:
        return (
            self.catalog.is_stored_id(definition_id)
            or self.catalog.is_cached_id(definition_id, reuse_weak=self.reuse_weak)
        )

    def filter(self, definition_ids: Iterable[DefinitionId]) -> set[DefinitionId]:
        return {did for did in definition_ids if self.contains(did)}

    def estimate_size(self) -> int | None:
        return None

    def all_ids(self) -> set[DefinitionId]:
        return self.catalog.all_known_ids(reuse_weak=self.reuse_weak)

    def with_catalog(self, catalog: Any) -> "KnownDomain":
        return KnownDomain(catalog, reuse_weak=self.reuse_weak)


@dataclass(frozen=True, slots=True)
class NestedDomain:
    catalog: Any
    name: str = "nested"

    def prepare(self, *, stats: QueryStats | None = None) -> None:
        return None

    def contains(self, definition_id: DefinitionId) -> bool:
        return self.catalog.has_stored_ancestor(definition_id)

    def filter(self, definition_ids: Iterable[DefinitionId]) -> set[DefinitionId]:
        return self.catalog.filter_nested_ids(set(definition_ids))

    def estimate_size(self) -> int | None:
        return None

    def all_ids(self) -> set[DefinitionId]:
        return self.catalog.nested_ids()

    def with_catalog(self, catalog: Any) -> "NestedDomain":
        return NestedDomain(catalog)
