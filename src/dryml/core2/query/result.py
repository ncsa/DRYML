from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Iterator

from ..definition import ConcreteDefinition
from ..object import Object
from ..policies import CachePolicy, InstancePolicy, RepoLoadOptions
from .model import (
    DefinitionOccurrence,
    QueryCardinalityError,
    QueryDomainError,
    QueryExplanation,
    ResultUniverse,
)


def _sort_cdefs(cdefs: Iterable[ConcreteDefinition]) -> tuple[ConcreteDefinition, ...]:
    return tuple(sorted(cdefs, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))


def _sort_occurrences(occurrences: Iterable[DefinitionOccurrence]) -> tuple[DefinitionOccurrence, ...]:
    return tuple(sorted(
        occurrences,
        key=lambda occ: (occ.owner.stable_hash(), str(occ.path), occ.definition.stable_hash(), repr(occ.definition)),
    ))


def _store_key(store) -> str:
    if hasattr(store, "catalog_key"):
        return store.catalog_key()
    return f"{type(store).__module__}.{type(store).__qualname__}:id:{id(store)}"


def _merge_replica_maps(*maps: Mapping[ConcreteDefinition, tuple[Any, ...]]) -> dict[ConcreteDefinition, tuple[Any, ...]]:
    merged: dict[ConcreteDefinition, dict[str, Any]] = {}
    for replica_map in maps:
        for cdef, stores in replica_map.items():
            bucket = merged.setdefault(cdef, {})
            for store in stores:
                bucket.setdefault(_store_key(store), store)
    return {
        cdef: tuple(bucket[key] for key in sorted(bucket))
        for cdef, bucket in merged.items()
    }


@dataclass(frozen=True, slots=True)
class DefinitionResultSet:
    repo: Any
    _definitions: tuple[ConcreteDefinition, ...]
    materializable: bool = True
    domain: str = "stored"
    explanation: QueryExplanation | None = None
    _replicas: dict[ConcreteDefinition, tuple[Any, ...]] | None = None

    def __init__(
            self,
            repo,
            definitions: Iterable[ConcreteDefinition],
            *,
            materializable: bool = True,
            domain: str = "stored",
            explanation: QueryExplanation | None = None,
            replicas: Mapping[ConcreteDefinition, tuple[Any, ...]] | None = None):
        if replicas is None:
            raise ValueError("DefinitionResultSet requires explicit replica metadata; use {} for nonmaterializable results.")
        object.__setattr__(self, "repo", repo)
        definitions_t = _sort_cdefs(dict.fromkeys(definitions).keys())
        object.__setattr__(self, "_definitions", definitions_t)
        object.__setattr__(self, "materializable", materializable)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "explanation", explanation)
        if materializable:
            missing = set(definitions_t) - set(replicas)
            if missing:
                raise ValueError("Materializable DefinitionResultSet requires a replica entry for every definition.")
        object.__setattr__(self, "_replicas", dict(replicas))

    def __iter__(self) -> Iterator[ConcreteDefinition]:
        return iter(self._definitions)

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, item: object) -> bool:
        return item in self._definitions

    def count(self) -> int:
        return len(self)

    def exists(self) -> bool:
        return len(self) > 0

    def one(self) -> ConcreteDefinition:
        if len(self) != 1:
            raise QueryCardinalityError(f"Expected exactly one result, found {len(self)}.")
        return self._definitions[0]

    def one_or_none(self) -> ConcreteDefinition | None:
        if len(self) > 1:
            raise QueryCardinalityError(f"Expected zero or one result, found {len(self)}.")
        return self._definitions[0] if self._definitions else None

    def first(self) -> ConcreteDefinition | None:
        return self._definitions[0] if self._definitions else None

    def refine(self, selector) -> "DefinitionResultSet":
        return self.query(selector).defs()

    def query(self, selector=None):
        from .query import DefinitionQuery

        universe = ResultUniverse(
            kind="definitions",
            definitions=self._definitions,
            materializable=self.materializable,
            domain=self.domain,
            replicas=dict(self._replicas),
        )
        return DefinitionQuery.from_source(
            self.repo,
            selector,
            domain=self.domain,
            universe=universe,
        )

    def union(self, other: "DefinitionResultSet") -> "DefinitionResultSet":
        self._check_compatible(other)
        return DefinitionResultSet(
            self.repo,
            tuple(self._definitions) + tuple(other._definitions),
            materializable=self.materializable and other.materializable,
            domain=self.domain,
            replicas=_merge_replica_maps(self._replicas, other._replicas),
        )

    def intersection(self, other: "DefinitionResultSet") -> "DefinitionResultSet":
        self._check_compatible(other)
        kept = [cdef for cdef in self._definitions if cdef in other]
        merged_replicas = _merge_replica_maps(self._replicas, other._replicas)
        return DefinitionResultSet(
            self.repo,
            kept,
            materializable=self.materializable and other.materializable,
            domain=self.domain,
            replicas={cdef: merged_replicas.get(cdef, ()) for cdef in kept},
        )

    def objects(
            self,
            *,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision=None,
            options: RepoLoadOptions | None = None) -> "ObjectResultSet":
        if not self.materializable:
            raise QueryDomainError(f"Definitions from domain {self.domain!r} cannot be materialized directly.")
        objs = {}
        for cdef in self._definitions:
            replicas = self._replicas.get(cdef, ())
            if replicas:
                self.repo.set_object_store(cdef, replicas[0])
            objs[cdef] = self.repo.load_object(
                cdef,
                instance=instance,
                restore_state=restore_state,
                build_missing=False,
                reuse_weak=reuse_weak,
                cache=cache,
                revision=revision,
                options=options,
            )
        return ObjectResultSet(self.repo, objs, domain=self.domain, explanation=self.explanation)

    def apply(self, func, *args, **kwargs):
        """Materialize this definition set and apply ``func`` to each object."""
        return self.objects().apply(func, *args, **kwargs)

    def replicas(self, cdef: ConcreteDefinition) -> tuple[Any, ...]:
        return self._replicas.get(cdef, ())

    def _check_compatible(self, other: "DefinitionResultSet") -> None:
        if self.repo is not other.repo:
            raise ValueError("Cannot combine result sets from different repos.")
        if self.domain != other.domain or self.materializable != other.materializable:
            raise ValueError(
                "Cannot combine DefinitionResultSets with different domains or materialization semantics."
            )


class QueryBackedDefinitionResultSet(DefinitionResultSet):
    """Definition result set that pages verified CDefs from a replayable query.

    The result set stores no SQLite connection, cursor, or read transaction. It
    asks its page factory for fresh bounded read views during iteration and
    caches verified results as they are yielded so repeated full iteration is
    stable without re-querying.
    """

    __slots__ = ("_page_factory", "_definition_cache", "_replica_cache", "_cache_complete")

    def __init__(
            self,
            repo,
            page_factory: Callable[[], Iterable[tuple[ConcreteDefinition, tuple[Any, ...]]]],
            *,
            materializable: bool = True,
            domain: str = "stored",
            explanation: QueryExplanation | None = None):
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "_definitions", ())
        object.__setattr__(self, "materializable", materializable)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "explanation", explanation)
        object.__setattr__(self, "_replicas", {})
        object.__setattr__(self, "_page_factory", page_factory)
        object.__setattr__(self, "_definition_cache", [])
        object.__setattr__(self, "_replica_cache", {})
        object.__setattr__(self, "_cache_complete", False)

    def __iter__(self) -> Iterator[ConcreteDefinition]:
        if self._cache_complete:
            return iter(self._definitions)
        return self._iter_query_backed()

    def __len__(self) -> int:
        return len(self._materialize_definitions())

    def __contains__(self, item: object) -> bool:
        return item in self._materialize_definitions()

    def count(self) -> int:
        return len(self)

    def exists(self) -> bool:
        return self.first() is not None

    def first(self) -> ConcreteDefinition | None:
        if self._definition_cache:
            return self._definition_cache[0]
        for cdef in self:
            return cdef
        return None

    def query(self, selector=None):
        self._materialize_definitions()
        return super().query(selector)

    def union(self, other: "DefinitionResultSet") -> "DefinitionResultSet":
        self._materialize_definitions()
        return super().union(other)

    def intersection(self, other: "DefinitionResultSet") -> "DefinitionResultSet":
        self._materialize_definitions()
        return super().intersection(other)

    def objects(self, **kwargs) -> "ObjectResultSet":
        self._materialize_definitions()
        return super().objects(**kwargs)

    def replicas(self, cdef: ConcreteDefinition) -> tuple[Any, ...]:
        if cdef not in self._replica_cache and not self._cache_complete:
            self._materialize_definitions()
        return self._replica_cache.get(cdef, ())

    def _iter_query_backed(self) -> Iterator[ConcreteDefinition]:
        seen = set(self._definition_cache)
        for cached in tuple(self._definition_cache):
            yield cached
        for cdef, replicas in self._page_factory():
            if cdef in seen:
                existing = self._replica_cache.get(cdef, ())
                self._replica_cache[cdef] = _merge_store_tuple(existing, tuple(replicas))
                continue
            seen.add(cdef)
            self._definition_cache.append(cdef)
            self._replica_cache[cdef] = tuple(replicas)
            yield cdef
        self._finish_cache()

    def _materialize_definitions(self) -> tuple[ConcreteDefinition, ...]:
        if not self._cache_complete:
            for _ in self._iter_query_backed():
                pass
        return self._definitions

    def _finish_cache(self) -> None:
        definitions = tuple(dict.fromkeys(self._definition_cache).keys())
        object.__setattr__(self, "_definitions", definitions)
        object.__setattr__(self, "_replicas", dict(self._replica_cache))
        object.__setattr__(self, "_cache_complete", True)


def _merge_store_tuple(left: tuple[Any, ...], right: tuple[Any, ...]) -> tuple[Any, ...]:
    merged: dict[str, Any] = {}
    for store in (*left, *right):
        merged.setdefault(_store_key(store), store)
    return tuple(merged[key] for key in sorted(merged))


@dataclass(frozen=True, slots=True)
class OccurrenceResultSet:
    repo: Any
    _occurrences: tuple[DefinitionOccurrence, ...] | None
    _occurrence_factory: Callable[[], Iterable[DefinitionOccurrence]] | None
    explanation: QueryExplanation | None = None
    _owner_replicas: dict[ConcreteDefinition, tuple[Any, ...]] | None = None

    def __init__(
            self,
            repo,
            occurrences: Iterable[DefinitionOccurrence] | None = None,
            *,
            occurrence_factory: Callable[[], Iterable[DefinitionOccurrence]] | None = None,
            explanation: QueryExplanation | None = None,
            owner_replicas: Mapping[ConcreteDefinition, tuple[Any, ...]] | None = None):
        if occurrences is None and occurrence_factory is None:
            occurrences = ()
        if occurrences is not None and occurrence_factory is not None:
            raise ValueError("Provide occurrences or occurrence_factory, not both.")
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "_occurrences", None if occurrences is None else _sort_occurrences(occurrences))
        object.__setattr__(self, "_occurrence_factory", occurrence_factory)
        object.__setattr__(self, "explanation", explanation)
        object.__setattr__(self, "_owner_replicas", None if owner_replicas is None else dict(owner_replicas))

    def __iter__(self) -> Iterator[DefinitionOccurrence]:
        if self._occurrences is not None:
            return iter(self._occurrences)
        return iter(self._occurrence_factory())

    def __len__(self) -> int:
        return len(self._materialize())

    def count(self) -> int:
        if self._occurrences is not None:
            return len(self._occurrences)
        return sum(1 for _ in self)

    def exists(self) -> bool:
        return next(iter(self), None) is not None

    def one(self) -> DefinitionOccurrence:
        occurrences = self._materialize()
        if len(occurrences) != 1:
            raise QueryCardinalityError(f"Expected exactly one occurrence, found {len(occurrences)}.")
        return occurrences[0]

    def one_or_none(self) -> DefinitionOccurrence | None:
        occurrences = self._materialize()
        if len(occurrences) > 1:
            raise QueryCardinalityError(f"Expected zero or one occurrence, found {len(occurrences)}.")
        return occurrences[0] if occurrences else None

    def first(self) -> DefinitionOccurrence | None:
        return next(iter(self), None)

    def definitions(self) -> DefinitionResultSet:
        occurrences = self._materialize()
        return DefinitionResultSet(
            self.repo,
            [occ.definition for occ in occurrences],
            materializable=False,
            domain="nested-definitions",
            explanation=self.explanation,
            replicas={},
        )

    def owners(self) -> DefinitionResultSet:
        occurrences = self._materialize()
        return DefinitionResultSet(
            self.repo,
            [occ.owner for occ in occurrences],
            materializable=True,
            domain="owners",
            explanation=self.explanation,
            replicas=self._require_owner_replicas(),
        )

    def objects(self, **kwargs):
        raise QueryDomainError("Raw nested occurrences cannot be materialized directly; use .owners().objects().")

    def refine(self, selector) -> "OccurrenceResultSet":
        return self.query(selector).execute()

    def query(self, selector=None):
        from .query import DefinitionQuery

        occurrences = self._materialize()
        universe = ResultUniverse(
            kind="occurrences",
            occurrences=occurrences,
            materializable=False,
            domain="nested",
            replicas=dict(self._owner_replicas or {}),
        )
        return DefinitionQuery.from_source(
            self.repo,
            selector,
            domain="nested",
            universe=universe,
        )

    def union(self, other: "OccurrenceResultSet") -> "OccurrenceResultSet":
        self._check_compatible(other)
        seen = set()
        out = []
        for occ in self._materialize() + other._materialize():
            key = (occ.owner, occ.path, occ.definition)
            if key not in seen:
                seen.add(key)
                out.append(occ)
        return OccurrenceResultSet(
            self.repo,
            out,
            explanation=self.explanation,
            owner_replicas=_merge_replica_maps(self._owner_replicas or {}, other._owner_replicas or {}),
        )

    def intersection(self, other: "OccurrenceResultSet") -> "OccurrenceResultSet":
        self._check_compatible(other)
        other_keys = {(occ.owner, occ.path, occ.definition) for occ in other._materialize()}
        return OccurrenceResultSet(
            self.repo,
            [occ for occ in self._materialize() if (occ.owner, occ.path, occ.definition) in other_keys],
            explanation=self.explanation,
            owner_replicas=_merge_replica_maps(self._owner_replicas or {}, other._owner_replicas or {}),
        )

    def _check_compatible(self, other: "OccurrenceResultSet") -> None:
        if self.repo is not other.repo:
            raise ValueError("Cannot combine result sets from different repos.")

    def _require_owner_replicas(self) -> dict[ConcreteDefinition, tuple[Any, ...]]:
        if self._owner_replicas is None:
            raise QueryDomainError("Owner replica metadata was not captured for this occurrence result.")
        occurrences = self._materialize()
        missing = {occ.owner for occ in occurrences} - set(self._owner_replicas)
        if missing:
            raise QueryDomainError("Owner replica metadata is incomplete for this occurrence result.")
        return dict(self._owner_replicas)

    def _materialize(self) -> tuple[DefinitionOccurrence, ...]:
        if self._occurrences is None:
            object.__setattr__(self, "_occurrences", _sort_occurrences(self._occurrence_factory()))
            object.__setattr__(self, "_occurrence_factory", None)
        return self._occurrences


@dataclass(frozen=True, slots=True)
class ObjectResultSet(Mapping):
    repo: Any
    _objects: dict[ConcreteDefinition, Object]
    domain: str = "stored"
    explanation: QueryExplanation | None = None

    def __init__(
            self,
            repo,
            objects: Mapping[ConcreteDefinition, Object],
            *,
            domain: str = "stored",
            explanation: QueryExplanation | None = None):
        object.__setattr__(self, "repo", repo)
        ordered = {cdef: objects[cdef] for cdef in _sort_cdefs(objects.keys())}
        object.__setattr__(self, "_objects", ordered)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "explanation", explanation)

    def __getitem__(self, key: ConcreteDefinition) -> Object:
        return self._objects[key]

    def __iter__(self) -> Iterator[ConcreteDefinition]:
        return iter(self._objects)

    def __len__(self) -> int:
        return len(self._objects)

    def count(self) -> int:
        return len(self)

    def exists(self) -> bool:
        return len(self) > 0

    def one(self) -> Object:
        if len(self) != 1:
            raise QueryCardinalityError(f"Expected exactly one object, found {len(self)}.")
        return next(iter(self._objects.values()))

    def one_or_none(self) -> Object | None:
        if len(self) > 1:
            raise QueryCardinalityError(f"Expected zero or one object, found {len(self)}.")
        return next(iter(self._objects.values())) if self._objects else None

    def first(self) -> Object | None:
        return next(iter(self._objects.values())) if self._objects else None

    def apply(self, func, *args, **kwargs) -> "ObjectResultSet":
        """Apply ``func`` to every loaded object in deterministic result order.

        The result set itself is returned so callers can chain additional result
        set operations while mutating or inspecting only the selected objects.
        """
        for obj in self._objects.values():
            func(obj, *args, **kwargs)
        return self
