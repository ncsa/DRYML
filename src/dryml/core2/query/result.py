from __future__ import annotations

from collections.abc import Iterable, Mapping
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
        object.__setattr__(self, "repo", repo)
        definitions_t = _sort_cdefs(dict.fromkeys(definitions).keys())
        object.__setattr__(self, "_definitions", definitions_t)
        object.__setattr__(self, "materializable", materializable)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "explanation", explanation)
        if replicas is None:
            replicas = {cdef: repo._query_catalog.stores_for_cdef(cdef) for cdef in definitions_t}
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
            replicas={**other._replicas, **self._replicas},
        )

    def intersection(self, other: "DefinitionResultSet") -> "DefinitionResultSet":
        self._check_compatible(other)
        return DefinitionResultSet(
            self.repo,
            [cdef for cdef in self._definitions if cdef in other],
            materializable=self.materializable and other.materializable,
            domain=self.domain,
            replicas={cdef: self._replicas.get(cdef, ()) for cdef in self._definitions if cdef in other},
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

    def replicas(self, cdef: ConcreteDefinition) -> tuple[Any, ...]:
        return self._replicas.get(cdef, ())

    def _check_compatible(self, other: "DefinitionResultSet") -> None:
        if self.repo is not other.repo:
            raise ValueError("Cannot combine result sets from different repos.")
        if self.domain != other.domain or self.materializable != other.materializable:
            raise ValueError(
                "Cannot combine DefinitionResultSets with different domains or materialization semantics."
            )


@dataclass(frozen=True, slots=True)
class OccurrenceResultSet:
    repo: Any
    _occurrences: tuple[DefinitionOccurrence, ...]
    explanation: QueryExplanation | None = None

    def __init__(
            self,
            repo,
            occurrences: Iterable[DefinitionOccurrence],
            *,
            explanation: QueryExplanation | None = None):
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "_occurrences", _sort_occurrences(occurrences))
        object.__setattr__(self, "explanation", explanation)

    def __iter__(self) -> Iterator[DefinitionOccurrence]:
        return iter(self._occurrences)

    def __len__(self) -> int:
        return len(self._occurrences)

    def count(self) -> int:
        return len(self)

    def exists(self) -> bool:
        return len(self) > 0

    def one(self) -> DefinitionOccurrence:
        if len(self) != 1:
            raise QueryCardinalityError(f"Expected exactly one occurrence, found {len(self)}.")
        return self._occurrences[0]

    def one_or_none(self) -> DefinitionOccurrence | None:
        if len(self) > 1:
            raise QueryCardinalityError(f"Expected zero or one occurrence, found {len(self)}.")
        return self._occurrences[0] if self._occurrences else None

    def first(self) -> DefinitionOccurrence | None:
        return self._occurrences[0] if self._occurrences else None

    def definitions(self) -> DefinitionResultSet:
        return DefinitionResultSet(
            self.repo,
            [occ.definition for occ in self._occurrences],
            materializable=False,
            domain="nested-definitions",
            explanation=self.explanation,
        )

    def owners(self) -> DefinitionResultSet:
        return DefinitionResultSet(
            self.repo,
            [occ.owner for occ in self._occurrences],
            materializable=True,
            domain="owners",
            explanation=self.explanation,
        )

    def objects(self, **kwargs):
        raise QueryDomainError("Raw nested occurrences cannot be materialized directly; use .owners().objects().")

    def refine(self, selector) -> "OccurrenceResultSet":
        return self.query(selector).execute()

    def query(self, selector=None):
        from .query import DefinitionQuery

        universe = ResultUniverse(
            kind="occurrences",
            occurrences=self._occurrences,
            materializable=False,
            domain="nested",
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
        for occ in tuple(self._occurrences) + tuple(other._occurrences):
            key = (occ.owner, occ.path, occ.definition)
            if key not in seen:
                seen.add(key)
                out.append(occ)
        return OccurrenceResultSet(self.repo, out)

    def intersection(self, other: "OccurrenceResultSet") -> "OccurrenceResultSet":
        self._check_compatible(other)
        other_keys = {(occ.owner, occ.path, occ.definition) for occ in other._occurrences}
        return OccurrenceResultSet(
            self.repo,
            [occ for occ in self._occurrences if (occ.owner, occ.path, occ.definition) in other_keys],
        )

    def _check_compatible(self, other: "OccurrenceResultSet") -> None:
        if self.repo is not other.repo:
            raise ValueError("Cannot combine result sets from different repos.")


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
