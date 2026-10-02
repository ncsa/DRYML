from __future__ import annotations

from collections.abc import Callable, ItemsView, Iterable, Mapping, ValuesView
from dataclasses import dataclass
from typing import Any, Iterator

from ..definition import ConcreteDefinition
from ..object import Object
from ..policies import CachePolicy
from .model import (
    ContainmentCarrier,
    ContainmentContext,
    DefinitionOccurrence,
    QueryCardinalityError,
    QueryDomainError,
    QueryExplanation,
    ReferenceOccurrence,
    ResultUniverse,
    containment_witness_key,
)


def _sort_cdefs(cdefs: Iterable[ConcreteDefinition]) -> tuple[ConcreteDefinition, ...]:
    return tuple(sorted(cdefs, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))


ContainmentOccurrence = DefinitionOccurrence | ReferenceOccurrence


def _sort_occurrences(
        occurrences: Iterable[ContainmentOccurrence]) -> tuple[ContainmentOccurrence, ...]:
    """Return deterministic occurrences without projecting exact terminals."""

    return tuple(sorted(occurrences, key=containment_witness_key))


def _unique_witnesses(
        witnesses: Iterable[ContainmentOccurrence]) -> tuple[ContainmentOccurrence, ...]:
    """Deduplicate complete witnesses while retaining their canonical order."""

    unique = {containment_witness_key(item): item for item in witnesses}
    return tuple(unique[key] for key in sorted(unique))


def _witnesses_for_members(
        witnesses: Iterable[ContainmentOccurrence],
        carrier: ContainmentCarrier,
        members: Iterable[Any]) -> tuple[ContainmentOccurrence, ...]:
    """Restrict evidence to surviving result carriers after a set operation."""

    member_keys = (
        {containment_witness_key(item) for item in members}
        if carrier == "occurrence"
        else set(members)
    )
    return _unique_witnesses(
        item for item in witnesses
        if (
            containment_witness_key(item) in member_keys
            if carrier == "occurrence"
            else (item.owner if carrier == "owner" else item.target) in member_keys
        )
    )


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
    """Immutable structural query results with replica and witness evidence.

    Args:
        repo: Repo used for refinement and optional materialization.
        definitions: Structural result representatives.
        materializable: Whether :meth:`objects` may construct these definitions.
        domain: Human-readable source domain retained by refinements.
        explanation: Optional terminal execution diagnostics.
        replicas: Explicit definition-to-Store authority. Materializable results
            require an entry for every definition; use an empty mapping for
            nonmaterializable results.
        witnesses: Graph-distinct CDefs retained for exact TemplateSelector
            refinement before structural result deduplication.
        witness_complete: Whether ``witnesses`` is complete immutable evidence
            for this result universe.

    Raises:
        ValueError: If replica metadata is absent or incomplete.
        QueryDomainError: On later exact refinement when witness evidence is not
            complete.

    Construction snapshots the supplied iterables and has no Store side effects.
    """

    repo: Any
    _definitions: tuple[ConcreteDefinition, ...]
    materializable: bool = True
    domain: str = "stored"
    explanation: QueryExplanation | None = None
    _replicas: dict[ConcreteDefinition, tuple[Any, ...]] | None = None
    _witnesses: tuple[ConcreteDefinition, ...] = ()
    _witness_complete: bool = False
    _containment: ContainmentContext | None = None
    _containment_witnesses: tuple[ContainmentOccurrence, ...] = ()
    _containment_carrier: ContainmentCarrier = "target"

    def __init__(
            self,
            repo,
            definitions: Iterable[ConcreteDefinition],
            *,
            materializable: bool = True,
            domain: str = "stored",
            explanation: QueryExplanation | None = None,
            replicas: Mapping[ConcreteDefinition, tuple[Any, ...]] | None = None,
            witnesses: Iterable[ConcreteDefinition] = (),
            witness_complete: bool = False,
            containment: ContainmentContext | None = None,
            containment_witnesses: Iterable[ContainmentOccurrence] = (),
            containment_carrier: ContainmentCarrier = "target"):
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
        object.__setattr__(self, "_witnesses", tuple(witnesses))
        object.__setattr__(self, "_witness_complete", bool(witness_complete))
        if containment is not None and containment_carrier not in {"target", "owner"}:
            raise ValueError("DefinitionResultSet containment carrier must be target or owner.")
        object.__setattr__(self, "_containment", containment)
        object.__setattr__(self, "_containment_witnesses", _unique_witnesses(containment_witnesses))
        object.__setattr__(self, "_containment_carrier", containment_carrier)

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

        if self._containment is not None:
            universe = ResultUniverse(
                kind="definitions",
                definitions=self._definitions,
                materializable=self.materializable,
                domain="nested",
                replicas=dict(self._replicas),
                witnesses=self._witnesses,
                witness_complete=self._witness_complete,
                containment=self._containment,
                containment_witnesses=self._containment_witnesses,
                containment_carrier=self._containment_carrier,
            )
            query = DefinitionQuery.from_source(
                self.repo,
                selector,
                domain="nested",
                universe=universe,
            )
            query = query.nested(
                edges=self._containment.edges,
                contains_ref=self._containment.contains_ref,
            )
            return query.definitions() if self._containment_carrier == "target" else query.owners()
        universe = ResultUniverse(
            kind="definitions",
            definitions=self._definitions,
            materializable=self.materializable,
            domain=self.domain,
            replicas=dict(self._replicas),
            witnesses=self._witnesses,
            witness_complete=self._witness_complete,
        )
        return DefinitionQuery.from_source(
            self.repo,
            selector,
            domain=self.domain,
            universe=universe,
        )

    def union(self, other: "DefinitionResultSet") -> "DefinitionResultSet":
        self._check_compatible(other)
        containment = self._combined_containment(other)
        definitions = tuple(self._definitions) + tuple(other._definitions)
        return DefinitionResultSet(
            self.repo,
            definitions,
            materializable=self.materializable and other.materializable,
            domain=self.domain,
            replicas=_merge_replica_maps(self._replicas, other._replicas),
            containment=containment,
            containment_witnesses=_witnesses_for_members(
                (*self._containment_witnesses, *other._containment_witnesses),
                self._containment_carrier,
                definitions,
            ) if containment is not None else (),
            containment_carrier=self._containment_carrier,
        )

    def intersection(self, other: "DefinitionResultSet") -> "DefinitionResultSet":
        self._check_compatible(other)
        containment = self._combined_containment(other)
        kept = [cdef for cdef in self._definitions if cdef in other]
        merged_replicas = _merge_replica_maps(self._replicas, other._replicas)
        return DefinitionResultSet(
            self.repo,
            kept,
            materializable=self.materializable and other.materializable,
            domain=self.domain,
            replicas={cdef: merged_replicas.get(cdef, ()) for cdef in kept},
            containment=containment,
            containment_witnesses=_witnesses_for_members(
                (*self._containment_witnesses, *other._containment_witnesses),
                self._containment_carrier,
                kept,
            ) if containment is not None else (),
            containment_carrier=self._containment_carrier,
        )

    def objects(self, *, cache: CachePolicy = "weak") -> "ObjectResultSet":
        from dryml.runtime import materialization_admission

        """Materialize structural definitions without implicitly selecting state.

        Args:
            cache: Cache tier selected for constructed objects.

        Returns:
            Objects built from structural CDefs while preserving Ref/Mat values.

        Raises:
            QueryDomainError: If the definitions are nonmaterializable.
            RuntimeTransitionError: If orchestration prohibits materialization.
        """

        with materialization_admission(operation="definition_result_set_objects"):
            if not self.materializable:
                raise QueryDomainError(f"Definitions from domain {self.domain!r} cannot be materialized directly.")
            objs = {}
            for cdef in self._definitions:
                replicas = self._replicas.get(cdef, ())
                if replicas:
                    self.repo.set_object_store(cdef, replicas[0])
                objs[cdef] = self.repo.load_object(cdef, cache=cache)
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
        if (self._containment is None) != (other._containment is None):
            raise ValueError("Cannot combine containment and non-containment result sets.")
        if self._containment is not None:
            if self._containment_carrier != other._containment_carrier:
                raise ValueError("Cannot combine containment result sets with different projections.")
            if not self._containment.compatible_with(other._containment):
                raise ValueError("Cannot combine result sets with incompatible containment contexts.")

    def _combined_containment(self, other: "DefinitionResultSet") -> ContainmentContext | None:
        """Merge compatible containment evidence metadata for a set operation."""

        if self._containment is None:
            return None
        return self._containment.combined_with(other._containment)


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
        object.__setattr__(self, "_witnesses", ())
        object.__setattr__(self, "_witness_complete", False)
        object.__setattr__(self, "_containment", None)
        object.__setattr__(self, "_containment_witnesses", ())
        object.__setattr__(self, "_containment_carrier", "target")
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
    """Nested definition occurrences with owner authority and exact witnesses.

    Args:
        repo: Repo used for refinement and owner materialization.
        occurrences: Eager occurrence values, mutually exclusive with
            ``occurrence_factory``.
        occurrence_factory: Replayable lazy occurrence producer.
        explanation: Optional terminal execution diagnostics.
        owner_replicas: Owner-to-Store authority used by :meth:`owners`.
        witnesses: Graph-distinct occurrences retained for exact TemplateSelector
            refinement and projection.
        witness_complete: Whether ``witnesses`` is complete immutable evidence
            for this occurrence universe.

    Raises:
        ValueError: If both eager and lazy occurrence sources are supplied.
        QueryDomainError: On later exact refinement when witness evidence is not
            complete, or on direct occurrence materialization.

    Eager inputs and witness evidence are snapshotted without Store side effects;
    a lazy factory is invoked only by terminal iteration.
    """

    repo: Any
    _occurrences: tuple[ContainmentOccurrence, ...] | None
    _occurrence_factory: Callable[[], Iterable[ContainmentOccurrence]] | None
    explanation: QueryExplanation | None = None
    _owner_replicas: dict[ConcreteDefinition, tuple[Any, ...]] | None = None
    _witnesses: tuple[DefinitionOccurrence, ...] = ()
    _witness_complete: bool = False
    _containment: ContainmentContext | None = None
    _containment_witnesses: tuple[ContainmentOccurrence, ...] = ()
    _containment_witnesses_implicit: bool = False

    def __init__(
            self,
            repo,
            occurrences: Iterable[ContainmentOccurrence] | None = None,
            *,
            occurrence_factory: Callable[[], Iterable[ContainmentOccurrence]] | None = None,
            explanation: QueryExplanation | None = None,
            owner_replicas: Mapping[ConcreteDefinition, tuple[Any, ...]] | None = None,
            witnesses: Iterable[DefinitionOccurrence] = (),
            witness_complete: bool = False,
            containment: ContainmentContext | None = None,
            containment_witnesses: Iterable[ContainmentOccurrence] | None = None,
            bounded: bool = False):
        if occurrences is None and occurrence_factory is None:
            occurrences = ()
        if occurrences is not None and occurrence_factory is not None:
            raise ValueError("Provide occurrences or occurrence_factory, not both.")
        eager_occurrences = None if occurrences is None else tuple(occurrences)
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "_occurrences", None if eager_occurrences is None else _sort_occurrences(eager_occurrences))
        object.__setattr__(self, "_occurrence_factory", occurrence_factory)
        object.__setattr__(self, "explanation", explanation)
        object.__setattr__(self, "_owner_replicas", None if owner_replicas is None else dict(owner_replicas))
        object.__setattr__(self, "_witnesses", tuple(witnesses))
        object.__setattr__(self, "_witness_complete", bool(witness_complete))
        if containment is not None and bounded:
            containment = ContainmentContext(
                target_kind=containment.target_kind,
                edges=containment.edges,
                contains_ref=containment.contains_ref,
                source_scope=containment.source_scope,
                complete=containment.complete,
                bounded=True,
            )
        object.__setattr__(self, "_containment", containment)
        implicit_containment_witnesses = containment_witnesses is None
        if implicit_containment_witnesses:
            containment_witnesses = () if eager_occurrences is None else eager_occurrences
        containment_witnesses = tuple(containment_witnesses)
        if containment is not None:
            from .model import containment_target_kind

            invalid = next(
                (
                    item for item in containment_witnesses
                    if containment_target_kind(item.target) != containment.target_kind
                ),
                None,
            )
            if invalid is not None:
                raise ValueError("Containment occurrence evidence does not match its target kind.")
        object.__setattr__(self, "_containment_witnesses", _unique_witnesses(containment_witnesses))
        object.__setattr__(self, "_containment_witnesses_implicit", implicit_containment_witnesses)

    def __iter__(self) -> Iterator[ContainmentOccurrence]:
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

    def one(self) -> ContainmentOccurrence:
        occurrences = self._materialize()
        if len(occurrences) != 1:
            raise QueryCardinalityError(f"Expected exactly one occurrence, found {len(occurrences)}.")
        return occurrences[0]

    def one_or_none(self) -> ContainmentOccurrence | None:
        occurrences = self._materialize()
        if len(occurrences) > 1:
            raise QueryCardinalityError(f"Expected zero or one occurrence, found {len(occurrences)}.")
        return occurrences[0] if occurrences else None

    def first(self) -> ContainmentOccurrence | None:
        return next(iter(self), None)

    def definitions(self) -> DefinitionResultSet:
        if self._containment is not None and self._containment.target_kind != "definition":
            raise QueryDomainError(
                "definitions() is unavailable for exact reference containment targets."
            )
        occurrences = self._materialize()
        return DefinitionResultSet(
            self.repo,
            [occ.definition for occ in occurrences],
            materializable=False,
            domain="nested-definitions",
            explanation=self.explanation,
            replicas={},
            witnesses=(occ.definition for occ in self._witnesses),
            witness_complete=self._witness_complete,
            containment=self._containment,
            containment_witnesses=self._containment_witnesses,
            containment_carrier="target",
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
            witnesses=(occ.owner for occ in self._witnesses),
            witness_complete=self._witness_complete,
            containment=self._containment,
            containment_witnesses=self._containment_witnesses,
            containment_carrier="owner",
        )

    def object_refs(self):
        """Project exact ObjectRef terminals from containment occurrences.

        Raises:
            QueryDomainError: If this is not ObjectRef containment evidence.
        """

        if self._containment is None or self._containment.target_kind != "object_ref":
            actual = (
                "non-containment occurrences" if self._containment is None
                else self._containment.target_kind.replace("_", " ").title().replace(" ", "")
            )
            raise QueryDomainError(
                f"object_refs() requires exact ObjectRef containment evidence, got {actual}."
            )
        from .reference import ObjectRefResultSet

        occurrences = self._materialize()
        return ObjectRefResultSet(
            self.repo,
            (occ.target for occ in occurrences),
            occurrences,
            containment=self._containment,
            containment_witnesses=self._containment_witnesses,
            owner_replicas=self._require_owner_replicas(),
        )

    def state_refs(self):
        """Project exact StateRef terminals from containment occurrences.

        Raises:
            QueryDomainError: If this is not StateRef containment evidence.
        """

        if self._containment is None or self._containment.target_kind != "state_ref":
            actual = (
                "non-containment occurrences" if self._containment is None
                else self._containment.target_kind.replace("_", " ").title().replace(" ", "")
            )
            raise QueryDomainError(
                f"state_refs() requires exact StateRef containment evidence, got {actual}."
            )
        from .reference import StateRefResultSet

        occurrences = self._materialize()
        return StateRefResultSet(
            self.repo,
            (occ.target for occ in occurrences),
            occurrences,
            containment=self._containment,
            containment_witnesses=self._containment_witnesses,
            owner_replicas=self._require_owner_replicas(),
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
            witnesses=self._witnesses,
            witness_complete=self._witness_complete,
            containment=self._containment,
            containment_witnesses=self._containment_witnesses,
        )
        query = DefinitionQuery.from_source(
            self.repo,
            selector,
            domain="nested",
            universe=universe,
        )
        if self._containment is not None:
            return query.nested(
                edges=self._containment.edges,
                contains_ref=self._containment.contains_ref,
            )
        return query

    def union(self, other: "OccurrenceResultSet") -> "OccurrenceResultSet":
        self._check_compatible(other)
        containment = self._combined_containment(other)
        out = _unique_witnesses(self._materialize() + other._materialize())
        return OccurrenceResultSet(
            self.repo,
            out,
            explanation=self.explanation,
            owner_replicas=_merge_replica_maps(self._owner_replicas or {}, other._owner_replicas or {}),
            containment=containment,
            containment_witnesses=_unique_witnesses(
                (*self._containment_witnesses, *other._containment_witnesses),
            ) if containment is not None else (),
        )

    def intersection(self, other: "OccurrenceResultSet") -> "OccurrenceResultSet":
        self._check_compatible(other)
        containment = self._combined_containment(other)
        other_keys = {containment_witness_key(occ) for occ in other._materialize()}
        kept = tuple(
            occ for occ in self._materialize()
            if containment_witness_key(occ) in other_keys
        )
        return OccurrenceResultSet(
            self.repo,
            kept,
            explanation=self.explanation,
            owner_replicas=_merge_replica_maps(self._owner_replicas or {}, other._owner_replicas or {}),
            containment=containment,
            containment_witnesses=_witnesses_for_members(
                (*self._containment_witnesses, *other._containment_witnesses),
                "occurrence",
                kept,
            ) if containment is not None else (),
        )

    def _check_compatible(self, other: "OccurrenceResultSet") -> None:
        if self.repo is not other.repo:
            raise ValueError("Cannot combine result sets from different repos.")
        if (self._containment is None) != (other._containment is None):
            raise ValueError("Cannot combine containment and non-containment result sets.")
        if self._containment is not None and not self._containment.compatible_with(other._containment):
            raise ValueError("Cannot combine result sets with incompatible containment contexts.")

    def _combined_containment(self, other: "OccurrenceResultSet") -> ContainmentContext | None:
        """Merge compatible containment evidence metadata for a set operation."""

        if self._containment is None:
            return None
        return self._containment.combined_with(other._containment)

    def _require_owner_replicas(self) -> dict[ConcreteDefinition, tuple[Any, ...]]:
        if self._owner_replicas is None:
            raise QueryDomainError("Owner replica metadata was not captured for this occurrence result.")
        occurrences = self._materialize()
        missing = {occ.owner for occ in occurrences} - set(self._owner_replicas)
        if missing:
            raise QueryDomainError("Owner replica metadata is incomplete for this occurrence result.")
        return dict(self._owner_replicas)

    def _materialize(self) -> tuple[ContainmentOccurrence, ...]:
        if self._occurrences is None:
            object.__setattr__(self, "_occurrences", _sort_occurrences(self._occurrence_factory()))
            object.__setattr__(self, "_occurrence_factory", None)
        if self._containment_witnesses_implicit:
            object.__setattr__(self, "_containment_witnesses", _unique_witnesses(self._occurrences))
            object.__setattr__(self, "_containment_witnesses_implicit", False)
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
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_getitem"):
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
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_one"):
            if len(self) != 1:
                raise QueryCardinalityError(f"Expected exactly one object, found {len(self)}.")
            return next(iter(self._objects.values()))

    def one_or_none(self) -> Object | None:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_one_or_none"):
            if len(self) > 1:
                raise QueryCardinalityError(f"Expected zero or one object, found {len(self)}.")
            return next(iter(self._objects.values())) if self._objects else None

    def first(self) -> Object | None:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_first"):
            return next(iter(self._objects.values())) if self._objects else None

    def items(self):
        """Return a repeatable view whose iteration holds one admission lease."""

        return _GuardedObjectItemsView(self)

    def values(self):
        """Return a repeatable view whose iteration holds one admission lease."""

        return _GuardedObjectValuesView(self)

    def apply(self, func, *args, **kwargs) -> "ObjectResultSet":
        """Apply ``func`` to retained live objects under one materialization lease."""

        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_apply"):
            for obj in self._objects.values():
                func(obj, *args, **kwargs)
            return self


class _GuardedObjectItemsView(ItemsView):
    """Mapping items view that admits retained Objects for each full iteration."""

    def __iter__(self):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_items"):
            yield from self._mapping._objects.items()


class _GuardedObjectValuesView(ValuesView):
    """Mapping values view that admits retained Objects for each full iteration."""

    def __iter__(self):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_values"):
            yield from self._mapping._objects.values()
