"""Private complete-identity and fixed-result primitives for Query V3.

These types deliberately do not change ``ConcreteDefinition`` equality, its
persisted codec, or the legacy query result surface. They give later V3 query
stages a detached identity and evidence boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
import hashlib
from typing import Any, Literal

from ..definition import ConcreteDefinition
from ..reference_values import ObjectRef, StateRef
from ..utils.graph.path import GraphPath, graph_path_sort_key, normalize_path
from .model import QueryCardinalityError, QueryDiagnostic


IdentityKind = Literal["cdef", "object_ref", "state_ref"]


def _digest_source(value: object) -> str:
    """Return an opaque source correlation token without rendering ``value``."""

    if isinstance(value, str):
        payload = value.encode("utf-8", "surrogatepass")
    else:
        cls = type(value)
        payload = f"{cls.__module__}.{cls.__qualname__}:{id(value)}".encode("utf-8")
    return hashlib.sha256(b"dryml-query-source-v3\x00" + payload).hexdigest()[:16]


def _source_material(value: object) -> str:
    """Return non-rendered source material before it is converted to a token."""

    if isinstance(value, str):
        return value
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}:{id(value)}"


@dataclass(frozen=True, slots=True, repr=False)
class SourceEvidence:
    """Detached provenance for one source that supplied a fixed member.

    ``source_id`` is evidence only, never part of query membership or identity.
    Its representation is an opaque correlation token so framework diagnostics
    cannot disclose source names, paths, or arbitrary source values.
    """

    source_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str):
            raise TypeError("Source evidence identifiers must be strings.")
        object.__setattr__(self, "source_id", _digest_source(self.source_id))

    @classmethod
    def from_source(cls, source: object) -> "SourceEvidence":
        """Capture detached source provenance without retaining a live handle."""

        return cls(_source_material(source))

    @property
    def diagnostic_id(self) -> str:
        """Return the bounded opaque identifier suitable for diagnostics."""

        return self.source_id

    def __repr__(self) -> str:
        return f"SourceEvidence({self.diagnostic_id})"


@dataclass(frozen=True, slots=True)
class IdentityEvidence:
    """Immutable source evidence attached to a member outside its identity."""

    sources: frozenset[SourceEvidence] = frozenset()

    def merged_with(self, other: "IdentityEvidence") -> "IdentityEvidence":
        """Return evidence that retains every contributing detached source."""

        return IdentityEvidence(self.sources | other.sources)


def _identity_kind(value: object) -> IdentityKind:
    if isinstance(value, ConcreteDefinition):
        return "cdef"
    if isinstance(value, ObjectRef):
        return "object_ref"
    if isinstance(value, StateRef):
        return "state_ref"
    raise TypeError(
        "Query V3 identity members must be ConcreteDefinition, ObjectRef, or StateRef."
    )


def _identity_digest(value: ConcreteDefinition | ObjectRef | StateRef) -> str:
    return value.graph_hash() if isinstance(value, ConcreteDefinition) else value.digest()


def _canonical_tie(value: ConcreteDefinition | ObjectRef | StateRef) -> str:
    """Return a token-free stable tie-breaker independent of the fast digest."""

    if isinstance(value, ConcreteDefinition):
        from ..cdef_codec import cdef_graph_hash

        return cdef_graph_hash(value)
    if isinstance(value, ObjectRef):
        return value.definition.graph_hash() + ":" + value.digest()
    return value.object.definition.graph_hash() + ":" + value.digest()


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class IdentityKey:
    """Collision-checked, kind-tagged key for one complete query identity.

    Digest equality only selects a verification bucket. CDefs then use rooted
    graph equality, while references use their existing complete equality.
    """

    kind: IdentityKind
    digest: str
    _value: ConcreteDefinition | ObjectRef | StateRef

    @classmethod
    def for_value(cls, value: ConcreteDefinition | ObjectRef | StateRef) -> "IdentityKey":
        """Construct a complete key without changing ordinary value identity."""

        return cls(_identity_kind(value), _identity_digest(value), value)

    @property
    def sort_key(self) -> tuple[str, str, str]:
        """Return a deterministic private ordering key for fixed collections."""

        return self.kind, self.digest, _canonical_tie(self._value)

    def __hash__(self) -> int:
        return hash((self.kind, self.digest))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IdentityKey):
            return NotImplemented
        if self.kind != other.kind or self.digest != other.digest:
            return False
        if self.kind == "cdef":
            return self._value.graph_equal(other._value)
        return self._value == other._value

    def __repr__(self) -> str:
        return f"IdentityKey(kind={self.kind!r}, digest={self.digest[:16]!r})"


def identity_key(value: ConcreteDefinition | ObjectRef | StateRef) -> IdentityKey:
    """Return the complete private Query V3 key for one supported identity."""

    return IdentityKey.for_value(value)


def _ordered_keys(keys: Iterable[IdentityKey]) -> Iterator[IdentityKey]:
    """Order by digest, encoding a canonical tie only in collision buckets."""

    buckets: dict[tuple[str, str], list[IdentityKey]] = {}
    for key in keys:
        buckets.setdefault((key.kind, key.digest), []).append(key)
    for bucket_id in sorted(buckets):
        bucket = buckets[bucket_id]
        yield from (bucket if len(bucket) == 1 else sorted(bucket, key=lambda key: key.sort_key))


@dataclass(frozen=True, slots=True, repr=False)
class Occurrence:
    """One typed root-to-target occurrence retained independently of evidence.

    Args:
        owner: Complete CDef, ObjectRef, or StateRef occurrence root.
        path: Immutable typed path from ``owner`` to ``target``.
        target: Complete CDef, ObjectRef, or StateRef terminal.

    Source provenance and traversal-hop details are evidence owned by
    :class:`OccurrenceSet`, not occurrence membership.
    """

    owner: ConcreteDefinition | ObjectRef | StateRef
    path: GraphPath | "RelationshipPath"
    target: ConcreteDefinition | ObjectRef | StateRef

    def __post_init__(self) -> None:
        from .relationships import RelationshipPath

        _identity_kind(self.owner)
        _identity_kind(self.target)
        if not isinstance(self.path, (GraphPath, RelationshipPath)):
            raise TypeError("Occurrence path must be a GraphPath or RelationshipPath.")
        object.__setattr__(
            self, "path",
            normalize_path(self.path) if isinstance(self.path, GraphPath) else self.path,
        )

    @property
    def key(self) -> "OccurrenceKey":
        """Return the complete typed occurrence membership key."""

        return OccurrenceKey(identity_key(self.owner), self.path, identity_key(self.target))

    def __repr__(self) -> str:
        return "Occurrence(owner=<identity>, path=<typed-path>, target=<identity>)"


@dataclass(frozen=True, slots=True, repr=False)
class OccurrenceKey:
    """Typed occurrence identity preserving owner, path, and target distinctions."""

    owner: IdentityKey
    path: GraphPath | "RelationshipPath"
    target: IdentityKey

    @property
    def sort_key(self) -> tuple[tuple[str, str, str], object, tuple[str, str, str]]:
        """Return deterministic ordering without rendering occurrence values."""

        path = _occurrence_path_key(self.path)
        return self.owner.sort_key, path, self.target.sort_key

    def __repr__(self) -> str:
        return "OccurrenceKey(owner=<identity>, path=<typed-path>, target=<identity>)"


def _ordered_occurrences(keys: Iterable[OccurrenceKey]) -> Iterator[OccurrenceKey]:
    """Order occurrence digests and paths before resolving genuine ties."""

    buckets: dict[tuple[Any, ...], list[OccurrenceKey]] = {}
    for key in keys:
        buckets.setdefault(
            (key.owner.kind, key.owner.digest, _occurrence_path_key(key.path),
             key.target.kind, key.target.digest), []
        ).append(key)
    for bucket_id in sorted(buckets):
        bucket = buckets[bucket_id]
        yield from (bucket if len(bucket) == 1 else sorted(bucket, key=lambda key: key.sort_key))


def _occurrence_path_key(path: GraphPath | "RelationshipPath") -> object:
    """Return a typed path bucket key without erasing association hops."""

    return (
        ("graph", graph_path_sort_key(path))
        if isinstance(path, GraphPath)
        else ("relationship", path.sort_key)
    )


class IdentitySet:
    """Fixed, complete-identity result members with detached source evidence.

    Construction eagerly snapshots only supplied members. Iteration, truth
    testing, and cardinality methods never consult a source or execute a query.
    ``bounded`` and an optional explicit ``requested_limit`` remain visible
    through fixed refinements and set algebra.
    """

    __slots__ = ("_entries", "bounded", "requested_limit")

    def __init__(
        self,
        members: Iterable[
            ConcreteDefinition | ObjectRef | StateRef
            | tuple[ConcreteDefinition | ObjectRef | StateRef, SourceEvidence]
        ] = (),
        *,
        bounded: bool = False,
        requested_limit: int | None = None,
    ):
        if type(bounded) is not bool:
            raise TypeError("IdentitySet bounded must be an exact bool.")
        if requested_limit is not None and (
                type(requested_limit) is not int or requested_limit < 0
        ):
            raise ValueError("IdentitySet requested_limit must be a non-negative exact int or None.")
        if requested_limit is not None and not bounded:
            raise ValueError("IdentitySet requested_limit requires bounded=True.")
        entries: dict[
            IdentityKey,
            tuple[ConcreteDefinition | ObjectRef | StateRef, IdentityEvidence],
        ] = {}
        for member in members:
            value, evidence = self._member_and_evidence(member)
            key = identity_key(value)
            existing = entries.get(key)
            entries[key] = (
                value if existing is None else existing[0],
                evidence if existing is None else existing[1].merged_with(evidence),
            )
        self._entries = entries
        self.bounded = bounded
        self.requested_limit = requested_limit

    @staticmethod
    def _member_and_evidence(member):
        if isinstance(member, tuple) and len(member) == 2 and isinstance(member[1], SourceEvidence):
            return member[0], IdentityEvidence(frozenset((member[1],)))
        return member, IdentityEvidence()

    @classmethod
    def _from_entries(
        cls, entries, *, bounded: bool, requested_limit: int | None = None,
    ) -> "IdentitySet":
        result = cls((), bounded=bounded, requested_limit=requested_limit)
        result._entries = entries
        return result

    def __iter__(self) -> Iterator[ConcreteDefinition | ObjectRef | StateRef]:
        for key in _ordered_keys(self._entries):
            yield self._entries[key][0]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, member: object) -> bool:
        try:
            return identity_key(member) in self._entries
        except TypeError:
            return False

    def count(self) -> int:
        """Return the fixed distinct-member cardinality without source access."""

        return len(self)

    def collect(self) -> "IdentitySet":
        """Return this already fixed collection without evaluating a source."""

        return self

    def exists(self) -> bool:
        """Return whether this fixed result has a member without source access."""

        return bool(self)

    def one(self):
        """Return one member or raise ``QueryCardinalityError`` otherwise."""

        if len(self) != 1:
            raise QueryCardinalityError(f"Expected exactly one result, found {len(self)}.")
        return next(iter(self))

    def one_or_none(self):
        """Return one member, ``None``, or raise on ambiguous cardinality."""

        if len(self) > 1:
            raise QueryCardinalityError(f"Expected zero or one result, found {len(self)}.")
        return next(iter(self), None)

    def evidence_for(self, member) -> IdentityEvidence:
        """Return detached source evidence for a retained complete identity."""

        return self._entries[identity_key(member)][1]

    def sources(self, member) -> frozenset[SourceEvidence]:
        """Return captured contributors without consulting current Store authority.

        Args:
            member: One complete identity in this fixed result.

        Returns:
            Detached source tokens that contributed the member, possibly empty.

        Raises:
            KeyError: If the identity is not a member.

        Side Effects:
            None. Contribution is not independent stored-membership proof.
        """

        return self.evidence_for(member).sources

    def query(self) -> "IdentityQuery":
        """Start a V3 identity query limited to this fixed membership.

        Returns:
            An :class:`IdentityQuery` over exactly this collection's captured
            members and bounds, without an implicit external authority scope.

        Raises:
            None.

        Side Effects:
            Does not read a source. Restrictions remain unevaluated until an
            explicit terminal is called.
        """

        from .query import IdentityQuery

        return IdentityQuery.from_set(self)

    def refine(self, predicate: Callable[[Any], bool]) -> "IdentitySet":
        """Return the fixed subset accepted by ``predicate``."""

        return IdentitySetQuery(self, (predicate,)).collect()

    def union(self, other: "IdentitySet") -> "IdentitySet":
        """Merge fixed complete identities and detached evidence without queries."""

        if not isinstance(other, IdentitySet):
            return NotImplemented
        entries = dict(self._entries)
        for key, (value, evidence) in other._entries.items():
            existing = entries.get(key)
            entries[key] = (
                value if existing is None else existing[0],
                evidence if existing is None else existing[1].merged_with(evidence),
            )
        return self._from_entries(
            entries,
            bounded=self.bounded or other.bounded,
        )

    def intersection(self, other: "IdentitySet") -> "IdentitySet":
        """Intersect fixed complete identities and retain both evidence ledgers."""

        if not isinstance(other, IdentitySet):
            return NotImplemented
        entries = {
            key: (value, evidence.merged_with(other._entries[key][1]))
            for key, (value, evidence) in self._entries.items()
            if key in other._entries
        }
        return self._from_entries(
            entries,
            bounded=self.bounded or other.bounded,
        )

    def diagnostic(self) -> QueryDiagnostic:
        """Return bounded structural diagnostics without member or source values."""

        return QueryDiagnostic.for_fixed(
            "identity", len(self), self.bounded, self._source_count(),
            requested_limit=self.requested_limit,
        )

    def _source_count(self) -> int:
        return len({source for _, evidence in self._entries.values() for source in evidence.sources})

    def __repr__(self) -> str:
        return repr(self.diagnostic())


class OccurrenceSet:
    """Fixed typed occurrences with detached source evidence and visible bounds."""

    __slots__ = ("_entries", "bounded", "requested_limit")

    def __init__(
        self,
        members: Iterable[Occurrence | tuple[Occurrence, SourceEvidence]] = (),
        *,
        bounded: bool = False,
        requested_limit: int | None = None,
    ):
        if type(bounded) is not bool:
            raise TypeError("OccurrenceSet bounded must be an exact bool.")
        if requested_limit is not None and (
                type(requested_limit) is not int or requested_limit < 0
        ):
            raise ValueError("OccurrenceSet requested_limit must be a non-negative exact int or None.")
        if requested_limit is not None and not bounded:
            raise ValueError("OccurrenceSet requested_limit requires bounded=True.")
        entries: dict[OccurrenceKey, tuple[Occurrence, IdentityEvidence]] = {}
        for member in members:
            occurrence, evidence = IdentitySet._member_and_evidence(member)
            if not isinstance(occurrence, Occurrence):
                raise TypeError("OccurrenceSet members must be Occurrence values.")
            existing = entries.get(occurrence.key)
            entries[occurrence.key] = (
                occurrence if existing is None else existing[0],
                evidence if existing is None else existing[1].merged_with(evidence),
            )
        self._entries = entries
        self.bounded = bounded
        self.requested_limit = requested_limit

    @classmethod
    def _from_entries(
        cls, entries, *, bounded: bool, requested_limit: int | None = None,
    ) -> "OccurrenceSet":
        result = cls((), bounded=bounded, requested_limit=requested_limit)
        result._entries = entries
        return result

    def __iter__(self) -> Iterator[Occurrence]:
        for key in _ordered_occurrences(self._entries):
            yield self._entries[key][0]

    def __len__(self) -> int:
        return len(self._entries)

    def count(self) -> int:
        """Return the fixed distinct-occurrence cardinality without source access."""

        return len(self)

    def collect(self) -> "OccurrenceSet":
        """Return this already fixed collection without evaluating a source."""

        return self

    def exists(self) -> bool:
        """Return whether this fixed occurrence result has a member."""

        return bool(self)

    def one(self) -> Occurrence:
        """Return one occurrence or raise ``QueryCardinalityError`` otherwise."""

        if len(self) != 1:
            raise QueryCardinalityError(f"Expected exactly one occurrence, found {len(self)}.")
        return next(iter(self))

    def one_or_none(self) -> Occurrence | None:
        """Return one occurrence, ``None``, or raise on ambiguity."""

        if len(self) > 1:
            raise QueryCardinalityError(f"Expected zero or one occurrence, found {len(self)}.")
        return next(iter(self), None)

    def evidence_for(self, occurrence: Occurrence) -> IdentityEvidence:
        """Return detached source evidence for one retained occurrence."""

        return self._entries[occurrence.key][1]

    def sources(self, occurrence: Occurrence) -> frozenset[SourceEvidence]:
        """Return detached contributors for a captured occurrence.

        Args:
            occurrence: A member with its complete root and typed path.

        Returns:
            Detached contributing source tokens, possibly empty.

        Raises:
            KeyError: If the occurrence is not in this fixed result.

        Side Effects:
            None. No source is reopened or granted authority.
        """

        return self.evidence_for(occurrence).sources

    def query(self) -> "OccurrenceQuery":
        """Start a V3 occurrence query over these captured paths alone.

        Returns:
            An :class:`OccurrenceQuery` over exactly this collection's
            captured occurrence paths and bounds.

        Raises:
            None.

        Side Effects:
            Does not read a source. Path and target restrictions remain
            unevaluated until an explicit terminal is called.
        """

        from .query import OccurrenceQuery

        return OccurrenceQuery.from_set(self)

    def union(self, other: "OccurrenceSet") -> "OccurrenceSet":
        """Merge fixed occurrence paths and evidence without source execution."""

        if not isinstance(other, OccurrenceSet):
            return NotImplemented
        entries = dict(self._entries)
        for key, (value, evidence) in other._entries.items():
            existing = entries.get(key)
            entries[key] = (
                value if existing is None else existing[0],
                evidence if existing is None else existing[1].merged_with(evidence),
            )
        return self._from_entries(
            entries,
            bounded=self.bounded or other.bounded,
        )

    def intersection(self, other: "OccurrenceSet") -> "OccurrenceSet":
        """Intersect fixed occurrence paths and retain both evidence ledgers."""

        if not isinstance(other, OccurrenceSet):
            return NotImplemented
        entries = {
            key: (value, evidence.merged_with(other._entries[key][1]))
            for key, (value, evidence) in self._entries.items()
            if key in other._entries
        }
        return self._from_entries(
            entries,
            bounded=self.bounded or other.bounded,
        )

    def diagnostic(self) -> QueryDiagnostic:
        """Return bounded structural diagnostics without occurrence values."""

        return QueryDiagnostic.for_fixed(
            "occurrence", len(self), self.bounded, self._source_count(),
            requested_limit=self.requested_limit,
        )

    def _source_count(self) -> int:
        return len({source for _, evidence in self._entries.values() for source in evidence.sources})

    def __repr__(self) -> str:
        return repr(self.diagnostic())


class _FixedSetQuery:
    """Common immutable refinement shell that rejects implicit evaluation."""

    __slots__ = ("_source", "_predicates")

    def __init__(self, source, predicates):
        self._source = source
        self._predicates = predicates

    def where(self, predicate):
        """Append one fixed-member predicate without evaluating the result."""

        if not callable(predicate):
            raise TypeError("Fixed result predicates must be callable.")
        return type(self)(self._source, (*self._predicates, predicate))

    def __bool__(self) -> bool:
        raise TypeError("Fixed result queries require an explicit terminal.")

    def exists(self) -> bool:
        """Return whether an explicit fixed refinement has any member."""

        return self.collect().exists()

    def one(self):
        """Return one refined member or raise the normal cardinality error."""

        return self.collect().one()

    def one_or_none(self):
        """Return one refined member, ``None``, or raise on ambiguity."""

        return self.collect().one_or_none()


class IdentitySetQuery(_FixedSetQuery):
    """Unevaluated refinement over an ``IdentitySet`` captured membership."""

    def collect(self) -> IdentitySet:
        """Evaluate only fixed members and return another detached identity set."""

        entries = {
            key: (value, evidence)
            for key, (value, evidence) in self._source._entries.items()
            if all(predicate(value) for predicate in self._predicates)
        }
        return IdentitySet._from_entries(
            entries,
            bounded=self._source.bounded,
            requested_limit=self._source.requested_limit,
        )

    def count(self) -> int:
        """Return the refined fixed-member count through an explicit terminal."""

        return self.collect().count()


class OccurrenceSetQuery(_FixedSetQuery):
    """Unevaluated occurrence-aware refinement over fixed occurrence paths."""

    def collect(self) -> OccurrenceSet:
        """Evaluate only fixed occurrences and return a detached occurrence set."""

        entries = {
            key: (value, evidence)
            for key, (value, evidence) in self._source._entries.items()
            if all(predicate(value) for predicate in self._predicates)
        }
        return OccurrenceSet._from_entries(
            entries,
            bounded=self._source.bounded,
            requested_limit=self._source.requested_limit,
        )

    def count(self) -> int:
        """Return the refined fixed-occurrence count through an explicit terminal."""

        return self.collect().count()
