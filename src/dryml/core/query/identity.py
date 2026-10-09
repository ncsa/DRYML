"""Private complete-identity and fixed-result primitives for Query V3.

These types deliberately do not change ``ConcreteDefinition`` equality, its
persisted codec, or the legacy query result surface. They give later V3 query
stages a detached identity and evidence boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
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


def _combined_requested_limit_state(
        first: int | None, first_conflict: bool,
        second: int | None, second_conflict: bool,
) -> tuple[int | None, bool]:
    """Combine visible prefix limits without erasing an earlier conflict."""

    if first_conflict or second_conflict:
        return None, True
    if first is None:
        return second, False
    if second is None or first == second:
        return first, False
    return None, True


class IdentitySet:
    """Fixed, complete-identity result members with detached source evidence.

    Construction eagerly snapshots only supplied members. Iteration, truth
    testing, and cardinality methods never consult a source or execute a query.
    ``bounded`` remains visible through fixed refinements and set algebra. An
    explicit ``requested_limit`` is retained while one originating limit remains
    unambiguous; algebra over different limits remains bounded without claiming a
    single combined prefix size.

    Args:
        members: Complete CDefs, ObjectRefs, StateRefs, or values paired with
            detached source evidence.
        bounded: Whether an explicit terminal bound limits completeness.
        requested_limit: The terminal prefix limit when ``bounded`` is true.

    Raises:
        TypeError: If boundedness or a member has an unsupported type.
        ValueError: If the requested limit is invalid or lacks boundedness.

    Side Effects:
        Eagerly snapshots only supplied values and evidence. Construction never
        reads Store authority or evaluates a query.
    """

    __slots__ = (
        "_entries", "bounded", "requested_limit",
        "_query_work", "_requested_limit_conflict",
    )

    def __init__(
        self,
        members: Iterable[
            ConcreteDefinition | ObjectRef | StateRef
            | tuple[ConcreteDefinition | ObjectRef | StateRef, SourceEvidence]
        ] = (),
        *,
        bounded: bool = False,
        requested_limit: int | None = None,
        _query_stats: tuple[int, int, int] = (0, 0, 0),
        _query_work: dict[object, tuple[int, int, int]] | None = None,
        _requested_limit_conflict: bool = False,
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
        self._requested_limit_conflict = _requested_limit_conflict
        self._query_work = dict(_query_work or {})
        if any(_query_stats):
            self._query_work[object()] = _query_stats

    @staticmethod
    def _member_and_evidence(member):
        if isinstance(member, tuple) and len(member) == 2 and isinstance(member[1], SourceEvidence):
            return member[0], IdentityEvidence(frozenset((member[1],)))
        return member, IdentityEvidence()

    @classmethod
    def _from_entries(
        cls, entries, *, bounded: bool, requested_limit: int | None = None,
        query_stats: tuple[int, int, int] = (0, 0, 0),
        query_work: dict[object, tuple[int, int, int]] | None = None,
        requested_limit_conflict: bool = False,
    ) -> "IdentitySet":
        result = cls(
            (), bounded=bounded, requested_limit=requested_limit,
            _query_stats=query_stats, _query_work=query_work,
            _requested_limit_conflict=requested_limit_conflict,
        )
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
        """Return detached source evidence for a retained complete identity.

        Args:
            member: One complete identity in this fixed result.

        Returns:
            The immutable evidence ledger captured for ``member``.

        Raises:
            KeyError: If the identity is not a member.

        Side Effects:
            None. No authority source is consulted.
        """

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

    def object_projection(self) -> "ObjectSelectorSet":
        """Project captured references to distinct recursive ObjectSelectors.

        Returns:
            A fixed ObjectSelectorSet. ConcreteDefinition members are omitted
            because they carry no realized ObjectIds to project. Evidence from
            references that collapse to one selector is merged.

        Side Effects:
            None. Projection reads no source and does not mutate authority.
        """

        from ..reference_values import ObjectRef, StateRef

        entries = {}
        for value, evidence in self._entries.values():
            if not isinstance(value, (ObjectRef, StateRef)):
                continue
            selector = value.object_projection()
            existing = entries.get(selector)
            entries[selector] = (
                selector,
                evidence if existing is None else existing[1].merged_with(evidence),
            )
        return ObjectSelectorSet._from_entries(
            entries,
            bounded=self.bounded,
            requested_limit=self.requested_limit,
        )

    def union(self, other: "IdentitySet") -> "IdentitySet":
        """Merge fixed identities, evidence, bounds, and prior work provenance.

        Args:
            other: Another detached identity result.

        Returns:
            A fixed set containing either input's identities. Work performed by
            shared producer terminals is counted once.

        Raises:
            None. Unsupported operands return ``NotImplemented``.

        Side Effects:
            None. No Store or query source is consulted.
        """

        if not isinstance(other, IdentitySet):
            return NotImplemented
        entries = dict(self._entries)
        for key, (value, evidence) in other._entries.items():
            existing = entries.get(key)
            entries[key] = (
                value if existing is None else existing[0],
                evidence if existing is None else existing[1].merged_with(evidence),
            )
        requested_limit, limit_conflict = _combined_requested_limit_state(
            self.requested_limit, self._requested_limit_conflict,
            other.requested_limit, other._requested_limit_conflict,
        )
        return self._from_entries(
            entries,
            bounded=self.bounded or other.bounded,
            requested_limit=requested_limit,
            requested_limit_conflict=limit_conflict,
            query_work={**self._query_work, **other._query_work},
        )

    def intersection(self, other: "IdentitySet") -> "IdentitySet":
        """Intersect fixed identities while retaining evidence and prior work.

        Args:
            other: Another detached identity result.

        Returns:
            A fixed set containing identities present in both inputs. Work
            performed by shared producer terminals is counted once.

        Raises:
            None. Unsupported operands return ``NotImplemented``.

        Side Effects:
            None. No Store or query source is consulted.
        """

        if not isinstance(other, IdentitySet):
            return NotImplemented
        entries = {
            key: (value, evidence.merged_with(other._entries[key][1]))
            for key, (value, evidence) in self._entries.items()
            if key in other._entries
        }
        requested_limit, limit_conflict = _combined_requested_limit_state(
            self.requested_limit, self._requested_limit_conflict,
            other.requested_limit, other._requested_limit_conflict,
        )
        return self._from_entries(
            entries,
            bounded=self.bounded or other.bounded,
            requested_limit=requested_limit,
            requested_limit_conflict=limit_conflict,
            query_work={**self._query_work, **other._query_work},
        )

    def diagnostic(self) -> QueryDiagnostic:
        """Return bounded structural diagnostics without sensitive values.

        Returns:
            A disclosure-capped :class:`QueryDiagnostic`. Keyset-page work
            counters remain exact and preserve producer provenance.

        Raises:
            None.

        Side Effects:
            None. The result and its sources remain detached.
        """

        candidate_rows, cdef_blobs, pages = (
            sum(values[index] for values in self._query_work.values())
            for index in range(3)
        )
        return QueryDiagnostic.for_fixed(
            "identity", len(self), self.bounded, self._source_count(),
            requested_limit=self.requested_limit,
            candidate_rows_read=candidate_rows,
            cdef_blobs_decoded=cdef_blobs,
            pages_fetched=pages,
        )

    def _source_count(self) -> int:
        return len({source for _, evidence in self._entries.values() for source in evidence.sources})

    def __repr__(self) -> str:
        return repr(self.diagnostic())


class ObjectSelectorSet:
    """Fixed distinct object-graph selectors with detached source evidence.

    Args:
        members: ObjectSelectors, optionally paired with SourceEvidence.
        bounded: Whether the source identity result was explicitly bounded.
        requested_limit: Source identity prefix bound when present.

    ObjectSelectorSet is a projected result domain rather than Store authority;
    it cannot be turned back into an IdentityQuery.
    """

    __slots__ = ("_entries", "bounded", "requested_limit")

    def __init__(
        self,
        members=(),
        *,
        bounded: bool = False,
        requested_limit: int | None = None,
    ):
        from ..selector import ObjectSelector

        if type(bounded) is not bool:
            raise TypeError("ObjectSelectorSet bounded must be an exact bool.")
        if requested_limit is not None and (
                type(requested_limit) is not int or requested_limit < 0
        ):
            raise ValueError(
                "ObjectSelectorSet requested_limit must be a non-negative exact int or None."
            )
        if requested_limit is not None and not bounded:
            raise ValueError("ObjectSelectorSet requested_limit requires bounded=True.")
        entries = {}
        for member in members:
            value, evidence = IdentitySet._member_and_evidence(member)
            if not isinstance(value, ObjectSelector):
                raise TypeError("ObjectSelectorSet members must be ObjectSelector values.")
            existing = entries.get(value)
            entries[value] = (
                value,
                evidence if existing is None else existing[1].merged_with(evidence),
            )
        self._entries = entries
        self.bounded = bounded
        self.requested_limit = requested_limit

    @classmethod
    def _from_entries(
        cls, entries, *, bounded: bool, requested_limit: int | None = None,
    ) -> "ObjectSelectorSet":
        result = cls((), bounded=bounded, requested_limit=requested_limit)
        result._entries = entries
        return result

    def __iter__(self):
        yield from (
            self._entries[key][0]
            for key in sorted(
                self._entries,
                key=lambda selector: (
                    selector.digest(), selector.reference.digest(),
                ),
            )
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, selector: object) -> bool:
        return selector in self._entries

    def count(self) -> int:
        """Return the number of distinct projected object graphs."""

        return len(self)

    def collect(self) -> "ObjectSelectorSet":
        """Return this already fixed selector collection."""

        return self

    def exists(self) -> bool:
        """Return whether this collection contains a selector."""

        return bool(self)

    def one(self):
        """Return one selector or raise QueryCardinalityError otherwise."""

        if len(self) != 1:
            raise QueryCardinalityError(
                f"Expected exactly one object selector, found {len(self)}."
            )
        return next(iter(self))

    def one_or_none(self):
        """Return zero or one selector while rejecting ambiguity."""

        if len(self) > 1:
            raise QueryCardinalityError(
                f"Expected zero or one object selector, found {len(self)}."
            )
        return next(iter(self), None)

    def evidence_for(self, selector):
        """Return merged detached evidence for one projected selector."""

        return self._entries[selector][1]

    def sources(self, selector) -> frozenset[SourceEvidence]:
        """Return detached sources contributing to one projected selector."""

        return self.evidence_for(selector).sources

    def __repr__(self) -> str:
        return (
            f"ObjectSelectorSet(count={len(self)}, bounded={self.bounded}, "
            f"requested_limit={self.requested_limit!r})"
        )


class OccurrenceSet:
    """Fixed typed occurrences with detached source evidence and visible bounds.

    Args:
        members: Complete occurrences, optionally paired with source evidence.
        bounded: Whether an explicit output cap limits completeness.
        requested_limit: The terminal prefix limit when ``bounded`` is true.

    Raises:
        TypeError: If a member or boundedness flag has an unsupported type.
        ValueError: If the requested limit is invalid or lacks boundedness.

    Side Effects:
        Eagerly snapshots supplied values only; no authority source is read.
    """

    __slots__ = (
        "_entries", "bounded", "requested_limit", "_requested_limit_conflict",
    )

    def __init__(
        self,
        members: Iterable[Occurrence | tuple[Occurrence, SourceEvidence]] = (),
        *,
        bounded: bool = False,
        requested_limit: int | None = None,
        _requested_limit_conflict: bool = False,
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
        self._requested_limit_conflict = _requested_limit_conflict

    @classmethod
    def _from_entries(
        cls, entries, *, bounded: bool, requested_limit: int | None = None,
        requested_limit_conflict: bool = False,
    ) -> "OccurrenceSet":
        result = cls(
            (), bounded=bounded, requested_limit=requested_limit,
            _requested_limit_conflict=requested_limit_conflict,
        )
        result._entries = entries
        return result

    def __iter__(self) -> Iterator[Occurrence]:
        for key in _ordered_occurrences(self._entries):
            yield self._entries[key][0]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, occurrence: object) -> bool:
        if not isinstance(occurrence, Occurrence):
            return False
        return occurrence.key in self._entries

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
        """Return detached source evidence for one retained occurrence.

        Args:
            occurrence: A complete occurrence in this fixed result.

        Returns:
            The immutable evidence ledger captured for ``occurrence``.

        Raises:
            KeyError: If the occurrence is not a member.

        Side Effects:
            None. No authority source is consulted.
        """

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
        """Merge fixed occurrence paths, evidence, and visible bounds.

        Args:
            other: Another detached occurrence result.

        Returns:
            A fixed result containing occurrences from either input.

        Raises:
            None. Unsupported operands return ``NotImplemented``.

        Side Effects:
            None. No source query executes.
        """

        if not isinstance(other, OccurrenceSet):
            return NotImplemented
        entries = dict(self._entries)
        for key, (value, evidence) in other._entries.items():
            existing = entries.get(key)
            entries[key] = (
                value if existing is None else existing[0],
                evidence if existing is None else existing[1].merged_with(evidence),
            )
        requested_limit, limit_conflict = _combined_requested_limit_state(
            self.requested_limit, self._requested_limit_conflict,
            other.requested_limit, other._requested_limit_conflict,
        )
        return self._from_entries(
            entries,
            bounded=self.bounded or other.bounded,
            requested_limit=requested_limit,
            requested_limit_conflict=limit_conflict,
        )

    def intersection(self, other: "OccurrenceSet") -> "OccurrenceSet":
        """Intersect fixed occurrence paths and retain both evidence ledgers.

        Args:
            other: Another detached occurrence result.

        Returns:
            A fixed result containing occurrences present in both inputs.

        Raises:
            None. Unsupported operands return ``NotImplemented``.

        Side Effects:
            None. No source query executes.
        """

        if not isinstance(other, OccurrenceSet):
            return NotImplemented
        entries = {
            key: (value, evidence.merged_with(other._entries[key][1]))
            for key, (value, evidence) in self._entries.items()
            if key in other._entries
        }
        requested_limit, limit_conflict = _combined_requested_limit_state(
            self.requested_limit, self._requested_limit_conflict,
            other.requested_limit, other._requested_limit_conflict,
        )
        return self._from_entries(
            entries,
            bounded=self.bounded or other.bounded,
            requested_limit=requested_limit,
            requested_limit_conflict=limit_conflict,
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
