"""Detached Store authority facts for Query V3 producer capture.

Knowledge inventory, stored membership, and metadata-holder eligibility are
kept separate because existing Store authority assigns them different meanings.
The module owns no Repo configuration and never reads a Store after capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from ..definition import ConcreteDefinition
from ..reference_values import ObjectRef, StateRef
from ..store.records import (
    DeclarationRecord, DefinitionRecord, ObjectAliasRecord, StateAliasRecord,
    StateRefRecord, StoredRootRecord,
)
from ..store.store import StoreAuthorityError
from .identity import IdentityKey, IdentitySet, SourceEvidence, identity_key
from .model import QueryError
from .relationships import iter_direct_relationships


IdentityValue = ConcreteDefinition | ObjectRef | StateRef


@dataclass(frozen=True, slots=True)
class CapturedStoreFacts:
    """One complete detached Store authority cut and its derived identity facts.

    Record tuples supply complete knowledge, independent membership, metadata
    eligibility, and validated authoritative ObjectId projections. A fresh
    ``capture_token`` identifies every replacement, including metadata-only
    enrichment; it is unrelated to a derived SQLite generation. Knowledge is
    derived once per fact table without Store reads or authority writes.
    """

    definitions: tuple[DefinitionRecord, ...]
    stored_roots: tuple[StoredRootRecord, ...]
    declarations: tuple[DeclarationRecord, ...]
    state_refs: tuple[StateRefRecord, ...]
    object_aliases: tuple[ObjectAliasRecord, ...]
    state_aliases: tuple[StateAliasRecord, ...]
    main_definition: ConcreteDefinition | None
    source: SourceEvidence
    metadata: tuple[tuple[IdentityKey, str, object], ...] = ()
    metadata_scopes: frozenset[str] = frozenset()
    capture_token: object = field(init=False, default_factory=object, compare=False, repr=False)
    _reference_projections: dict = field(default_factory=dict, compare=False, repr=False)
    _knowledge: IdentitySet | None = field(init=False, default=None, compare=False, repr=False)
    _metadata_index: dict[tuple[IdentityKey, str], object] = field(
        init=False, compare=False, repr=False,
    )
    _stored_cdefs: frozenset[IdentityKey] = field(init=False, compare=False, repr=False)
    _stored_objects: frozenset[IdentityKey] = field(init=False, compare=False, repr=False)
    _stored_states: frozenset[IdentityKey] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_metadata_index",
            {(key, scope): value for key, scope, value in self.metadata},
        )
        object.__setattr__(
            self, "_stored_cdefs",
            frozenset(identity_key(value) for value in self._authoritative_root_definitions()),
        )
        object.__setattr__(
            self, "_stored_objects",
            frozenset(identity_key(record.object_ref) for record in self.declarations) | frozenset(
                identity_key(record.state_ref.object) for record in self.state_refs
            ),
        )
        object.__setattr__(
            self, "_stored_states",
            frozenset(identity_key(record.state_ref) for record in self.state_refs),
        )

    @classmethod
    def build(
        cls,
        *,
        definitions: Iterable[DefinitionRecord],
        stored_roots: Iterable[StoredRootRecord],
        declarations: Iterable[DeclarationRecord],
        state_refs: Iterable[StateRefRecord],
        object_aliases: Iterable[ObjectAliasRecord],
        state_aliases: Iterable[StateAliasRecord],
        main_definition: ConcreteDefinition | None,
        source: SourceEvidence,
        metadata: Iterable[tuple[IdentityKey, str, object]] = (),
        metadata_scopes: frozenset[str] = frozenset(),
    ) -> "CapturedStoreFacts":
        """Validate cross-record authority and return a detached fact table.

        Record iterables contain validated definitions, markers, declarations,
        snapshots, and aliases. ``main_definition`` is the fenced, resolved main
        target; ``source`` is detached evidence. Optional ``metadata`` triples and
        ``metadata_scopes`` describe fields read in the same cut. Duplicate
        digests, missing or incompatible targets, and authoritative ObjectId
        conflicts raise StoreAuthorityError. No Store access or writes occur.
        """

        definitions = tuple(definitions)
        by_digest = {record.digest: record for record in definitions}
        if len(by_digest) != len(definitions):
            raise StoreAuthorityError("Definition authority has duplicate record digests.")
        stored_roots = tuple(stored_roots)
        for root in stored_roots:
            if root.definition_digest not in by_digest:
                raise StoreAuthorityError("Stored-root authority targets a missing definition record.")
        state_refs = tuple(state_refs)
        state_by_digest = {record.digest: record for record in state_refs}
        if len(state_by_digest) != len(state_refs):
            raise StoreAuthorityError("StateRef authority has duplicate record digests.")
        state_aliases = tuple(state_aliases)
        for alias in state_aliases:
            record = state_by_digest.get(alias.state_ref_digest)
            if record is None or record.state_ref.object != alias.object_ref:
                raise StoreAuthorityError("State alias targets missing or incompatible StateRef authority.")
        declarations = tuple(declarations)
        object_aliases = tuple(object_aliases)
        projections = _validate_object_reference_roots((
            *(record.object_ref for record in declarations),
            *(record.state_ref.object for record in state_refs),
            *(record.object_ref for record in object_aliases),
            *(record.object_ref for record in state_aliases),
        ))
        return cls(
            definitions, stored_roots, declarations, state_refs,
            object_aliases, state_aliases, main_definition, source,
            tuple(metadata), metadata_scopes, projections,
        )

    def with_metadata(
        self, metadata: Iterable[tuple[IdentityKey, str, object]], *, scopes: frozenset[str],
    ) -> "CapturedStoreFacts":
        """Enrich this cut with detached metadata and a fresh capture token.

        ``metadata`` contains captured identity/scope/value triples; ``scopes``
        is the complete requested scope set. Returns a new table sharing already
        validated identity structure. The caller owns fenced reads and failures;
        this method performs no Store access and never revalidates reference roots.
        """

        enriched = replace(self, metadata=tuple(metadata), metadata_scopes=scopes)
        object.__setattr__(enriched, "_knowledge", self._knowledge)
        return enriched

    def knowledge(self) -> IdentitySet:
        """Return cached full represented/derivable knowledge as an IdentitySet.

        The first call expands detached direct relationships; later calls reuse
        that structure and source evidence. It does not read Store records,
        promote embedded identities to stored authority, or change authority.
        """

        if self._knowledge is not None:
            return self._knowledge
        roots: list[IdentityValue] = [record.definition for record in self.definitions]
        roots.extend(record.object_ref for record in self.declarations)
        roots.extend(record.state_ref for record in self.state_refs)
        roots.extend(record.object_ref for record in self.object_aliases)
        roots.extend(record.object_ref for record in self.state_aliases)
        if self.main_definition is not None:
            roots.append(self.main_definition)
        known = IdentitySet((value, self.source) for value in _derive_identities(roots))
        object.__setattr__(self, "_knowledge", known)
        return known

    def is_stored(self, value: IdentityValue) -> bool:
        """Return kind-specific independent stored authority from captured records."""

        if isinstance(value, ConcreteDefinition):
            return identity_key(value) in self._stored_cdefs
        if isinstance(value, ObjectRef):
            return identity_key(value) in self._stored_objects
        if isinstance(value, StateRef):
            return identity_key(value) in self._stored_states
        raise TypeError("Stored membership requires a Query V3 identity value.")

    def holds_metadata(self, value: ObjectRef | StateRef) -> bool:
        """Return existing metadata-holder eligibility without widening storage proof."""

        if isinstance(value, StateRef):
            return self.is_stored(value)
        if not isinstance(value, ObjectRef):
            raise TypeError("Metadata holder checks require an ObjectRef or StateRef.")
        if self.is_stored(value):
            return True
        for record in self.state_refs:
            for path in record.state_ref.object.objects:
                if record.state_ref.object.at(path) == value:
                    return True
        return False

    def captured_metadata(self, value: ObjectRef | StateRef, scope: str):
        """Return one fact captured under this Store cut.

        A missing entry means that the target was not a holder or authority was
        unavailable while fenced; it is deliberately not interpreted as a
        missing metadata field.
        """

        try:
            return self._metadata_index[identity_key(value), scope]
        except KeyError:
            raise KeyError("captured metadata authority is unavailable") from None

    def _authoritative_root_definitions(self) -> Iterable[ConcreteDefinition]:
        by_digest = {record.digest: record.definition for record in self.definitions}
        yield from (by_digest[root.definition_digest] for root in self.stored_roots)
        yield from (record.object_ref.definition for record in self.declarations)
        yield from (record.state_ref.definition for record in self.state_refs)
        yield from (record.object_ref.definition for record in self.object_aliases)
        if self.main_definition is not None:
            yield self.main_definition


def _derive_identities(roots: Iterable[IdentityValue]) -> tuple[IdentityValue, ...]:
    """Expand retained direct relationships without Store reads or side effects."""

    values: dict[object, IdentityValue] = {}
    pending = list(roots)
    while pending:
        value = pending.pop()
        key = identity_key(value)
        if key in values:
            continue
        values[key] = value
        pending.extend(edge.target for edge in iter_direct_relationships(value))
    return tuple(values.values())


def validate_captured_reference_authority(
        facts: Iterable[CapturedStoreFacts]) -> None:
    """Validate complete captured ObjectId mappings without revisiting roots.

    ``facts`` supplies validated Store cuts. Their authoritative projection maps
    are merged, returning None on success and raising QueryError for incompatible
    ObjectId mappings. Embedded references never enter these maps independently.
    This performs no Store access, payload reads, or authority writes.
    """

    by_object_id = {}
    for item in facts:
        for object_id, projected in item._reference_projections.items():
            previous = by_object_id.setdefault(object_id, projected)
            if previous != projected:
                raise QueryError("Query V3 object-reference authority conflicts.")


def _validate_object_reference_roots(roots: Iterable[ObjectRef]) -> dict:
    """Return validated projections without promoting embedded references."""

    by_object_id = {}
    for root in roots:
        for path, object_id in root.objects.items():
            projected = root.at(path)
            previous = by_object_id.setdefault(object_id, projected)
            if previous != projected:
                raise StoreAuthorityError(
                    "Object-reference authority contains incompatible ObjectId mappings."
                )
    return by_object_id
