"""Detached Store authority facts for Query V3 producer capture.

Knowledge inventory, stored membership, and metadata-holder eligibility are
kept separate because existing Store authority assigns them different meanings.
The module owns no Repo configuration and never reads a Store after capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..definition import ConcreteDefinition
from ..reference_values import ObjectRef, StateRef
from ..store.records import (
    DeclarationRecord, DefinitionRecord, ObjectAliasRecord, StateAliasRecord,
    StateRefRecord, StoredRootRecord,
)
from ..store.store import StoreAuthorityError
from .identity import IdentitySet, SourceEvidence, identity_key
from .relationships import iter_direct_relationships


IdentityValue = ConcreteDefinition | ObjectRef | StateRef


@dataclass(frozen=True, slots=True)
class CapturedStoreFacts:
    """One complete detached Store authority cut and its derived identity facts.

    Record tuples are retained for later U3-U6 consumers. Membership and holder
    checks use those captured values without broad iterator rescans.
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
    _metadata_index: dict[tuple[IdentityKey, str], object] = field(
        init=False, compare=False, repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_metadata_index",
            {(key, scope): value for key, scope, value in self.metadata},
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
        """Validate cross-record references and create one detached fact table."""

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
        return cls(
            definitions, stored_roots, tuple(declarations), state_refs,
            tuple(object_aliases), state_aliases, main_definition, source,
            tuple(metadata), metadata_scopes,
        )

    def knowledge(self) -> IdentitySet:
        """Return all represented and retained derivable identities in this cut."""

        roots: list[IdentityValue] = [record.definition for record in self.definitions]
        roots.extend(record.object_ref for record in self.declarations)
        roots.extend(record.state_ref for record in self.state_refs)
        roots.extend(record.object_ref for record in self.object_aliases)
        roots.extend(record.object_ref for record in self.state_aliases)
        if self.main_definition is not None:
            roots.append(self.main_definition)
        return IdentitySet((value, self.source) for value in _derive_identities(roots))

    def is_stored(self, value: IdentityValue) -> bool:
        """Return kind-specific independent stored authority from captured records."""

        if isinstance(value, ConcreteDefinition):
            return any(
                identity_key(candidate) == identity_key(value)
                for candidate in self._authoritative_root_definitions()
            )
        if isinstance(value, ObjectRef):
            return any(record.object_ref == value for record in self.declarations) or any(
                record.state_ref.object == value for record in self.state_refs
            )
        if isinstance(value, StateRef):
            return any(record.state_ref == value for record in self.state_refs)
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
