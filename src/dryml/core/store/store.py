"""Logical Store authority interface independent from backend paths.

Current Stores persist immutable definition, local-state, declaration, claim,
and StateRef records plus mutable references.  This module deliberately has no
object-root or state-generation protocol: graph publication is owned by U5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport, ValidationReport
from .records import (
    ClaimRecord, DeclarationRecord, DefinitionRecord, LocalStateManifest,
    MainRefRecord, ObjectAliasRecord, StateAliasRecord, StateRefRecord,
)


class StoreAuthorityError(RuntimeError):
    """Raised when Store authority is malformed, missing, or incompatible."""


class StoreCapabilityError(StoreAuthorityError):
    """Raised before mutation when a backend cannot provide required semantics."""


class StoreAliasConflictError(StoreAuthorityError):
    """Raised when concurrent mutable alias changes conflict."""


@dataclass(frozen=True, slots=True)
class StorePublicationCapabilities:
    """Declared backend guarantees used before local-state publication.

    Attributes:
        writable: Whether authority mutation is supported.
        immutable_install: Whether an immutable record/directory can be made
            visible only as a complete absent destination.
        atomic_replace: Whether mutable small records replace atomically.
        writer_serialization: Whether cooperating writers serialize mutation.
        same_store_staging: Whether local-state staging is on the publication
            backend's own atomicity domain.
    """

    writable: bool
    immutable_install: bool
    atomic_replace: bool
    writer_serialization: bool
    same_store_staging: bool

    def require_writable(self, operation: str, *, local_state: bool = False) -> None:
        """Fail closed unless this backend supports ``operation`` publication."""
        if not self.writable:
            raise StoreCapabilityError(f"{operation} requires a writable Store backend.")
        if not (self.immutable_install and self.atomic_replace and self.writer_serialization):
            raise StoreCapabilityError(f"{operation} requires atomic immutable install, small-file replacement, and writer serialization.")
        if local_state and not self.same_store_staging:
            raise StoreCapabilityError(f"{operation} requires same-Store local-state staging.")


class Store(ABC):
    """Backend-neutral authority contract for current logical Store records."""

    @property
    @abstractmethod
    def publication_capabilities(self) -> StorePublicationCapabilities:
        """Return the backend's declared current publication guarantees."""

    def preflight_publication(self, operation: str, *, local_state: bool = False) -> None:
        """Validate writable publication semantics before a caller invokes hooks."""
        self.publication_capabilities.require_writable(operation, local_state=local_state)

    def writer_lock(self):
        """Return a context manager serializing cooperating Store writers.

        Backends with durable authority must override this with the same lock
        used by their mutable-record replacement primitive.  The default keeps
        read-only or in-memory test Stores source-compatible.
        """
        return nullcontext()

    @abstractmethod
    def read_definition_record(self, digest: str) -> DefinitionRecord | None:
        """Read one immutable DefinitionRecord by its derived digest."""

    @abstractmethod
    def write_definition_record(self, record: DefinitionRecord) -> DefinitionRecord:
        """Install an immutable DefinitionRecord or validate an idempotent collision."""

    @abstractmethod
    def iter_definition_records(self) -> Iterable[DefinitionRecord]:
        """Yield every complete immutable DefinitionRecord in this Store."""

    @abstractmethod
    def create_local_state_staging(self) -> object:
        """Create a backend-owned empty staging handle before serializer hooks run."""

    @abstractmethod
    def install_local_state(self, source: object, manifest: LocalStateManifest) -> LocalStateManifest:
        """Install a complete immutable local state from a backend-defined staging handle."""

    @abstractmethod
    def open_local_state(self, graph_hash: str, state_hash: str) -> object:
        """Return a verified backend-defined local-state handle for consumption."""

    @abstractmethod
    def validate_local_state(self, definition, state_hash: str) -> object:
        """Verify a local state against its exact graph definition before reuse."""

    @abstractmethod
    def copy_local_state_from(self, source: "Store", definition, state_hash: str) -> LocalStateManifest:
        """Copy one verified immutable local state into this Store's authority."""

    @abstractmethod
    def read_state_ref_record(self, digest: str) -> StateRefRecord | None:
        """Read one immutable StateRefRecord by digest."""

    @abstractmethod
    def write_state_ref_record(self, record: StateRefRecord) -> StateRefRecord:
        """Install one immutable StateRefRecord after its closure is available."""

    def iter_state_ref_records(self) -> Iterable[StateRefRecord]:
        """Yield complete immutable StateRef records for authority scans."""
        raise NotImplementedError("This Store does not expose StateRef authority scans.")

    @abstractmethod
    def read_declaration_record(self, digest: str) -> DeclarationRecord | None:
        """Read one immutable DeclarationRecord by ObjectRef digest."""

    @abstractmethod
    def write_declaration_record(self, record: DeclarationRecord) -> DeclarationRecord:
        """Install one immutable DeclarationRecord."""

    def iter_declaration_records(self) -> Iterable[DeclarationRecord]:
        """Yield complete immutable declaration records for authority scans."""
        raise NotImplementedError("This Store does not expose declaration authority scans.")

    @abstractmethod
    def read_claim_record(self, digest: str) -> ClaimRecord | None:
        """Read one mutable ClaimRecord by ObjectRef digest."""

    @abstractmethod
    def write_claim_record(self, record: ClaimRecord) -> ClaimRecord:
        """Atomically replace one mutable ClaimRecord under writer serialization."""

    @abstractmethod
    def read_main_ref(self) -> MainRefRecord | None:
        """Read the Store's mutable main-definition reference."""

    @abstractmethod
    def write_main_ref(self, record: MainRefRecord) -> MainRefRecord:
        """Atomically replace the Store's mutable main-definition reference."""

    @abstractmethod
    def read_object_alias(self, alias: str) -> ObjectAliasRecord | None:
        """Read one mutable object alias record."""

    @abstractmethod
    def write_object_alias(self, record: ObjectAliasRecord) -> ObjectAliasRecord:
        """Atomically replace one mutable object alias record."""

    @abstractmethod
    def read_state_alias(self, object_digest: str, alias: str) -> StateAliasRecord | None:
        """Read one mutable StateRef alias scoped by ObjectRef digest."""

    @abstractmethod
    def write_state_alias(self, record: StateAliasRecord) -> StateAliasRecord:
        """Atomically replace one mutable StateRef alias record."""

    # Query remains derived: it only scans immutable definitions.
    def hydrate_index(self):
        """Yield definitions reconstructed from authoritative DefinitionRecords."""
        return (record.definition for record in self.iter_definition_records())

    def catalog_key(self) -> str:
        """Return a backend-local identity for derived query catalog deduplication."""
        return f"{type(self).__module__}.{type(self).__qualname__}:id:{id(self)}"

    def open_query_index(self):
        """Return the optional derived query index, if this backend owns one."""
        return None

    def query_index_status(self) -> QueryIndexStatus:
        """Return backend-neutral disabled query-index status."""
        return QueryIndexStatus("none", self.catalog_key(), None, None, {}, "disabled")

    def rebuild_query_index(self) -> ReconcileReport:
        """Reject rebuilding when this Store has no persistent derived index."""
        raise QueryIndexUnavailable(f"Store {self!r} does not own a rebuildable query index.")

    def reconcile_query_index(self) -> ReconcileReport:
        """Rebuild the derived query index when the backend supports it."""
        return self.rebuild_query_index()

    def validate_query_index(self, *, thorough: bool = False) -> ValidationReport:
        """Return a successful disabled-index validation report."""
        return ValidationReport("none", self.catalog_key(), True)

    def commit(self) -> None:
        """Commit buffered backend authority; direct Stores implement a no-op."""

    def close(self) -> None:
        """Release backend-local resources."""
