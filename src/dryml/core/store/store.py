"""Logical Store authority interface independent from backend paths.

Current Stores persist immutable definitions, declarations, claims, complete
snapshot directories, and mutable references. Local-state payloads are valid
only within the snapshot directory that publishes their exact StateRef.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Iterable

from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport, ValidationReport
from .records import (
    ClaimRecord, DeclarationRecord, DefinitionRecord, LocalStateManifest,
    MainRefRecord, ObjectAliasRecord, StateAliasRecord, StateRefRecord,
    StoredRootRecord,
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


@dataclass(frozen=True, slots=True)
class LocalStateSource:
    """Validated local-state bytes borrowed from one open Store.

    ``handle`` is backend-private.  A source prepared from staging is owned by
    its allocating Store until discarded; a source opened from a snapshot is
    borrowed immutable authority and cannot be discarded by its consumer.
    """

    store: "Store"
    handle: object
    manifest: LocalStateManifest


class Store(ABC):
    """Backend-neutral authority contract for complete snapshot-local records."""

    @property
    @abstractmethod
    def publication_capabilities(self) -> StorePublicationCapabilities:
        """Return the backend's declared current publication guarantees."""

    def preflight_publication(self, operation: str, *, local_state: bool = False) -> None:
        """Validate writable publication semantics before a caller invokes hooks."""
        self.publication_capabilities.require_writable(operation, local_state=local_state)

    def to_definition(self) -> dict[str, Any]:
        """Export a detached portable existing-Store descriptor.

        Returns:
            A detached mapping naming this Store's supported backend, persistent
            location, and opening settings.

        Raises:
            RepoDefinitionError: If this Store is unsupported, has nonportable
            settings, or is a dirty/file-like Zip transaction.

        Side Effects:
            None. Export neither saves data nor commits buffered authority.
        """

        from ..repo_definition import definition_from_store

        return definition_from_store(self)

    @classmethod
    def from_definition(cls, definition: Mapping[str, Any]) -> "Store":
        """Open existing Store authority from one detached portable descriptor.

        Args:
            definition: Mapping emitted by :meth:`to_definition` for a supported
                existing DirStore or path-backed ZipStore.

        Returns:
            A concrete Store. With an active Session resource cache, a matching
            valid handle can be returned; otherwise the caller owns a fresh handle.

        Raises:
            TypeError: If ``definition`` is not a mapping.
            RepoDefinitionError: If the descriptor is malformed or unsupported.
            StoreAuthorityError: If existing authority is unavailable, malformed,
                or incompatible.

        Side Effects:
            Opens only existing authority and may register a cache-owned handle.
            It never creates, repairs, commits, or materializes Store contents.
        """

        if not isinstance(definition, Mapping):
            raise TypeError("Store definition must be a mapping.")
        from ..repo_definition import _open_store_descriptor

        return _open_store_descriptor(definition)

    def writer_lock(self):
        """Return a context manager serializing cooperating Store writers.

        Backends with durable authority must override this with the same lock
        used by their mutable-record replacement primitive.  The default keeps
        read-only or in-memory test Stores source-compatible.
        """
        return nullcontext()

    def authority_fence_key(self) -> str:
        """Return the stable lock-domain key for an authority evidence cut.

        Returns:
            A deterministic key identifying the cooperating writer domain used by
            :meth:`authority_read_fence`.

        Raises:
            StoreCapabilityError: If this backend does not declare serialized
            writer authority and therefore cannot provide a stable evidence cut.

        Side Effects:
            None. This method neither reads nor changes Store authority.
        """
        if not self.publication_capabilities.writer_serialization:
            raise StoreCapabilityError(
                "Store does not provide a stable authority-read fence."
            )
        return self.catalog_key()

    def authority_read_fence(self):
        """Return the fence that stabilizes authoritative metadata reads.

        Returns:
            A context manager excluding cooperating writers for the duration of
            one metadata evidence cut.

        Raises:
            StoreCapabilityError: If the backend cannot provide writer-serialized
            authority reads.

        Side Effects:
            Acquires the backend's cooperating-writer fence; it never publishes,
            loads payloads, or consults a derived query index.
        """
        self.authority_fence_key()
        return self.writer_lock()

    @abstractmethod
    def read_definition_record(self, digest: str) -> DefinitionRecord | None:
        """Read one immutable DefinitionRecord by its derived digest."""

    @abstractmethod
    def write_definition_record(
            self, record: DefinitionRecord, *, stored_root: bool = True
    ) -> DefinitionRecord:
        """Install a definition and optionally publish stored-root membership."""

    @abstractmethod
    def iter_definition_records(self) -> Iterable[DefinitionRecord]:
        """Yield every complete immutable DefinitionRecord in this Store."""

    @abstractmethod
    def iter_stored_root_records(self):
        """Yield authoritative stored-root membership records."""

    def read_stored_root_record(self, digest: str) -> StoredRootRecord | None:
        """Read and validate one stored-root membership by DefinitionRecord digest.

        Args:
            digest: Derived digest naming both the requested StoredRootRecord and
                its referenced DefinitionRecord.

        Returns:
            The matching authoritative StoredRootRecord, or ``None`` when no
            membership exists for ``digest``.

        Raises:
            StoreAuthorityError: If the requested membership is malformed, does
                not name ``digest``, or references missing DefinitionRecord
                authority.

        Side Effects:
            None. Existing Store implementations inherit this compatibility
            fallback, which validates through their complete stored-root iterator.
            Backends should override it with a digest-addressed authority read.
        """
        try:
            expected = StoredRootRecord(digest)
        except Exception as error:
            raise StoreAuthorityError("Stored-root digest is malformed.") from error
        for record in self.iter_stored_root_records():
            if record.definition_digest != digest:
                continue
            if record != expected:
                raise StoreAuthorityError("Stored-root membership does not match its requested digest.")
            if self.read_definition_record(digest) is None:
                raise StoreAuthorityError(
                    "Stored-root membership targets a missing DefinitionRecord."
                )
            return record
        return None

    @abstractmethod
    def create_local_state_staging(self) -> object:
        """Create a backend-owned empty staging handle before serializer hooks run."""

    @abstractmethod
    def discard_local_state_staging(self, handle: object) -> None:
        """Discard one unconsumed backend-owned local-state staging handle."""

        raise NotImplementedError("This Store does not expose owned local-state staging.")

    @abstractmethod
    def prepare_local_state(self, source: object, manifest: LocalStateManifest) -> LocalStateSource:
        """Validate completed owned staging and return a typed payload source."""

        raise NotImplementedError("This Store does not prepare local-state sources.")

    @abstractmethod
    def prepare_rebound_local_state(self, source: LocalStateSource, target_definition) -> LocalStateSource:
        """Copy a verified source into owned staging with rebound definition evidence."""

        raise NotImplementedError("This Store does not prepare rebound local state.")

    @abstractmethod
    def open_local_state(self, reference, path) -> LocalStateSource:
        """Open and fully validate one snapshot-local payload source."""

        raise NotImplementedError("This Store does not expose snapshot-local payloads.")

    def validate_local_state(self, reference, path) -> LocalStateManifest:
        """Validate one snapshot-local payload manifest and return it."""

        raise NotImplementedError("This Store does not validate snapshot-local payloads.")

    @abstractmethod
    def read_state_ref_record(self, digest: str) -> StateRefRecord | None:
        """Read one immutable StateRefRecord by digest."""

    @abstractmethod
    def get_snapshot_directory(self, target) -> object:
        """Return the complete snapshot directory for one exact StateRef."""

        raise NotImplementedError("This Store does not expose snapshot directories.")

    @abstractmethod
    def publish_snapshot(self, reference, *, evidence, local_states, children=None):
        """Atomically publish one complete snapshot-local authority directory."""

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

    def iter_object_alias_records(self) -> Iterable[ObjectAliasRecord]:
        """Yield complete object aliases for derived-index rebuilds.

        Backends that cannot enumerate aliases may return no entries; aliases
        remain authoritative through direct lookup.
        """
        return ()

    def iter_state_alias_records(self) -> Iterable[StateAliasRecord]:
        """Yield complete state aliases for derived-index rebuilds.

        Backends that cannot enumerate aliases may return no entries; aliases
        remain authoritative through direct lookup.
        """
        return ()

    # Query remains derived: it only scans immutable definitions.
    def authoritative_root_definitions(self):
        """Return complete query-root definitions reconstructed from authority.

        Returns:
            Deduplicated definitions named by stored-root, declaration, StateRef,
            object-alias, or main-reference authority.

        Raises:
            StoreAuthorityError: If a root reference targets missing definition
                authority or the backend cannot enumerate current records.

        Side Effects:
            None. This method validates records without changing Store authority
            or derived index files.
        """
        records = {
            record.digest: record for record in self.iter_definition_records()
        }
        definitions = []
        by_hash = {}

        def add(definition):
            bucket = by_hash.setdefault(definition.graph_hash(), [])
            if any(existing.graph_equal(definition) for existing in bucket):
                return
            bucket.append(definition)
            definitions.append(definition)

        for root in self.iter_stored_root_records():
            record = records.get(root.definition_digest)
            if record is None:
                raise StoreAuthorityError(
                    "Stored-root membership targets a missing DefinitionRecord."
                )
            add(record.definition)
        for record in self.iter_declaration_records():
            add(record.object_ref.definition)
        for record in self.iter_state_ref_records():
            add(record.state_ref.definition)
        for record in self.iter_object_alias_records():
            add(record.object_ref.definition)
        main = self.read_main_ref()
        if main is not None:
            record = records.get(main.definition_digest)
            if record is None:
                raise StoreAuthorityError(
                    "MainRefRecord targets a missing DefinitionRecord."
                )
            add(record.definition)
        return tuple(definitions)

    def hydrate_index(self):
        """Return validated authoritative roots for memory-index hydration.

        Returns:
            The definitions returned by :meth:`authoritative_root_definitions`.

        Raises:
            StoreAuthorityError: If root authority is incomplete or malformed.
        """
        return self.authoritative_root_definitions()

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
        """Release backend-local resources.

        Raises:
            RuntimeError: If an active Session resource cache retains this raw
                Store handle.
        """

        from ..session import _assert_resource_close_allowed

        _assert_resource_close_allowed(self)
