"""Logical Store authority interface independent from backend paths.

Format-4 Stores distinguish shared Node metadata, independent root memberships,
ObjectId projection anchors, historical manifests and CDef-owned state bytes.
Legacy reference consumer interfaces remain explicit cutover boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport, ValidationReport


class StoreAuthorityError(RuntimeError):
    """Raised when Store authority is malformed, missing, or incompatible."""


class StoreCapabilityError(StoreAuthorityError):
    """Raised when a backend cannot provide required Store semantics.

    Preflight checks raise before mutation when possible. A filesystem
    persistence barrier can fail after publication becomes visible, so this
    error does not by itself promise rollback; Store callers reconcile authority
    where the publication phase matters.
    """


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
        durability: ``durable`` for direct publication, ``buffered`` for extracted
            archive publication, ``read-only`` for read sources, or ``unspecified``
            for custom backends that have not declared this completion boundary.
    """

    writable: bool
    immutable_install: bool
    atomic_replace: bool
    writer_serialization: bool
    same_store_staging: bool
    durability: str = "unspecified"

    def __post_init__(self):
        if self.durability not in {"durable", "buffered", "read-only", "unspecified"}:
            raise ValueError("Unknown Store publication durability capability.")

    def require_durable(self, operation: str) -> None:
        """Require immediate durable publication, refusing buffered/unknown backends.

        operation names the caller's boundary. Raises StoreCapabilityError before
        effects unless writable guarantees and direct durability are declared.
        Archive callers instead use their explicit successful commit boundary.
        """
        self.require_writable(operation)
        if self.durability != "durable":
            raise StoreCapabilityError(f"{operation} requires durable publication, not {self.durability} authority.")

    def require_writable(self, operation: str, *, local_state: bool = False) -> None:
        """Fail closed unless this backend supports ``operation`` publication."""
        if not self.writable:
            raise StoreCapabilityError(f"{operation} requires a writable Store backend.")
        if not (self.immutable_install and self.atomic_replace and self.writer_serialization):
            raise StoreCapabilityError(f"{operation} requires atomic immutable install, small-file replacement, and writer serialization.")
        if local_state and not self.same_store_staging:
            raise StoreCapabilityError(f"{operation} requires same-Store local-state staging.")


@dataclass(frozen=True, slots=True)
class StateDescriptor:
    """Detached historical state metadata, with availability observed at one read.

    state is a LocalState; manifest is a StateVariantManifest; available is current
    publication plus directory presence, not byte integrity or future protection;
    incarnation is the publication UUID, or None before publication. Metadata
    remains usable after Store closure. Construction has no side effects.
    """

    state: object
    manifest: object
    available: bool
    incarnation: str | None


@dataclass(frozen=True, slots=True)
class StateInstallation:
    """Receipt for a state installation primitive, not a journal/index completion.

    state is the detached LocalState, installed distinguishes reuse from new data,
    and durability is durable for DirStore or buffered for ZipStore until commit.
    A receipt confers no lease or retention. Construction performs no IO.
    """

    state: object
    installed: bool
    durability: str


@dataclass(slots=True)
class StateReadSource:
    """Private data borrowed within Store.read_state, never a portable selection.

    The directory is accessible only inside the context and while its Store is
    open. manifest and state are detached metadata. Closure/context exit raises
    RuntimeError on directory access. Built-in sources retain a native full-key
    lease for the context lifetime, excluding cooperating cleanup.
    """

    store: object
    state: object
    manifest: object
    _directory: str
    _active: bool = True
    _metadata_evidence: object = None
    _metadata_owner: object = None

    def __reduce__(self):
        raise TypeError("Borrowed state sources cannot be serialized; transport detached LocalState metadata.")

    @property
    def data_directory(self):
        """Return only codec-private data/ (str); fail after context/owner closure."""
        self.store._assert_state_owner_open()
        if not self._active:
            raise RuntimeError("State read source context has expired.")
        return self._directory


class Store(ABC):
    """Backend-neutral authority contract for metadata and private state sources.

    Backends own public record/metadata readers, local source lifetimes, and
    capability failures. Optional protected read-cut seams preserve those public
    paths by default; explicit legacy backend opt-in may share evidence under
    a retained Repo cut, but cannot establish cross-operation payload validity.
    """

    @property
    def retention(self):
        """Return the durable owner-ledger adapter or refuse unsupported authority.

        Built-in DirStore supports prepare/protect/adopt/abandon/release primitives;
        callers own association policy. Custom/buffered owners raise
        StoreCapabilityError. Merely obtaining the adapter has no side effects.
        """
        raise StoreCapabilityError("Store does not support durable retention ownership.")

    def state_leases(self, keys, *, exclusive=False, blocking=True):
        """Return native full-key protection context, yielding acquisition bool.

        keys is an iterable of StateKeys. Exclusive nonblocking acquisition returns
        False on active use; no upgrade or queued deletion occurs. Unsupported
        backends raise StoreCapabilityError. Acquire before Store writer fences;
        growing a held key set requires release/reacquire and fresh validation.
        """
        raise StoreCapabilityError("Store does not support active StateKey leases.")

    def preflight_save_batch(self):
        """Admit retained Node save batches before capture, or raise capability error.

        Returns None. Implementations must support journaled state/graph/grant
        installation and synchronous enabled projection, including protection
        before association. This base refusal has no effects.
        """
        raise StoreCapabilityError("Store does not support retained Node save batches.")

    def read_states(self, keys):
        """Borrow a full-key batch with one creation interpretation per bucket/cut.

        keys is an iterable of StateKeys. The context yields a key/StateReadSource
        mapping containing available keys only; absent/withdrawn keys are omitted,
        malformed or corrupt authority raises StoreAuthorityError. Sources expire
        on exit. Native ordered leases protect all requested keys. Unsupported
        implementations raise StoreCapabilityError before reading payloads.
        """
        raise StoreCapabilityError("Store does not support grouped state sources.")

    def publish_save_batch(self, batch):
        """Publish one prepared Node save, returning (mutation UUID, authority cut).

        batch is the owning Repo's detached NodeSaveBatch. Validation, payload
        authentication and graph preparation precede writer fences. Installs,
        grants, memberships, declarations and names share one journal/projection
        commit. Protection and owner adoption concern target-local resources;
        published graph metadata does not certify external payload availability.
        Repo separately qualifies protected recovery closures across targets.
        Partial failures preserve exact journal and provisional-root
        evidence. Unsupported backends raise StoreCapabilityError before effects.
        """
        raise StoreCapabilityError("Store does not support retained Node save batches.")

    def repair_save_batch(self, mutation_id, token):
        """Replay an exact failed save under fresh owner/grant authority.

        UUID mutation_id/token come from SavePublication. Returns repaired cut;
        abandoned/retired tokens, missing staging, conflicting cuts or active use
        fail explicitly. No capture or hooks run. Unsupported backends raise
        StoreCapabilityError before effects.
        """
        raise StoreCapabilityError("Store does not support retained Node save repair.")

    def save_batch_needs_recovery(self, batch):
        """Inspect availability metadata to schedule an exclusive save phase.

        batch is NodeSaveBatch. Returns bool, not a lease or future proof. A changed
        availability cut must still be rejected/revalidated during publication;
        callers never upgrade held shared leases. Unsupported backends raise
        StoreCapabilityError without effects.
        """
        raise StoreCapabilityError("Store does not support Node save recovery scheduling.")

    def delete_state(self, key):
        """Immediately try private-data reclamation; return StateRemovalOutcome.

        key is a full StateKey. Retained/in-use results have no queued request.
        Successful withdrawal preserves graph/header/manifest history. Only physical
        garbage failure returns cleanup-pending; journal/index failures propagate.
        Unsupported backends raise StoreCapabilityError before effects.
        """
        raise StoreCapabilityError("Store does not support private-state reclamation.")

    def retry_state_removal(self, key, incarnation):
        """Retry exact withdrawn UUID under a fresh nonblocking exclusive key lease.

        Returns StateRemovalOutcome; newer same-key publication obsoletes public
        data removal. Missing evidence raises StoreAuthorityError; unsupported
        backends raise StoreCapabilityError. Never infers release from owner death.
        """
        raise StoreCapabilityError("Store does not support incarnation-safe cleanup.")

    @property
    @abstractmethod
    def publication_capabilities(self) -> StorePublicationCapabilities:
        """Return the backend's declared current publication guarantees."""

    def preflight_publication(self, operation: str, *, local_state: bool = False,
                              record_family: str | None = None) -> None:
        """Validate publication capability before hooks, with explicit format intent.

        operation is a diagnostic name; local_state requests staging guarantees.
        record_family names a consumer's persistent contract. Built-in format-4
        Stores require ``dryml-node-graph/v1``; an omitted family identifies an
        unmigrated consumer. Returns None or raises StoreCapabilityError without
        authority effects. Custom Stores retain their own family admission policy.
        """
        self.publication_capabilities.require_writable(operation, local_state=local_state)

    def install_creation(self, root):
        """Install normalized creation metadata and return its digest (str).

        root is a ConcreteNode/CDef. Does not publish state, root membership or
        identity declarations. Invalid/incompatible authority raises
        StoreAuthorityError; unsupported backends raise StoreCapabilityError.
        """
        raise StoreCapabilityError("Store does not support identity-3 creation metadata.")

    def install_creations(self, roots):
        """Install creation roots in one shared Node table; return digest tuple.

        roots is an iterable of concrete roots. Validation precedes publication;
        failure may leave installed immutable metadata but never replaces it.
        Unsupported backends raise StoreCapabilityError.
        """
        raise StoreCapabilityError("Store does not support identity-3 creation metadata.")

    def read_creation(self, cdef_digest):
        """Expand validated bucket metadata to a detached ConcreteNode or None.

        cdef_digest is a carried lowercase digest. No private data is read.
        Malformed closure/association raises StoreAuthorityError; an unsupported
        backend raises StoreCapabilityError. This confers no availability.
        """
        raise StoreCapabilityError("Store does not support identity-3 creation metadata.")

    def state_data_path(self, key):
        """Derive backend-private data location from StateKey without graph work.

        Returns a path for supported directory-backed implementations; this is
        addressing only, not availability, validation or a borrowed read lifetime.
        Repo/managed consumers use read_state instead. Unsupported backends raise
        StoreCapabilityError; malformed keys raise TypeError.
        """
        raise StoreCapabilityError("Store does not expose state data addressing.")

    def install_state(self, root, staging, *, codec, codec_version, interpretation):
        """Authenticate owned staging and install/reuse one complete StateKey.

        root is concrete creation evidence, staging an owned allocation from
        create_local_state_staging, codec a name, codec_version a positive int,
        interpretation a JSON mapping. Returns StateInstallation. Codec content
        is confined to staging/data. Invalid evidence raises StoreAuthorityError;
        unsupported publication raises StoreCapabilityError. I/O errors propagate
        and may follow visibility. Built-in Stores journal supported mutations;
        explicit unsupported index coverage is refused before effects. Failures
        retain confirmed authority/cut/replay evidence, without rollback.
        A built-in installation retains active protection only during publication;
        the caller must establish an explicit owning/provisional root before a
        retained association can become current. The receipt itself is unretained.
        """
        raise StoreCapabilityError("Store does not support state buckets.")

    def inspect_states(self, cdef_digest, *, available_only=False):
        """Return detached StateDescriptors for one carried creation digest.

        Reads metadata/presence only, interpreting same-bucket creation once per
        detached, cut-revalidated batch. Header-only buckets return (). Malformed records raise
        StoreAuthorityError; unsupported backends raise StoreCapabilityError.
        No Objects or payload bytes are loaded; no future availability is promised.
        """
        raise StoreCapabilityError("Store does not support state bucket inspection.")

    def read_state(self, key, *, destination=None, codec=None, codec_version=None):
        """Borrow a StateReadSource context for one full StateKey.

        destination, codec name and positive integer codec_version optionally
        validate compatibility/admission before payload IO and must be supplied
        together. Metadata and every private byte
        are validated anew. Missing/malformed/incompatible state raises
        StoreAuthorityError. The source expires on context exit or Store closure;
        built-in Stores retain active full-key protection through context exit.
        Unsupported backends raise StoreCapabilityError.
        """
        raise StoreCapabilityError("Store does not support state bucket sources.")

    def publish_node(self, root, *, kind="recipe"):
        """Install identity-3 root metadata, anchors and explicit membership.

        root is a Node/CDef, kind is recipe or saved. Saved roots must have complete
        facets. Returns its NodeRootRecord, buffered in ZipStore. ObjectId anchors
        compare canonical projections, permitting state-pin changes. Missing or
        conflicting required anchors raise StoreAuthorityError before membership.
        Built-in backends journal this transition and refuse unsupported explicit
        index policies. No retention completion is implied. Unsupported backends
        raise StoreCapabilityError. IO failures carry inspectable partial metadata.
        """
        raise StoreCapabilityError("Store does not support identity-3 root publication.")

    def read_node(self, digest, *, kind="recipe"):
        """Return a detached published Node or None, validating required anchors.

        digest is the carried semantic root key; kind is recipe or saved. Reads
        shared metadata/membership only. Corrupt/missing required authority raises
        StoreAuthorityError; unsupported backends raise StoreCapabilityError.
        Does not load payloads, construct Objects or change authority.
        """
        raise StoreCapabilityError("Store does not support identity-3 root readers.")

    def read_declaration_anchor(self, object_id):
        """Return a validated ObjectId-addressed anchor or None for a new identity.

        Established missing/incompatible projection authority raises
        StoreAuthorityError; unsupported backends raise StoreCapabilityError.
        Reads metadata only and does not create declarations or load Objects.
        """
        raise StoreCapabilityError("Store does not support declaration anchors.")

    def authority_cut(self):
        """Return UUID/epoch/pending AuthorityRecord, or refuse unsupported authority.

        This is a metadata cut, buffered for ZipStore. Missing/corrupt authority
        raises StoreAuthorityError; it never certifies a derived index by itself.
        """
        raise StoreCapabilityError("Store does not support mutation cuts.")

    def pending_mutations(self):
        """Return exact pending intent/replay records without performing repair.

        Unsupported backends raise StoreCapabilityError; inconsistent evidence
        raises StoreAuthorityError. Records do not load graphs or private data.
        """
        raise StoreCapabilityError("Store does not support mutation replay evidence.")

    def recover_authority(self):
        """Classify interrupted publication and return exact pending replay records.

        Recovery preserves authoritative files and staging, without rerunning
        capture/restore/hooks or inventing rollback. Unsupported backends raise
        StoreCapabilityError; corrupt evidence/IO failures remain explicit.
        """
        raise StoreCapabilityError("Store does not support mutation recovery.")

    def apply_prepared_mutation(self, facts):
        """Apply owned immutable PreparedMutation facts; return (mutation UUID, cut).

        Stale witnesses fail before effects. Enabled persistent projection and
        exact acknowledgement complete before return. Nonpersistent low-level
        batches leave work pending for their owner. Unsupported backends raise
        StoreCapabilityError; partial IO failures retain actual cut/evidence.
        """
        raise StoreCapabilityError("Store does not support prepared mutation batches.")

    def replay_mutation(self, mutation_id):
        """Complete one classified pending UUID's exact prepared transitions.

        Returns repaired AuthorityRecord. Later conflicting authority or missing
        staging refuses repair; no capture/restore/hook work runs. Unsupported
        backends raise StoreCapabilityError. Repair is journaled and never rolls
        back or deletes authoritative state to satisfy derived bookkeeping.
        """
        raise StoreCapabilityError("Store does not support exact mutation replay.")

    def acknowledge_mutation(self, mutation_id, epoch):
        """Acknowledge only one classified UUID/cut whose projection was incorporated.

        Returns current AuthorityRecord, retaining later work. Wrong evidence
        raises StoreAuthorityError; unsupported backends raise StoreCapabilityError.
        """
        raise StoreCapabilityError("Store does not support exact mutation acknowledgement.")

    def read_node_name(self, name):
        """Return a validated format-4 NodeNameRecord or None for a path-safe name.

        Dangling/malformed names raise StoreAuthorityError; unsupported backends
        raise StoreCapabilityError. Reads metadata only and does not mutate it.
        """
        raise StoreCapabilityError("Store does not support format-4 Node names.")

    def compare_and_set_node_name(self, record, *, expected):
        """Journal a NodeNameRecord CAS against a record or expected absence (None).

        Returns record; target must be independently published. A mismatch raises
        StoreAliasConflictError before effects; unsupported backends raise
        StoreCapabilityError. Failed IO preserves classified partial authority.
        """
        raise StoreCapabilityError("Store does not support format-4 Node names.")

    def delete_node_name(self, name, *, expected):
        """Remove an exact name CAS, returning bool without deleting graph/state.

        expected is a NodeNameRecord or None. Mismatch raises
        StoreAliasConflictError; unsupported backends raise StoreCapabilityError.
        """
        raise StoreCapabilityError("Store does not support format-4 Node names.")

    def rename_node_name(self, name, new_name, *, expected, expected_target=None):
        """Journal source/target name CAS; return the resulting NodeNameRecord.

        expected is the source record; expected_target is the target record or
        None. Mismatch raises StoreAliasConflictError; interruption preserves
        complete partial mappings/evidence. Unsupported stores raise
        StoreCapabilityError. Graph and private state are never removed.
        """
        raise StoreCapabilityError("Store does not support format-4 Node names.")

    def withdraw_node(self, digest, *, kind="recipe"):
        """Withdraw membership and its declaration contributions, preserving history.

        Returns bool (False for absent/still-named root). No state reclamation or
        retention is implied. Corrupt authority raises StoreAuthorityError;
        unsupported backends raise StoreCapabilityError. Failed IO retains exact
        partial cut and replay evidence.
        """
        raise StoreCapabilityError("Store does not support Node membership withdrawal.")

    def query(self):
        """Create an unevaluated Query V3 identity universe for this Store.

        Returns:
            An :class:`~dryml.core.query.IdentityQuery` whose producer is this
            exact Store. Use ``sel()``, kind restrictions, metadata predicates,
            or relationship operations before an explicit terminal.

        Raises:
            StoreAuthorityError: At terminal evaluation if authoritative records
                are malformed or incomplete.
            StoreCapabilityError: At terminal evaluation if this backend cannot
                enumerate a required authority family.

        Side Effects:
            Construction has none. A terminal reads fenced record authority but
            never materializes Objects, reads local-state payloads, allocates
            identities, writes records, or observes the current host.
        """
        from ..query import IdentityQuery

        return IdentityQuery.from_store(self)

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
    def create_local_state_staging(self) -> object:
        """Create a backend-owned empty staging handle before serializer hooks run."""

    @abstractmethod
    def discard_local_state_staging(self, handle: object) -> None:
        """Discard one unconsumed backend-owned local-state staging handle."""

        raise NotImplementedError("This Store does not expose owned local-state staging.")

    def read_metadata(self, target, *, scope=None):
        """Return the current mapping for one concrete Node, if present.

        Args:
            target: Concrete Node object projection or saved graph already validated
                by the caller as held in this Store.
            scope: object/state, or None to infer from the Node's selected state.

        Returns:
            A detached current mapping, or ``None`` for an absent attachment.

        Raises:
            TypeError: If the target is unsupported.
            StoreAuthorityError: If a present record is malformed or names another
                exact target.
            StoreCapabilityError: If the backend has no current-metadata reader.

        Side Effects:
            Never opens Stores, materializes Objects, reads payload files, or
            changes authority. Repo owns cross-Store consistency checks.
        """

        raise StoreCapabilityError("This Store does not expose current metadata.")

    def write_metadata(self, target, values, *, scope=None) -> None:
        """Atomically replace one current metadata mapping.

        Args:
            target: Concrete Node object projection or saved graph held by this Store.
            values: Valid complete metadata mapping replacing the prior mapping.
            scope: object/state, or None to infer from the Node's selected state.

        Returns:
            ``None`` after the backend accepts the replacement.

        Raises:
            TypeError: If target or values have unsupported types.
            ValueError: If values violate the metadata codec bounds.
            StoreAuthorityError: If the write cannot name valid target authority.
            StoreCapabilityError: If atomic current-metadata mutation is unsupported.

        Side Effects:
            Implementations preserve whole-map last-writer-wins semantics under
            their writer fence. Unsupported mutation fails before changing
            authority; no captured snapshot metadata is rewritten.
        """

        raise StoreCapabilityError("This Store does not support current metadata mutation.")

    def delete_metadata(self, target, *, scope=None) -> bool:
        """Atomically remove one current metadata mapping and report its presence.

        Args:
            target: Concrete Node object projection or saved graph held by this Store.
            scope: object/state, or None to infer from the Node's selected state.

        Returns:
            ``True`` when a mapping was removed, otherwise ``False``.

        Raises:
            TypeError: If the target is unsupported.
            StoreCapabilityError: If current-metadata mutation is unsupported.

        Side Effects:
            Runs under the backend writer fence and removes no snapshot, payload,
            lineage, or reference authority.
        """

        raise StoreCapabilityError("This Store does not support current metadata mutation.")

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
