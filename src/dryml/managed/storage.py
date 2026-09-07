"""Selected DirStore authority for managed control and Object state.

This module deliberately resolves only caller-selected Stores or the current core
session Repo.  It neither records a locator nor searches control Stores.
"""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import get_ident

from dryml.core.reference_values import ObjectId, StateRef
from dryml.core.repo import Repo, RepoSaveError
from dryml.core.session import current_repo
from dryml.core.store.dir import DirStore
from dryml.locking import FileLock, LockError

from .errors import ManagedConflictError, ManagedRecoveryError, ManagedStoreError


@dataclass(frozen=True, slots=True)
class ResolvedStores:
    """Immutable selected state and control authority for one managed operation.

    Attributes:
        state_store: DirStore holding exact Object-state references.
        control_store: DirStore holding managed current-operation authority.

    The caller retains ownership of both Store handles.  Resolution opens no
    query indexes and does not create managed control files.
    """

    state_store: DirStore
    control_store: DirStore


class _StateStoreLockLease(AbstractContextManager):
    """A process/thread-bound retained lease over selected state-Store lock files.

    Instances are created by :func:`_acquire_state_locks`; callers retain no native
    descriptors themselves.  The lease is invalid after fork and cannot release a
    parent process's locks or a different thread's locks.
    """

    def __init__(self, state_store: DirStore, object_ids: tuple[object, ...], leases: tuple[FileLock, ...]):
        """Record the acquired shared-owner leases for one exact ObjectId set."""

        self.state_store = state_store
        self.object_ids = object_ids
        self._leases = leases
        self._pid = os.getpid()
        self._thread_id = get_ident()
        self._active = True

    @property
    def active(self) -> bool:
        """Return whether the creating process and thread still own every lease."""

        return self._active and self._pid == os.getpid() and self._thread_id == get_ident()

    def __enter__(self) -> "_StateStoreLockLease":
        """Return this active lease for context-manager use.

        Raises:
            ManagedConflictError: If the lease belongs to another thread/process
                or was already released.
        """

        self.require_owner()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Release the owned state locks without suppressing an exception."""

        self.release()
        return False

    def require_owner(self) -> None:
        """Require use by the process/thread that acquired this lease.

        Raises:
            ManagedConflictError: If the token is inactive, inherited, or used by
                a non-owning thread.
        """

        if not self.active:
            raise ManagedConflictError("invalid_state_owner", "state lock ownership belongs to another process/thread or is inactive")

    def release(self) -> bool:
        """Release every held lock in reverse acquisition order.

        Returns:
            ``True`` when this owning call released the lease, otherwise ``False``
            for an already released, inherited, or cross-thread token.

        Raises:
            ManagedStoreError: If a shared native lock release fails after all
                possible local cleanup has been attempted.
        """

        if not self.active:
            return False
        self._active = False
        error = None
        for lease in reversed(self._leases):
            try:
                lease.release()
            except LockError as caught:
                if error is None:
                    error = caught
        if error is not None:
            raise ManagedStoreError("state_lock_release_failed", "could not release managed state ownership") from error
        return True


class _ManagedStateOwnership(AbstractContextManager):
    """Compose a core live-graph reservation with selected state-Store leases.

    The private owner token is created by :func:`_acquire_state_ownership` and is
    retained through a future U6 invocation, including user callbacks and final
    publication.  It never serializes a Repo, Store, or native descriptor.
    """

    def __init__(self, reservation, state_locks: _StateStoreLockLease):
        """Join already-acquired core and state-Store ownership resources."""

        self.reservation = reservation
        self.state_locks = state_locks
        self.object_ids = state_locks.object_ids
        self._pid = os.getpid()
        self._thread_id = get_ident()
        self._active = True

    @property
    def active(self) -> bool:
        """Return whether both ownership layers remain valid for this caller."""

        return (
            self._active
            and self._pid == os.getpid()
            and self._thread_id == get_ident()
            and self.reservation.active
            and self.state_locks.active
        )

    def __enter__(self) -> "_ManagedStateOwnership":
        """Return the active combined owner for context-manager use."""

        self.require_owner()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Release state-Store and live-graph ownership without suppressing errors."""

        self.release()
        return False

    def require_owner(self) -> None:
        """Require the exact process/thread that acquired combined ownership.

        Raises:
            ManagedConflictError: If either layer is inactive, inherited, or used
                by a non-owning thread.
        """

        if not self.active:
            raise ManagedConflictError("invalid_state_owner", "managed state ownership belongs to another process/thread or is inactive")

    def release(self) -> bool:
        """Release state leases and then the core reservation when still owned.

        Returns:
            ``True`` if this owning call released the combined token. Inherited or
            cross-thread calls return ``False`` and cannot release parent state.

        Raises:
            ManagedStoreError: If shared state-lock cleanup fails.
        """

        if not self.active:
            return False
        self._active = False
        try:
            self.state_locks.release()
        finally:
            self.reservation.release()
        return True


def resolve_stores(
        obj: object,
        *,
        state_store: DirStore | None = None,
        control_store: DirStore | None = None,
        require_writable: bool = True) -> ResolvedStores:
    """Resolve managed Store authority before an operation mutates anything.

    Args:
        obj: Live Object whose exact last completed StateRef may disambiguate a
            multi-Store session Repo.
        state_store: Explicit DirStore for state, if supplied.
        control_store: Explicit DirStore for control, if supplied.
        require_writable: Whether both selected Stores must support publication.

    Returns:
        The selected state/control Store pair.  Omitted control authority uses
        the selected state Store.

    Raises:
        ManagedStoreError: If a Store is unsupported, session discovery is
            missing or ambiguous, publication is unavailable, or an exact saved
            StateRef cannot be validated without materializing an Object.

    Side Effects:
        Validation reads immutable StateRef authority only.  It never creates a
        Repo, opens a query index, writes a Store, or migrates state.
    """

    if state_store is not None and type(state_store) is not DirStore:
        raise ManagedStoreError(message="state_store must be an exact DirStore")
    if control_store is not None and type(control_store) is not DirStore:
        raise ManagedStoreError(message="control_store must be an exact DirStore")

    selected_state = state_store
    if selected_state is None:
        selected_state = _resolve_session_state_store(obj)
    selected_control = selected_state if control_store is None else control_store
    if require_writable:
        _require_publication(selected_state, "managed state publication", local_state=True)
        _require_publication(selected_control, "managed control publication", local_state=False)
    return ResolvedStores(selected_state, selected_control)


def validate_state_ref(store: DirStore, state_ref: StateRef) -> StateRef:
    """Validate one exact StateRef closure in one selected Store without hooks.

    Args:
        store: DirStore that must contain the complete immutable closure.
        state_ref: Exact StateRef to validate.

    Returns:
        The same exact StateRef after authority-only preflight succeeds.

    Raises:
        ManagedRecoveryError: If the record is absent, differs from the exact
            reference, or its local-state closure cannot be read.

    Side Effects:
        Creates and closes a private non-owning Repo view.  It never
        materializes, restores, indexes, or mutates user Objects or Stores.
    """

    if type(store) is not DirStore or type(state_ref) is not StateRef:
        raise ManagedRecoveryError("invalid_state_reference", "managed state validation requires exact DirStore and StateRef")
    try:
        record = store.read_state_ref_record(state_ref.digest())
        if record is None or record.state_ref != state_ref:
            raise ManagedRecoveryError("missing_state_reference", "selected state Store lacks the exact StateRef")
        from dryml.core.materialization import build_exact_state_load_plan

        view = Repo._for_state_io((store,))
        try:
            build_exact_state_load_plan(view, state_ref)
        finally:
            view.close()
    except ManagedRecoveryError:
        raise
    except BaseException as error:
        raise ManagedRecoveryError("invalid_state_reference", "selected state Store cannot validate the exact StateRef closure") from error
    return state_ref


def _state_lock_path(state_store: DirStore, object_id: object) -> str:
    """Return the canonical managed lifetime-lock path for one ObjectId.

    Args:
        state_store: Exact selected DirStore whose physical root owns the lock
            namespace.
        object_id: Exact core ``ObjectId`` for one stateful graph node.

    Returns:
        A path below ``managed/locks/v1/<hh>/`` named by a SHA-256 digest of the
        canonical ObjectId encoding.

    Raises:
        ManagedStoreError: If the Store root or ObjectId cannot supply managed
            lifetime ownership.

    Side Effects:
        None. This function never creates a namespace or opens a lock file.
    """

    root = _state_store_root(state_store)[0]
    digest = _object_lock_digest(object_id)
    return os.path.join(root, "managed", "locks", "v1", digest[:2], digest + ".lock")


def _acquire_state_locks(state_store: DirStore, object_ids) -> _StateStoreLockLease:
    """Acquire every selected state-Store ObjectId lock nonblockingly.

    Args:
        state_store: Exact DirStore selected as Object-state authority.
        object_ids: Iterable of exact stateful ``ObjectId`` values.

    Returns:
        A retained, process/thread-bound lease covering the deduplicated canonical
        ObjectId order.

    Raises:
        ManagedConflictError: If any lock is already held by a cooperating owner.
        ManagedStoreError: If no stateful identities are supplied, bootstrap fails,
            or native lock support encounters a non-contention failure.

    Side Effects:
        Initializes the selected state Store's closed ``managed/`` format gate
        before creating lock files. Partial acquisition is released in reverse
        order, so a failure never leaves a subset of this operation's leases held.
    """

    ordered = _ordered_object_ids(object_ids)
    if not ordered:
        raise ManagedStoreError("stateful_graph_required", "managed ownership requires at least one stateful ObjectId")
    _initialize_state_lock_namespace(state_store)
    leases = []
    try:
        for object_id in ordered:
            lease = FileLock(_state_lock_path(state_store, object_id))
            if not lease.acquire(blocking=False):
                raise ManagedConflictError("state_lock_conflict", "managed state lock is already held")
            leases.append(lease)
    except BaseException as error:
        cleanup_error = None
        for lease in reversed(leases):
            try:
                lease.release()
            except LockError as caught:
                if cleanup_error is None:
                    cleanup_error = caught
        if cleanup_error is not None:
            raise ManagedStoreError("state_lock_release_failed", "could not release partial managed state ownership") from cleanup_error
        if isinstance(error, ManagedConflictError):
            raise
        if isinstance(error, LockError):
            raise ManagedStoreError("state_lock_unavailable", "could not acquire managed state ownership") from error
        raise
    return _StateStoreLockLease(state_store, ordered, tuple(leases))


def _acquire_state_ownership(repo: Repo, obj: object, state_store: DirStore) -> _ManagedStateOwnership:
    """Reserve one live graph and lease all of its selected state-Store ObjectIds.

    Args:
        repo: Repo owning live graph preflight and process-local reservation.
        obj: Live graph root whose stateful descendants require ownership.
        state_store: Explicit DirStore selected for all managed Object-state work.

    Returns:
        A combined owner token that must remain active for the complete future
        managed invocation.

    Raises:
        ManagedConflictError: If the live graph or any selected state lock is
            already owned, or a token is used by another thread/process.
        ManagedStoreError: If pending initial-construction claims require another
            physical state Store or the selected Store cannot support ownership.

    Side Effects:
        Acquires a core reservation followed by all selected state-Store locks.
        Any later acquisition failure releases the reservation without consuming or
        abandoning caller-owned first-construction claims.
    """

    if not isinstance(repo, Repo):
        raise TypeError("managed state ownership requires a Repo")
    _, nodes, object_ids = repo._state_graph_evidence(obj)
    _validate_pending_claim_stores(nodes, state_store)
    try:
        reservation = repo.reserve_state_graph(obj)
    except RepoSaveError as error:
        raise ManagedConflictError("state_graph_conflict", str(error)) from error
    try:
        state_locks = _acquire_state_locks(state_store, object_ids)
    except BaseException:
        reservation.release()
        raise
    return _ManagedStateOwnership(reservation, state_locks)


def _probe_state_ownership(state_store: DirStore, object_ids) -> bool:
    """Return whether every requested managed state lock is presently available.

    Args:
        state_store: Selected state authority containing the managed lock namespace.
        object_ids: Iterable of stateful ObjectIds addressed by an inspected owner.

    Returns:
        ``True`` only when all leases were acquired and then released. ``False``
        denotes contention and is deliberately inconclusive about a specific
        control-snapshot owner.

    Raises:
        ManagedStoreError: If no valid initialized managed lock namespace exists
            or a native lock failure prevents a reliable probe.

    Side Effects:
        Briefly opens then releases every requested FileLock. It performs no
        control mutation and never interprets contention as process death.
    """

    _require_state_lock_namespace(state_store)
    try:
        lease = _acquire_state_locks(state_store, object_ids)
    except ManagedConflictError:
        return False
    lease.release()
    return True


def _resolve_session_state_store(obj: object) -> DirStore:
    """Choose the sole physical Store or a unique exact-current StateRef match."""

    repo = current_repo()
    if repo is None:
        raise ManagedStoreError("explicit_state_store_required", "no current Repo; supply state_store explicitly")
    stores = tuple(getattr(repo, "stores", ()))
    if not stores:
        raise ManagedStoreError("explicit_state_store_required", "current Repo has no Stores; supply state_store explicitly")
    if any(type(store) is not DirStore for store in stores):
        raise ManagedStoreError("explicit_state_store_required", "current Repo has a non-DirStore authority; supply state_store explicitly")
    physical = _deduplicate_roots(stores)
    if len(physical) == 1:
        return physical[0]
    state_ref = getattr(obj, "last_state_ref", None)
    if type(state_ref) is not StateRef:
        raise ManagedStoreError("explicit_state_store_required", "multiple Stores require one exact current StateRef match")
    matches = []
    for store in physical:
        try:
            validate_state_ref(store, state_ref)
        except ManagedRecoveryError as error:
            if error.__cause__ is not None:
                raise ManagedStoreError("state_store_unreadable", "could not inspect a current Repo Store") from error
            continue
        except BaseException as error:
            raise ManagedStoreError("state_store_unreadable", "could not inspect a current Repo Store") from error
        matches.append(store)
    if len(matches) != 1:
        raise ManagedStoreError("explicit_state_store_required", "multiple Stores require exactly one full exact StateRef match")
    return matches[0]


def _deduplicate_roots(stores: tuple[DirStore, ...]) -> tuple[DirStore, ...]:
    """Return one handle per canonical physical DirStore root in stable order."""

    selected = []
    seen = set()
    for store in stores:
        try:
            root = os.path.realpath(store.base_dir)
            stat_result = os.stat(root)
        except OSError as error:
            raise ManagedStoreError("state_store_unreadable", "could not inspect current Repo Store root") from error
        key = (root, stat_result.st_dev, stat_result.st_ino)
        if key not in seen:
            seen.add(key)
            selected.append(store)
    return tuple(selected)


def _state_store_root(store: DirStore) -> tuple[str, int, int]:
    """Return the canonical physical state-Store root identity for lock paths."""

    if type(store) is not DirStore:
        raise ManagedStoreError("invalid_state_store", "managed state locks require an exact DirStore")
    try:
        root = os.path.realpath(store.base_dir)
        evidence = os.stat(root)
    except OSError as error:
        raise ManagedStoreError("state_store_unreadable", "could not inspect the selected state Store root") from error
    if not stat.S_ISDIR(evidence.st_mode):
        raise ManagedStoreError("invalid_state_store", "selected state Store root is not a directory")
    return root, evidence.st_dev, evidence.st_ino


def _object_lock_digest(object_id: object) -> str:
    """Encode an exact ObjectId as the durable lock-file digest component."""

    if type(object_id) is not ObjectId:
        raise ManagedStoreError("invalid_object_id", "managed state locks require exact ObjectId values")
    try:
        payload = object_id.__stable_leaf_bytes__()
    except (AttributeError, TypeError, ValueError) as error:
        raise ManagedStoreError("invalid_object_id", "managed state locks require exact ObjectId values") from error
    if type(payload) is not bytes:
        raise ManagedStoreError("invalid_object_id", "managed ObjectId encoding must be bytes")
    return hashlib.sha256(payload).hexdigest()


def _ordered_object_ids(object_ids) -> tuple[object, ...]:
    """Deduplicate and sort ObjectIds by their full canonical lock digest."""

    pairs = {}
    for object_id in object_ids:
        pairs[_object_lock_digest(object_id)] = object_id
    return tuple(pairs[digest] for digest in sorted(pairs))


def _initialize_state_lock_namespace(state_store: DirStore) -> None:
    """Bootstrap a state-only managed lock namespace through the U4 gate protocol."""

    from .control import ManagedControlStore

    ManagedControlStore(state_store, state_store).initialize()


def _require_state_lock_namespace(state_store: DirStore) -> None:
    """Require an existing valid state lock gate without creating control authority."""

    from .control import ManagedControlStore

    adapter = ManagedControlStore(state_store, state_store)
    if not adapter._namespace_present():
        raise ManagedStoreError("missing_state_lock_namespace", "managed state lock namespace has not been initialized")


def _validate_pending_claim_stores(nodes: tuple[object, ...], state_store: DirStore) -> None:
    """Reject managed cross-Store initial claims before any reservation or save."""

    selected = _state_store_root(state_store)
    for node in nodes:
        leases = list(getattr(node, "_claim_leases", ()))
        lease = getattr(node, "_claim_lease", None)
        if lease is not None and lease not in leases:
            leases.append(lease)
        for claim in leases:
            claim_store = getattr(claim, "store", None)
            if claim_store is not None and _state_store_root(claim_store) != selected:
                raise ManagedStoreError("initial_claim_state_store_mismatch", "pending initial state claims require their declaration Store")


def _require_publication(store: DirStore, operation: str, *, local_state: bool) -> None:
    """Translate Store capability failures to a managed boundary error."""

    try:
        store.preflight_publication(operation, local_state=local_state)
    except BaseException as error:
        raise ManagedStoreError("store_capability_unavailable", f"{operation} is unavailable") from error


__all__ = [
    "ResolvedStores",
    "resolve_stores",
    "validate_state_ref",
]
