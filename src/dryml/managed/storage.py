"""Repo selection, topology evidence, and lifetime locks for managed work."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from contextlib import AbstractContextManager
from collections.abc import Mapping
from dataclasses import dataclass
from threading import get_ident

from dryml.core.reference_values import ObjectId, StateRef
from dryml.core.repo import Repo, RepoSaveError, _commit_save_report
from dryml.core.repo_plan import _unique_stores
from dryml.core.session import current_repo
from dryml.core.store.dir import DirStore
from dryml.core.store.store import Store
from dryml.core.store.zip import ZipStore
from dryml.formats import CanonicalJSONError, canonical_json_bytes, canonical_json_load_bytes
from dryml.locking import FileLock, LockError, interprocess_lock

from .errors import (
    ManagedConflictError,
    ManagedPublicationError,
    ManagedRecoveryError,
    ManagedStoreError,
)

_MAX_STORES = 256
_MAX_OBJECTS = 4096
_MAX_PAIRS = 65536


@dataclass(slots=True)
class ResolvedStores:
    """Selected state Repo and independent control authority.

    Args:
        state_repo: Borrowed connected Repo used for managed state authority.
        control_store: Exact ``DirStore`` holding lifecycle control authority.

    ``close()`` returns no value and releases only a one-Store Repo wrapper
    created for an explicit Store. Borrowed Repo and Store handles are never
    flushed or closed. Resolution owns no publication or topology lease and is
    safe only while callers retain the selected borrowed resources.
    """

    state_repo: Repo
    control_store: DirStore
    _wrapper: Repo | None = None

    def close(self) -> None:
        """Release a call-created wrapper without committing borrowed Stores.

        Returns:
            ``None``.

        Side Effects:
            Closes only the private wrapper created by :func:`resolve_stores`.
            The selected caller-owned Repo and Stores remain open and unmodified.

        Raises:
            BaseException: Propagates a private-wrapper cleanup failure after the
                wrapper has begun non-committing shutdown.
        """

        if self._wrapper is not None:
            wrapper, self._wrapper = self._wrapper, None
            wrapper.close(flush=False)


class _StateStoreLockLease(AbstractContextManager):
    """A process/thread-bound retained set of physical Store lock leases."""

    def __init__(self, object_ids, leases: tuple[FileLock, ...]):
        self.object_ids = object_ids
        self._leases = leases
        self._pid = os.getpid()
        self._thread_id = get_ident()
        self._active = True

    @property
    def active(self) -> bool:
        """Return whether the acquiring process/thread still owns this token."""

        return self._active and self._pid == os.getpid() and self._thread_id == get_ident()

    def __enter__(self):
        self.require_owner()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False

    def require_owner(self) -> None:
        """Reject inactive, inherited, and cross-thread token use."""

        if not self.active:
            raise ManagedConflictError("invalid_state_owner", "state lock ownership belongs to another process/thread or is inactive")

    def release(self) -> bool:
        """Release held locks in reverse order without unlinking shared files."""

        if not self.active:
            return False
        self._active = False
        error = None
        for lease in reversed(self._leases):
            try:
                lease.release()
            except LockError as caught:
                error = error or caught
        if error is not None:
            raise ManagedStoreError("state_lock_release_failed", "could not release managed state ownership") from error
        return True


class _ManagedStateOwnership(AbstractContextManager):
    """Combine core graph/topology reservations and physical state locks."""

    def __init__(self, reservation, topology, state_locks, ownership):
        self.reservation = reservation
        self.topology = topology
        self.state_locks = state_locks
        self.object_ids = state_locks.object_ids
        self.ownership = ownership
        self._pid = os.getpid()
        self._thread_id = get_ident()
        self._active = True

    @property
    def active(self) -> bool:
        """Return whether every joined ownership layer remains valid."""

        return self._active and self._pid == os.getpid() and self._thread_id == get_ident() and self.reservation.active and self.state_locks.active

    def __enter__(self):
        self.require_owner()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False

    def require_owner(self) -> None:
        """Require use by the exact process/thread that acquired ownership."""

        if not self.active:
            raise ManagedConflictError("invalid_state_owner", "managed state ownership belongs to another process/thread or is inactive")

    def release(self) -> bool:
        """Release locks, graph reservation, and topology lease in safe order."""

        if not self.active:
            return False
        self._active = False
        try:
            self.state_locks.release()
        finally:
            try:
                self.reservation.release()
            finally:
                self.topology.__exit__(None, None, None)
        return True


def resolve_stores(obj: object, *, state_repo: Repo | Store | None = None,
                    control_store: DirStore | None = None,
                    require_writable: bool = True) -> ResolvedStores:
    """Resolve managed Repo/control authority without state or control mutation.

    Omitted state authority is exactly the configured current Repo. An explicit
    Store is wrapped for this call only; it never becomes a session default.

    Args:
        obj: Managed receiver retained for API symmetry; it is not materialized or
            mutated during resolution.
        state_repo: Borrowed Repo or Store, or ``None`` for the configured Repo.
        control_store: Exact control ``DirStore``, or ``None`` for the Repo default.
        require_writable: Whether selected state/control Stores must preflight
            publication capability.

    Returns:
        Selected state/control authority and any private one-Store wrapper.

    Raises:
        ManagedStoreError: If authority is absent, malformed, unsupported, or
            cannot meet requested publication capability.

    Side Effects:
        May construct a private non-owning Repo wrapper for an explicit Store.
        It never writes, commits, closes, or changes caller-owned resources.
    """

    if state_repo is None:
        repo = current_repo()
        if repo is None:
            raise ManagedStoreError("explicit_state_repo_required", "no current Repo; supply state_repo explicitly")
        wrapper = None
    elif isinstance(state_repo, Repo):
        repo, wrapper = state_repo, None
    elif isinstance(state_repo, Store):
        try:
            repo = Repo._for_state_io((state_repo,))
        except BaseException as error:
            raise ManagedStoreError("invalid_state_repo", "could not wrap the supplied state Store") from error
        wrapper = repo
    else:
        raise ManagedStoreError("invalid_state_repo", "state_repo must be a Repo or Store")
    try:
        selected_control = repo.default_store if control_store is None else control_store
        if type(selected_control) is not DirStore:
            raise ManagedStoreError("invalid_control_store", "control_store must be an exact DirStore")
        stores = _physical_stores(repo)
        if require_writable:
            for store in stores:
                _require_publication(store, "managed state publication", local_state=True)
            _require_publication(selected_control, "managed control publication", local_state=False)
        return ResolvedStores(repo, selected_control, wrapper)
    except BaseException:
        if wrapper is not None:
            wrapper.close(flush=False)
        raise


def validate_state_ref(state_repo: Repo, state_ref: StateRef) -> StateRef:
    """Validate an exact StateRef closure across the selected connected Repo.

    Args:
        state_repo: Borrowed Repo supplying all exact recovery authority.
        state_ref: Exact StateRef whose full closure must be available.

    Returns:
        The unchanged validated ``state_ref``.

    Raises:
        ManagedRecoveryError: If types are invalid or the exact closure is absent.

    Side Effects:
        Reads Store authority only; it creates, commits, and changes no state.
    """

    if not isinstance(state_repo, Repo) or type(state_ref) is not StateRef:
        raise ManagedRecoveryError("invalid_state_reference", "managed state validation requires a Repo and exact StateRef")
    try:
        from dryml.core.materialization import build_exact_state_load_plan

        build_exact_state_load_plan(state_repo, state_ref)
    except BaseException as error:
        raise ManagedRecoveryError("missing_state_reference", "state Repo lacks the exact StateRef closure") from error
    return state_ref


def publish_managed_state(state_repo: Repo, obj: object, *, reservation):
    """Deep-publish and durably verify one managed state boundary.

    The normal Repo save engine remains the only router and publication ledger.
    This wrapper commits only the exact replica and recovery Stores named by its
    report, then proves the complete root/descendant closure from freshly reopened
    buffered archives before lifecycle control may associate the receipt.

    Args:
        state_repo: Borrowed selected Repo whose current routing policy applies.
        obj: Managed live graph root covered by ``reservation``.
        reservation: Active exact graph reservation retained for the invocation.

    Returns:
        ``(StateRef, StoreReport)`` after required buffered Stores are durable.

    Raises:
        RepoSaveError: If core publication, bounded commit, or durable exact
            validation fails. Its cause chain includes ManagedPublicationError.

    Side Effects:
        Captures state through current routing and commits only dirty required
        buffered Stores. Borrowed Repos and Stores remain open.
    """

    report = None
    try:
        state_ref, report = state_repo.save_object(
            obj, deep_capture=True, reservation=reservation, report_stores=True,
        )
        required = _required_managed_publication_stores(report)
        report = _commit_save_report(
            state_repo, state_ref, report, stores=required, dirty_only=True,
        )
        _validate_managed_publication(state_repo, state_ref, report, required)
        return state_ref, report
    except RepoSaveError as error:
        _raise_managed_publication_error(error, error.report or report)
    except BaseException as error:
        report = getattr(error, "report", None) or report
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            if report is not None:
                error.report = report
            raise
        if report is None:
            raise
        _raise_managed_publication_error(error, report)


def _required_managed_publication_stores(report) -> tuple[Store, ...]:
    """Return ordered replica and sufficient-recovery Stores from one core report."""

    return _unique_stores((
        *(
            store
            for snapshot in report.snapshots
            for store in (*snapshot.stores, *snapshot.required_stores)
        ),
        *report.required_stores,
    ))


def _validate_managed_publication(state_repo: Repo, state_ref: StateRef, report, required) -> None:
    """Require complete matching snapshots and durable exact recovery authority."""

    if any(
            publication.status != "completed"
            for publication in report.publications
            if publication.phase in {"definition", "state", "snapshot", "membership", "claim", "commit"}
    ):
        raise ManagedRecoveryError("incomplete_state_publication", "managed state report has incomplete authority")
    for snapshot in report.snapshots:
        if state_ref_for_digest(state_repo, snapshot.state_ref.digest()) != snapshot.state_ref:
            raise ManagedRecoveryError("conflicting_state_reference", "managed replica StateRef authority conflicts")
        for store in snapshot.stores:
            record = store.read_state_ref_record(snapshot.state_ref.digest())
            if record is None or record.state_ref != snapshot.state_ref:
                raise ManagedRecoveryError("missing_state_reference", "managed replica StateRef authority is incomplete")
    validate_state_ref(state_repo, state_ref)
    _validate_reopened_managed_state(state_repo, state_ref, report, required)


def _validate_reopened_managed_state(state_repo: Repo, state_ref: StateRef, report, required) -> None:
    """Prove the bounded required authority survives fresh buffered archive views.

    Only Stores selected by the save report participate.  This prevents unrelated
    live authority from satisfying a required archive replica or dependency.
    """

    reopened = {}
    try:
        for store in required:
            if type(store) is ZipStore:
                reopened[id(store)] = ZipStore.open_existing(store.archive_path)
        durable = Repo._for_state_io(tuple(
            reopened.get(id(store), store) for store in required
        ))
        try:
            for snapshot in report.snapshots:
                for store in snapshot.stores:
                    durable_store = reopened.get(id(store), store)
                    record = durable_store.read_state_ref_record(snapshot.state_ref.digest())
                    if record is None or record.state_ref != snapshot.state_ref:
                        raise ManagedRecoveryError(
                            "missing_state_reference",
                            "reopened managed replica authority is incomplete",
                        )
                # Exact replica records prove selected publication; the root's exact
                # closure covers every projected descendant and materializing seed.
            validate_state_ref(durable, state_ref)
        finally:
            durable.close(flush=False)
    finally:
        for store in reversed(tuple(reopened.values())):
            store.close()


def _raise_managed_publication_error(error: BaseException, report) -> None:
    """Expose core report evidence through a classified managed publication cause."""

    outcome = "indeterminate" if report is not None and any(
        publication.phase == "commit" and publication.status == "uncertain"
        for publication in report.publications
    ) else "not_committed"
    managed = ManagedPublicationError(outcome, "managed state publication was not durable")
    managed.__cause__ = error
    raise RepoSaveError("managed state publication failed", report=report) from managed


def state_ref_for_digest(state_repo: Repo, digest: str) -> StateRef:
    """Return one identical retained StateRef record from a connected Repo.

    Args:
        state_repo: Borrowed Repo searched for retained exact StateRef records.
        digest: Lowercase StateRef digest to resolve.

    Returns:
        The one consistent retained StateRef value.

    Raises:
        ManagedRecoveryError: If records are missing, malformed, or disagree.

    Side Effects:
        Reads connected Stores only and never creates or changes authority.
    """

    matches = []
    for store in state_repo.stores:
        record = store.read_state_ref_record(digest)
        if record is not None:
            if type(record.state_ref) is not StateRef or record.state_ref.digest() != digest:
                raise ManagedRecoveryError("invalid_state_reference", "state Repo has malformed retained StateRef authority")
            matches.append(record.state_ref)
    if not matches or any(item != matches[0] for item in matches[1:]):
        raise ManagedRecoveryError("missing_state_reference", "state Repo lacks one consistent retained StateRef")
    return matches[0]


def ownership_evidence(repo: Repo, object_ids) -> dict[str, object]:
    """Return closed hashed physical topology evidence for one state lock set.

    Args:
        repo: Borrowed Repo whose connected physical Store set is captured.
        object_ids: Stateful exact ObjectIds included in the managed graph.

    Returns:
        A v1 mapping of sorted unique SHA-256 Store and ObjectId lock keys.

    Raises:
        ManagedStoreError: If the Repo, Store topology, ObjectIds, or size bounds
            cannot support managed ownership.

    Side Effects:
        Inspects persistent Store identity only; it does not create namespaces or
        acquire locks.
    """

    stores = _physical_stores(repo)
    objects = _ordered_object_ids(object_ids)
    if not objects:
        raise ManagedStoreError("stateful_graph_required", "managed ownership requires at least one stateful ObjectId")
    if len(stores) > _MAX_STORES or len(objects) > _MAX_OBJECTS or len(stores) * len(objects) > _MAX_PAIRS:
        raise ManagedStoreError("ownership_bounds", "managed ownership topology exceeds its supported bounds")
    return {
        "version": 1,
        "store_keys": sorted(_store_key(store) for store in stores),
        "object_keys": sorted(_object_lock_digest(object_id) for object_id in objects),
    }


def _physical_stores(repo: Repo) -> tuple[Store, ...]:
    """Return one supported handle per physical Store in stable connected order."""

    if not isinstance(repo, Repo):
        raise ManagedStoreError("invalid_state_repo", "managed state authority requires a Repo")
    selected = []
    seen = set()
    for store in tuple(repo.stores):
        key = _physical_store_key(store)
        if key in seen:
            continue
        seen.add(key)
        selected.append(store)
    if not selected:
        raise ManagedStoreError("state_repo_empty", "state_repo has no connected Stores")
    return tuple(selected)


def _physical_store_key(store: Store):
    """Return one supported canonical physical Store identity."""

    try:
        key = Repo._physical_store_key(store)
    except (OSError, ValueError) as error:
        raise ManagedStoreError("state_store_unreadable", "could not inspect a managed state Store") from error
    if key is None:
        raise ManagedStoreError("unsupported_state_store", "managed ownership requires a direct DirStore or path-backed ZipStore")
    return key


def _store_key(store: Store) -> str:
    """Hash canonical physical Store identity without serializing its location."""

    key = _physical_store_key(store)
    payload = b"\0".join(str(item).encode("utf-8") for item in key)
    return hashlib.sha256(payload).hexdigest()


def _object_lock_digest(object_id: object) -> str:
    """Return the SHA-256 lock namespace key for one exact ObjectId."""

    if type(object_id) is not ObjectId:
        raise ManagedStoreError("invalid_object_id", "managed state locks require exact ObjectId values")
    try:
        payload = object_id.__stable_leaf_bytes__()
    except (AttributeError, TypeError, ValueError) as error:
        raise ManagedStoreError("invalid_object_id", "managed state locks require exact ObjectId values") from error
    if type(payload) is not bytes:
        raise ManagedStoreError("invalid_object_id", "managed ObjectId encoding must be bytes")
    return hashlib.sha256(payload).hexdigest()


def _ordered_object_ids(object_ids) -> tuple[ObjectId, ...]:
    """Deduplicate exact ObjectIds in deterministic lock-name order."""

    pairs = {_object_lock_digest(object_id): object_id for object_id in object_ids}
    return tuple(pairs[key] for key in sorted(pairs))


def _state_lock_path(store: Store, object_id: object) -> str:
    """Return the canonical external lifetime lock path for one Store/ObjectId."""

    return _state_lock_path_for_key(store, _object_lock_digest(object_id))


def _state_lock_path_for_key(store: Store, digest: str) -> str:
    """Return one canonical lifetime lock path for a retained ObjectId digest."""

    return os.path.join(_state_lock_namespace(store), digest[:2], digest + ".lock")


def _state_lock_root(store: Store) -> str:
    """Return the persistent managed namespace root after Store identity checks."""

    if type(store) is DirStore:
        root, _, _ = _dir_root(store)
        return os.path.join(root, "managed")
    if type(store) is ZipStore:
        archive = store.archive_path
        if archive is None:
            raise ManagedStoreError("unsupported_state_store", "managed ownership requires a path-backed ZipStore")
        parent = os.path.dirname(os.path.normcase(os.path.realpath(archive))) or "."
        try:
            evidence = os.stat(parent)
        except OSError as error:
            raise ManagedStoreError("state_store_unreadable", "could not inspect a managed ZipStore parent") from error
        if not stat.S_ISDIR(evidence.st_mode):
            raise ManagedStoreError("invalid_state_store", "managed ZipStore parent is not a directory")
        return f"{os.path.normcase(os.path.realpath(archive))}.dryml-managed"
    raise ManagedStoreError("unsupported_state_store", "managed ownership requires a direct DirStore or path-backed ZipStore")


def _state_lock_namespace(store: Store) -> str:
    """Return the v1 lifetime-lock namespace without creating it."""

    return os.path.join(_state_lock_root(store), "locks", "v1")


def _dir_root(store: DirStore):
    """Validate and return canonical direct-Store root evidence."""

    try:
        root = os.path.normcase(os.path.realpath(store.base_dir))
        evidence = os.stat(root)
    except OSError as error:
        raise ManagedStoreError("state_store_unreadable", "could not inspect a managed state Store root") from error
    if not stat.S_ISDIR(evidence.st_mode):
        raise ManagedStoreError("invalid_state_store", "managed state Store root is not a directory")
    return root, evidence.st_dev, evidence.st_ino


def _bootstrap_state_lock_namespaces(stores: tuple[Store, ...], ownership: Mapping[str, object]) -> None:
    """Create valid v1 namespaces only while admitting a new state owner."""

    expected_keys = _canonical_ownership(ownership)["store_keys"]
    for store, expected_key in zip(stores, expected_keys, strict=True):
        if _store_key(store) != expected_key:
            raise ManagedRecoveryError("ownership_store_changed", "state Store identity changed during ownership admission")
        try:
            _bootstrap_state_lock_namespace(store)
        except (ManagedRecoveryError, ManagedStoreError):
            raise
        except (LockError, OSError) as error:
            raise ManagedStoreError("state_lock_namespace_bootstrap_failed", "could not create managed state lock namespace") from error
        _validate_state_lock_namespace(store, expected_key)


def _bootstrap_state_lock_namespace(store: Store) -> None:
    """Publish or extend one Store's shared managed namespace under its stable lock."""

    if type(store) is DirStore:
        _bootstrap_managed_namespace(store, prepare=_prepare_state_lock_namespace)
        return
    root = _state_lock_root(store)
    with interprocess_lock(_state_namespace_bootstrap_lock_path(store)):
        _publish_managed_namespace(root, prepare=_prepare_state_lock_namespace)


def _bootstrap_managed_namespace(store: DirStore, *, prepare=None, sync_directory=None) -> None:
    """Use a direct Store writer lock to atomically publish its shared namespace.

    ``prepare`` may add non-control hierarchy inside a new staging directory or an
    already validated namespace while the same canonical writer lock is retained.
    ``sync_directory`` preserves control publication's existing durability seam;
    state-lock bootstrap makes no additional directory-fsync promise.
    """

    root = _state_lock_root(store)
    with store.writer_lock():
        _publish_managed_namespace(root, prepare=prepare, sync_directory=sync_directory)


def _state_namespace_bootstrap_lock_path(store: Store) -> str:
    """Return a Zip sibling bootstrap lock distinct from its replaceable namespace."""

    return _state_lock_root(store) + ".bootstrap.lock"


def _publish_managed_namespace(root: str, *, prepare=None, sync_directory=None) -> None:
    """Install a complete namespace or extend an existing valid one without replacement."""

    try:
        os.lstat(root)
    except FileNotFoundError:
        parent = os.path.dirname(root)
        staging = tempfile.mkdtemp(prefix=".managed-staging-", dir=parent)
        try:
            _write_state_namespace_format(os.path.join(staging, "format.json"))
            if prepare is not None:
                prepare(staging)
            if sync_directory is not None:
                sync_directory(staging)
            os.rename(staging, root)
            if sync_directory is not None:
                sync_directory(parent)
        except BaseException:
            _remove_unpublished_namespace_staging(staging)
            raise
        return
    except OSError:
        raise
    _require_directory(root, "managed state lock namespace")
    _validate_state_namespace_format(root)
    if prepare is not None:
        prepare(root)


def _prepare_state_lock_namespace(root: str) -> None:
    """Create the complete versioned lifetime-lock hierarchy before publication."""

    try:
        os.makedirs(os.path.join(root, "locks", "v1"), mode=0o700, exist_ok=True)
    except OSError as error:
        raise ManagedStoreError("state_lock_namespace_bootstrap_failed", "could not create managed state lock namespace") from error


def _remove_unpublished_namespace_staging(path: str) -> None:
    """Remove only this call's never-published namespace staging directory."""

    try:
        shutil.rmtree(path)
    except OSError:
        pass


def _validate_state_lock_namespaces(stores: tuple[Store, ...], ownership: Mapping[str, object]) -> None:
    """Require retained lock namespaces without creating recovery evidence."""

    expected_keys = _canonical_ownership(ownership)["store_keys"]
    for store, expected_key in zip(stores, expected_keys, strict=True):
        _validate_state_lock_namespace(store, expected_key)


def _validate_state_lock_namespace(store: Store, expected_key: str) -> None:
    """Validate one stable, versioned lock namespace for a read-only probe."""

    if _store_key(store) != expected_key:
        raise ManagedRecoveryError("ownership_store_changed", "state Store identity changed during owner probe")
    root = _state_lock_root(store)
    _require_directory(root, "managed state lock namespace")
    _validate_state_namespace_format(root)
    _require_directory(os.path.join(root, "locks"), "managed state lock hierarchy")
    _require_directory(os.path.join(root, "locks", "v1"), "managed state lock version namespace")
    if _store_key(store) != expected_key:
        raise ManagedRecoveryError("ownership_store_changed", "state Store identity changed during owner probe")


def _require_directory(path: str, label: str) -> None:
    """Require one non-symlink directory without treating absence as available."""

    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as error:
        raise ManagedRecoveryError("state_lock_namespace_missing", f"{label} is absent") from error
    except OSError as error:
        raise ManagedStoreError("state_lock_namespace_unavailable", f"could not inspect {label}") from error
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise ManagedRecoveryError("invalid_state_lock_namespace", f"{label} is not a directory")


def _write_state_namespace_format(path: str) -> None:
    """Write one complete shared format gate into call-owned staging authority."""

    payload = canonical_json_bytes(
        {"schema": "dryml-managed", "version": 1},
        max_depth=1, max_nodes=4, max_entries=2, max_string=64, max_int_bits=8,
    )
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ManagedStoreError("state_lock_namespace_bootstrap_failed", "managed state namespace staging already has a format gate") from error
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(fd: int, payload: bytes) -> None:
    """Write a bounded format payload fully or fail before its staging is published."""

    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("managed state namespace format write was incomplete")
        offset += written


def _validate_state_namespace_format(root: str) -> None:
    """Require the exact shared managed format gate without bootstrapping it."""

    path = os.path.join(root, "format.json")
    try:
        mode = os.lstat(path).st_mode
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            raise ManagedRecoveryError("invalid_state_lock_namespace", "managed state namespace format is not a regular file")
        with open(path, "rb") as stream:
            payload = stream.read(4096)
    except ManagedRecoveryError:
        raise
    except FileNotFoundError as error:
        raise ManagedRecoveryError("state_lock_namespace_missing", "managed state namespace format is absent") from error
    except OSError as error:
        raise ManagedStoreError("state_lock_namespace_unavailable", "could not read managed state namespace format") from error
    try:
        data = canonical_json_load_bytes(payload, max_depth=1, max_nodes=4, max_entries=2, max_string=64, max_int_bits=8)
    except CanonicalJSONError as error:
        raise ManagedRecoveryError("invalid_state_lock_namespace", "managed state namespace format is malformed") from error
    if (
            not hasattr(data, "keys")
            or set(data) != {"schema", "version"}
            or data["schema"] != "dryml-managed"
            or type(data["version"]) is not int
            or data["version"] != 1):
        raise ManagedRecoveryError("invalid_state_lock_namespace", "managed state namespace format is unsupported")


def _require_existing_lock_paths(paths) -> None:
    """Require every retained lock file before a non-creating availability probe."""

    for path, _, _ in paths:
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError as error:
            raise ManagedRecoveryError("state_lock_evidence_missing", "retained managed state lock evidence is absent") from error
        except OSError as error:
            raise ManagedStoreError("state_lock_unavailable", "could not inspect retained managed state lock evidence") from error
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            raise ManagedRecoveryError("invalid_state_lock_namespace", "retained managed state lock evidence is not a regular file")


def _acquire_state_locks(repo: Repo, object_ids, *, ownership=None) -> _StateStoreLockLease:
    """Acquire every physical Store/ObjectId lock nonblockingly and atomically."""

    ordered_objects = _ordered_object_ids(object_ids)
    evidence = ownership_evidence(repo, ordered_objects) if ownership is None else _canonical_ownership(ownership)
    if tuple(_object_lock_digest(object_id) for object_id in ordered_objects) != tuple(evidence["object_keys"]):
        raise ManagedRecoveryError("ownership_mismatch", "captured ObjectIds do not match retained ownership evidence")
    stores = _stores_for_ownership(repo, evidence)
    _bootstrap_state_lock_namespaces(stores, evidence)
    return _acquire_lock_paths(
        ((_state_lock_path_for_key(store, object_key), store, object_key)
         for store in stores for object_key in evidence["object_keys"]),
        ordered_objects,
        create=True,
    )


def _acquire_lock_paths(paths, object_ids, *, create: bool) -> _StateStoreLockLease:
    """Acquire a supplied deterministic lock-path set or release every partial lease."""

    paths = sorted(paths, key=lambda item: item[0])
    if not create:
        _require_existing_lock_paths(paths)
    leases = []
    try:
        for path, _, _ in paths:
            if create:
                os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
            lease = FileLock(path)
            if not lease.acquire(blocking=False):
                raise ManagedConflictError("state_lock_conflict", "managed state lock is already held")
            leases.append(lease)
    except BaseException as error:
        cleanup_error = None
        for lease in reversed(leases):
            try:
                lease.release()
            except LockError as caught:
                cleanup_error = cleanup_error or caught
        if cleanup_error is not None:
            raise ManagedStoreError("state_lock_release_failed", "could not release partial managed state ownership") from cleanup_error
        if isinstance(error, ManagedConflictError):
            raise
        if isinstance(error, (LockError, OSError)):
            raise ManagedStoreError("state_lock_unavailable", "could not acquire managed state ownership") from error
        raise
    return _StateStoreLockLease(tuple(object_ids), tuple(leases))


def _acquire_state_ownership(repo: Repo, obj: object, *, expected_ownership=None) -> _ManagedStateOwnership:
    """Retain core graph/topology claims and every physical state lock pair."""

    if not isinstance(repo, Repo):
        raise TypeError("managed state ownership requires a Repo")
    topology = repo.retain_topology()
    topology.__enter__()
    try:
        try:
            plan, nodes, object_ids = repo._state_graph_evidence(obj)
        except RepoSaveError as error:
            raise ManagedRecoveryError("invalid_target", "managed target requires a fresh exact load") from error
        evidence = ownership_evidence(repo, object_ids)
        if expected_ownership is not None and _canonical_ownership(evidence) != _canonical_ownership(expected_ownership):
            raise ManagedRecoveryError("ownership_mismatch", "live receiver topology does not match retained running ownership")
        _validate_pending_claim_stores(nodes, repo)
        try:
            reservation = repo._reserve_state_graph_evidence(plan, nodes, object_ids)
        except RepoSaveError as error:
            raise ManagedConflictError("state_graph_conflict", str(error)) from error
        try:
            state_locks = _acquire_state_locks(repo, object_ids, ownership=evidence)
        except BaseException:
            reservation.release()
            raise
    except BaseException:
        topology.__exit__(None, None, None)
        raise
    return _ManagedStateOwnership(reservation, topology, state_locks, evidence)


def _probe_state_ownership(repo: Repo, ownership: Mapping[str, object]) -> bool:
    """Probe exactly retained lock pairs; contention remains inconclusive.

    Current control authority contains hashes rather than reconstructable ObjectIds.
    A probe therefore maps its retained Store keys to the supplied Repo and uses the
    retained Object keys directly for filenames.  It must not infer a smaller graph
    from the caller's current receiver before declaring an owner absent.
    """

    stores = _stores_for_ownership(repo, ownership)
    evidence = _canonical_ownership(ownership)
    object_keys = evidence["object_keys"]
    _validate_state_lock_namespaces(stores, evidence)
    try:
        lease = _acquire_lock_paths(
            ((_state_lock_path_for_key(store, object_key), store, object_key)
             for store in stores for object_key in object_keys),
            (),
            create=False,
        )
    except ManagedConflictError:
        return False
    lease.release()
    return True


def _stores_for_ownership(repo: Repo, ownership: Mapping[str, object]) -> tuple[Store, ...]:
    """Map every retained hashed Store key to one physical connected Store."""

    expected = _canonical_ownership(ownership)["store_keys"]
    available = {_store_key(store): store for store in _physical_stores(repo)}
    if any(key not in available for key in expected):
        raise ManagedRecoveryError("ownership_store_missing", "state_repo does not map every retained ownership key")
    return tuple(available[key] for key in expected)


def _canonical_ownership(ownership: Mapping[str, object]) -> dict[str, object]:
    """Validate all bounded v1 ownership fields before any lock side effects."""

    if (
            not isinstance(ownership, Mapping)
            or set(ownership) != {"version", "store_keys", "object_keys"}
            or type(ownership.get("version")) is not int
            or ownership["version"] != 1):
        raise ManagedRecoveryError("invalid_ownership", "retained managed ownership evidence is malformed")
    stores = ownership["store_keys"]
    objects = ownership["object_keys"]
    if type(stores) not in (list, tuple) or type(objects) not in (list, tuple):
        raise ManagedRecoveryError("invalid_ownership", "retained managed ownership keys are malformed")
    if not stores or not objects or len(stores) > _MAX_STORES or len(objects) > _MAX_OBJECTS or len(stores) * len(objects) > _MAX_PAIRS:
        raise ManagedRecoveryError("invalid_ownership", "retained managed ownership topology is outside supported bounds")
    store_keys, object_keys = tuple(stores), tuple(objects)
    if any(type(key) is not str or len(key) != 64 or any(char not in "0123456789abcdef" for char in key) for key in (*store_keys, *object_keys)):
        raise ManagedRecoveryError("invalid_ownership", "retained managed ownership keys are malformed")
    if store_keys != tuple(sorted(set(store_keys))) or object_keys != tuple(sorted(set(object_keys))):
        raise ManagedRecoveryError("invalid_ownership", "retained managed ownership keys are malformed")
    return {"version": 1, "store_keys": store_keys, "object_keys": object_keys}


def _validate_pending_claim_stores(nodes: tuple[object, ...], repo: Repo) -> None:
    """Reject pending claims whose physical Store is outside retained topology."""

    connected = {_physical_store_key(store) for store in _physical_stores(repo)}
    for node in nodes:
        leases = list(getattr(node, "_claim_leases", ()))
        lease = getattr(node, "_claim_lease", None)
        if lease is not None and lease not in leases:
            leases.append(lease)
        for claim in leases:
            if claim.store is not None and _physical_store_key(claim.store) not in connected:
                raise ManagedStoreError("initial_claim_state_store_mismatch", "pending initial state claims require a connected declaration Store")


def _require_publication(store: Store, operation: str, *, local_state: bool) -> None:
    """Translate Store capability failure at the managed boundary."""

    try:
        store.preflight_publication(operation, local_state=local_state)
    except BaseException as error:
        raise ManagedStoreError("store_capability_unavailable", f"{operation} is unavailable") from error


__all__ = ["ResolvedStores", "ownership_evidence", "publish_managed_state", "resolve_stores", "state_ref_for_digest", "validate_state_ref"]
