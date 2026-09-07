"""Dependency-light cross-platform advisory locks for cooperating processes.

This module is DRYML's sole owner of native advisory-lock APIs.  It coordinates
same-process leases before delegating to POSIX ``flock`` or Windows byte-range
locking; consumers retain their own paths, lifetime, and recovery policies.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import errno
import os
import stat
import threading
from weakref import WeakSet

try:
    import fcntl
except ImportError:  # pragma: no cover - selected on Windows hosts.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - selected on POSIX hosts.
    msvcrt = None


class LockError(OSError):
    """Raised when an advisory-lock operation cannot be completed.

    This error wraps native adapter, descriptor, and lock-file failures.  A
    nonblocking request returns ``False`` only for recognized lock contention;
    callers can therefore distinguish unavailable locking from a busy peer.
    """


class LockUnavailableError(LockError):
    """Raised when this platform does not expose a supported lock primitive.

    Callers may translate this failure to a domain-specific unsupported-backend
    error.  The error does not imply that another process owns a lock.
    """


class _PathState:
    """Coordinate local leases without replacing the native interprocess lock."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.exclusive_owner: int | None = None
        self.pending_exclusive: int | None = None
        self.shared_owners: dict[int, int] = defaultdict(int)


_PATH_STATES: dict[str, _PathState] = {}
_PATH_STATES_GUARD = threading.Lock()
_ACTIVE_FILE_LOCKS: WeakSet[FileLock] = WeakSet()
_ACTIVE_FILE_LOCKS_GUARD = threading.Lock()
_LOCK_STATE = threading.local()


def supports_advisory_locking(path: str | os.PathLike[str]) -> bool:
    """Return whether ``path`` has a local directory shape for this adapter.

    Args:
        path: Existing or prospective lock-file path.

    Returns:
        ``True`` when the direct parent is a normal non-symlink directory.

    This narrow check is not proof that the backing filesystem provides correct
    advisory-lock behavior.  Consumers must still handle :class:`LockError`.
    """

    directory = os.path.dirname(os.path.abspath(os.fspath(path))) or "."
    try:
        mode = os.lstat(directory).st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(mode) and not stat.S_ISLNK(mode)


def _is_posix_contention(error: OSError) -> bool:
    """Return whether a POSIX ``flock`` failure denotes a held peer lease."""

    return error.errno in {errno.EAGAIN, errno.EWOULDBLOCK}


def _is_windows_contention(error: OSError) -> bool:
    """Return whether a Windows byte-range lock failure denotes contention."""

    winerror = getattr(error, "winerror", None)
    if winerror is not None:
        return winerror in {32, 33}
    return error.errno == errno.EACCES


def _lock_file(fd: int, *, shared: bool, blocking: bool) -> bool:
    """Apply one native lock without taking ownership of ``fd``."""

    if fcntl is not None:
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, operation)
        except OSError as error:
            if not blocking and _is_posix_contention(error):
                return False
            raise LockError(f"POSIX advisory lock failed: {error}") from error
        return True

    if msvcrt is None:
        raise LockUnavailableError("No supported advisory-lock primitive is available on this platform.")
    try:
        if os.fstat(fd).st_size == 0:
            if os.write(fd, b"\0") != 1:
                raise OSError("could not initialize the Windows lock byte")
        os.lseek(fd, 0, os.SEEK_SET)
    except OSError as error:
        raise LockError(f"Windows advisory lock preparation failed: {error}") from error
    try:
        operation = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(fd, operation, 1)
    except OSError as error:
        if not blocking and _is_windows_contention(error):
            return False
        raise LockError(f"Windows advisory lock failed: {error}") from error
    return True


def try_lock_file(fd: int, *, shared: bool = False) -> bool:
    """Attempt a nonblocking advisory lock on a caller-owned descriptor.

    Args:
        fd: Open file descriptor whose lock lifetime the caller owns.
        shared: Request a shared reader lock where POSIX supports it.  Windows
            uses its compatible exclusive byte-range fallback.

    Returns:
        ``True`` when the lock was acquired, or ``False`` only when another
        cooperating owner holds an incompatible lock.

    Raises:
        LockUnavailableError: No supported native locking primitive is present.
        LockError: The descriptor or native adapter failed for another reason.

    This function never opens, closes, unlinks, or otherwise takes ownership of
    the descriptor or its path.
    """

    return _lock_file(fd, shared=shared, blocking=False)


def unlock_file(fd: int) -> None:
    """Release an advisory lock on a caller-owned descriptor.

    Args:
        fd: Open descriptor previously locked through this module.

    Raises:
        LockUnavailableError: No supported native locking primitive is present.
        LockError: The native unlock operation fails.

    The descriptor remains open and its path is never unlinked.
    """

    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as error:
            raise LockError(f"POSIX advisory unlock failed: {error}") from error
        return
    if msvcrt is None:
        raise LockUnavailableError("No supported advisory-lock primitive is available on this platform.")
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError as error:
        raise LockError(f"Windows advisory unlock failed: {error}") from error


def _path_state(path: str) -> _PathState:
    """Return the process-local coordinator for one normalized path."""

    with _PATH_STATES_GUARD:
        return _PATH_STATES.setdefault(path, _PathState())


def _reserve_slot(path: str, *, shared: bool, blocking: bool) -> tuple[_PathState, bool] | None:
    """Reserve local compatibility before a file descriptor enters native locking."""

    state = _path_state(path)
    thread_id = threading.get_ident()
    effective_shared = shared and fcntl is not None
    with state.condition:
        while True:
            own_exclusive = state.exclusive_owner == thread_id
            own_shared = state.shared_owners.get(thread_id, 0) > 0
            if effective_shared:
                if own_exclusive:
                    if blocking:
                        raise LockError("A separate FileLock on this thread already owns the path.")
                    return None
                available = state.exclusive_owner is None and state.pending_exclusive is None
                if available:
                    state.shared_owners[thread_id] += 1
                    return state, True
            else:
                if own_exclusive or own_shared:
                    if blocking:
                        raise LockError("A separate FileLock on this thread already owns the path.")
                    return None
                available = (
                    state.exclusive_owner is None
                    and not state.shared_owners
                    and state.pending_exclusive is None
                )
                if available:
                    state.pending_exclusive = thread_id
                    return state, False
            if not blocking:
                return None
            state.condition.wait()


def _release_slot(state: _PathState, *, shared: bool) -> None:
    """Release one local reservation after native release or acquisition failure."""

    thread_id = threading.get_ident()
    with state.condition:
        if shared:
            state.shared_owners[thread_id] -= 1
            if state.shared_owners[thread_id] == 0:
                del state.shared_owners[thread_id]
        else:
            if state.pending_exclusive == thread_id:
                state.pending_exclusive = None
            elif state.exclusive_owner == thread_id:
                state.exclusive_owner = None
        state.condition.notify_all()


def _activate_slot(state: _PathState) -> None:
    """Mark a successfully native-locked exclusive reservation as active."""

    with state.condition:
        state.exclusive_owner = state.pending_exclusive
        state.pending_exclusive = None


class FileLock:
    """Retain one non-reentrant advisory lease for a lock-file path.

    Args:
        path: Existing or prospective durable lock-file path.
        shared: Request a shared reader lease on POSIX.  Windows uses an
            exclusive fallback because its byte-range adapter has no equivalent
            shared mode.

    The instance owns only the file descriptor it opens.  It never unlinks the
    path, may be released repeatedly, and cannot be acquired again until a
    prior successful acquisition has been released. Acquisition and release
    must occur on the same thread; cross-thread release raises :class:`LockError`
    without changing the owner's lease.
    """

    def __init__(self, path: str | os.PathLike[str], *, shared: bool = False) -> None:
        self.path = os.path.abspath(os.fspath(path))
        self.shared = shared
        self._file = None
        self._state: _PathState | None = None
        self._effective_shared = False
        self._owner_thread: int | None = None
        self._operation_guard = threading.Lock()

    def acquire(self, *, blocking: bool = True) -> bool:
        """Acquire this lease and retain its descriptor until :meth:`release`.

        Args:
            blocking: Wait for compatible owners when ``True``.  A false value
                performs one nonblocking attempt.

        Returns:
            ``True`` after acquisition, or ``False`` only for nonblocking
            contention from a local or cooperating external owner.

        Raises:
            LockError: The instance is already acquired or lock-file/native I/O
                fails.
            LockUnavailableError: No supported native adapter is available.

        Side Effects:
            Creates the lock-file parent and lock file when necessary.  The
            successful lease owns an open descriptor until release.
        """

        with self._operation_guard:
            if self._file is not None:
                raise LockError("FileLock is already acquired.")
            slot = _reserve_slot(self.path, shared=self.shared, blocking=blocking)
            if slot is None:
                return False
            state, effective_shared = slot
            file = None
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                file = open(self.path, "a+b")
                if not _lock_file(file.fileno(), shared=self.shared, blocking=blocking):
                    file.close()
                    _release_slot(state, shared=effective_shared)
                    return False
                if not effective_shared:
                    _activate_slot(state)
                self._file = file
                self._state = state
                self._effective_shared = effective_shared
                self._owner_thread = threading.get_ident()
                with _ACTIVE_FILE_LOCKS_GUARD:
                    _ACTIVE_FILE_LOCKS.add(self)
                return True
            except LockError:
                if file is not None:
                    file.close()
                self._file = None
                self._state = None
                self._effective_shared = False
                self._owner_thread = None
                _release_slot(state, shared=effective_shared)
                raise
            except OSError as error:
                if file is not None:
                    file.close()
                self._file = None
                self._state = None
                self._effective_shared = False
                self._owner_thread = None
                _release_slot(state, shared=effective_shared)
                raise LockError(f"Could not open advisory lock file {self.path!r}: {error}") from error
            except BaseException:
                try:
                    if file is not None:
                        file.close()
                finally:
                    self._file = None
                    self._state = None
                    self._effective_shared = False
                    self._owner_thread = None
                    with _ACTIVE_FILE_LOCKS_GUARD:
                        _ACTIVE_FILE_LOCKS.discard(self)
                    _release_slot(state, shared=effective_shared)
                raise

    def release(self) -> None:
        """Release this lease and close only its retained descriptor.

        Returns:
            ``None`` after releasing an owned lease or when already released.

        Raises:
            LockError: The native unlock fails, or a thread other than the
                acquiring thread attempts to release an active lease.

        Repeated calls after a successful release are no-ops.  The method does
        not unlink the durable lock file; a native unlock failure is reported as
        :class:`LockError` after local coordination and descriptor cleanup. A
        different thread cannot release an active lease and receives
        :class:`LockError` without changing it.
        """

        with self._operation_guard:
            file = self._file
            state = self._state
            effective_shared = self._effective_shared
            if file is None or state is None:
                return
            if self._owner_thread != threading.get_ident():
                raise LockError("FileLock must be released by its acquiring thread.")
            self._file = None
            self._state = None
            self._effective_shared = False
            self._owner_thread = None
            with _ACTIVE_FILE_LOCKS_GUARD:
                _ACTIVE_FILE_LOCKS.discard(self)
            try:
                unlock_file(file.fileno())
            finally:
                try:
                    file.close()
                finally:
                    _release_slot(state, shared=effective_shared)

    def __enter__(self) -> FileLock:
        """Acquire this lease in blocking mode and return it for a ``with`` block."""

        if not self.acquire():  # pragma: no cover - blocking acquisition cannot contend.
            raise LockError("Blocking FileLock acquisition unexpectedly reported contention.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release this lease when its context exits without suppressing errors."""

        self.release()

    def _after_fork_child(self) -> None:
        """Drop the inherited descriptor without unlocking the parent's lease."""

        file = self._file
        self._file = None
        self._state = None
        self._effective_shared = False
        self._owner_thread = None
        self._operation_guard = threading.Lock()
        if file is not None:
            file.close()


@contextmanager
def interprocess_lock(path: str | os.PathLike[str]):
    """Acquire a reentrant blocking exclusive Store-compatible lock.

    Args:
        path: Existing or new lock-file path shared by cooperating processes.

    Yields:
        ``None`` while the calling thread owns the exclusive lease.

    Raises:
        LockError: The lock path or native adapter fails.

    Same-thread recursion is wrapper policy for existing Store writers.  Other
    :class:`FileLock` users remain non-reentrant.
    """

    path = os.path.abspath(os.fspath(path))
    held = getattr(_LOCK_STATE, "held", {})
    if path in held:
        held[path][0] += 1
        try:
            yield
        finally:
            held[path][0] -= 1
        return
    lease = FileLock(path)
    if not lease.acquire():  # pragma: no cover - blocking acquisition cannot contend.
        raise LockError("Blocking Store lock acquisition unexpectedly reported contention.")
    held[path] = [1, lease]
    _LOCK_STATE.held = held
    try:
        yield
    finally:
        held.pop(path, None)
        lease.release()


@contextmanager
def interprocess_read_lock(path: str | os.PathLike[str]):
    """Acquire a blocking shared Store-compatible reader lock.

    Args:
        path: Existing or new lock-file path shared by cooperating processes.

    Yields:
        ``None`` while the caller retains a reader lease.

    Raises:
        LockError: The lock path or native adapter fails.

    POSIX readers overlap.  The Windows byte-range adapter uses an exclusive
    fallback, preserving safety while sacrificing reader/reader overlap.
    """

    with FileLock(path, shared=True):
        yield


def _after_fork_child() -> None:
    """Reset inherited mutex bridges and close child copies of active leases."""

    global _PATH_STATES, _PATH_STATES_GUARD, _ACTIVE_FILE_LOCKS, _ACTIVE_FILE_LOCKS_GUARD, _LOCK_STATE
    for lock in tuple(_ACTIVE_FILE_LOCKS):
        lock._after_fork_child()
    _PATH_STATES = {}
    _PATH_STATES_GUARD = threading.Lock()
    _ACTIVE_FILE_LOCKS = WeakSet()
    _ACTIVE_FILE_LOCKS_GUARD = threading.Lock()
    _LOCK_STATE = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


__all__ = [
    "FileLock",
    "LockError",
    "LockUnavailableError",
    "interprocess_lock",
    "interprocess_read_lock",
    "supports_advisory_locking",
    "try_lock_file",
    "unlock_file",
]
