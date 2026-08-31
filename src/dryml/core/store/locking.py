"""Cross-platform advisory locks for cooperating Store processes."""

from __future__ import annotations

import os
import stat
import threading
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None
    import msvcrt


_LOCK_STATE = threading.local()


def supports_advisory_locking(path: str) -> bool:
    """Return whether ``path`` is a local directory usable by this lock adapter.

    Args:
        path: Existing or prospective lock-file path.

    Returns:
        ``True`` when the path parent is a normal local directory. This adapter
        deliberately does not claim support for non-directory or symlink roots;
        backend-specific distributed filesystem support requires another lock
        implementation rather than silently reusing this one.
    """

    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        mode = os.lstat(directory).st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(mode) and not stat.S_ISLNK(mode)


@contextmanager
def interprocess_lock(path: str):
    """Acquire an exclusive advisory lock identified by a durable path.

    Args:
        path: Existing or new lock-file path shared by cooperating processes.

    Yields:
        ``None`` while the caller exclusively owns the lock.

    Raises:
        OSError: If the lock file cannot be opened or locked.

    Side Effects:
        Creates the lock-file parent and lock file when necessary. The lock is
        released before the file handle closes on every exit path.
    """

    path = os.path.abspath(path)
    held = getattr(_LOCK_STATE, "held", {})
    if path in held:
        held[path][0] += 1
        try:
            yield
        finally:
            held[path][0] -= 1
        return

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with open(path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        else:  # pragma: no cover - exercised on Windows
            if os.fstat(lock_file.fileno()).st_size == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        held[path] = [1]
        _LOCK_STATE.held = held
        try:
            yield
        finally:
            held.pop(path, None)
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover - exercised on Windows
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def interprocess_read_lock(path: str):
    """Acquire a shared advisory lock identified by a durable path.

    Args:
        path: Existing or new lock-file path shared by cooperating processes.

    Yields:
        ``None`` while the caller retains a reader lease.

    Raises:
        OSError: If the lock file cannot be opened or locked.

    Side Effects:
        Creates the lock-file parent and lock file when necessary. On platforms
        without shared advisory locks, this takes an exclusive lock instead, so
        readers remain safe while losing read/read overlap.
    """

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with open(path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        else:  # pragma: no cover - exercised on Windows
            if os.fstat(lock_file.fileno()).st_size == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover - exercised on Windows
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
