"""Lifetime-held platform file locks for managed operation ownership."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

from .errors import ManagedLeaseConflictError, ManagedStoreUnsupportedError


class PlatformFileLock:
    """Hold one non-blocking OS file lock for this object's lifetime.

    The lock file is stable Store-local coordination state. Its contents are not
    ownership proof; only the open file descriptor and platform lock are.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._file = None
        self._held = False

    @property
    def held(self) -> bool:
        """Return whether this object currently holds the platform lock."""

        return self._held

    def acquire(self) -> None:
        """Acquire the lock without waiting or raise an explicit conflict."""

        if self._held:
            raise RuntimeError("platform lock is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_obj = open(self.path, "a+b", buffering=0)
        try:
            mode = os.fstat(file_obj.fileno()).st_mode
            if not stat.S_ISREG(mode):
                raise ManagedStoreUnsupportedError("managed locking requires a regular local lock file")
            if os.name == "nt":
                self._acquire_windows(file_obj)
            elif os.name == "posix":
                self._acquire_posix(file_obj)
            else:
                raise ManagedStoreUnsupportedError(
                    f"managed locking is unsupported on platform {os.name!r}"
                )
        except Exception:
            file_obj.close()
            raise
        self._file = file_obj
        self._held = True

    def release(self) -> None:
        """Release the held platform lock and close its descriptor."""

        if not self._held:
            return
        assert self._file is not None
        try:
            if os.name == "nt":
                self._release_windows(self._file)
            else:
                self._release_posix(self._file)
        finally:
            self._held = False
            self._file.close()
            self._file = None

    def __enter__(self) -> "PlatformFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    @staticmethod
    def _acquire_posix(file_obj) -> None:
        try:
            import fcntl
        except ImportError as exc:
            raise ManagedStoreUnsupportedError("POSIX managed locking requires fcntl") from exc
        try:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ManagedLeaseConflictError("managed operation is already owned") from exc
            raise ManagedStoreUnsupportedError("filesystem does not provide supported flock semantics") from exc

    @staticmethod
    def _release_posix(file_obj) -> None:
        import fcntl

        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _acquire_windows(file_obj) -> None:
        try:
            import msvcrt
        except ImportError as exc:
            raise ManagedStoreUnsupportedError("Windows managed locking requires msvcrt") from exc
        file_obj.seek(0)
        if not file_obj.read(1):
            file_obj.write(b"\0")
        file_obj.seek(0)
        try:
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise ManagedLeaseConflictError("managed operation is already owned") from exc
            raise ManagedStoreUnsupportedError("filesystem does not provide supported locking semantics") from exc

    @staticmethod
    def _release_windows(file_obj) -> None:
        import msvcrt

        file_obj.seek(0)
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)


def process_is_alive(pid: int) -> bool:
    """Best-effort diagnostic check for a same-host process ID."""

    if type(pid) is not int or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


__all__ = ["PlatformFileLock", "process_is_alive"]
