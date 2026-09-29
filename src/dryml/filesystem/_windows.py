"""Private Windows filesystem publication primitives."""

from __future__ import annotations

import os
from uuid import uuid4


_MOVEFILE_REPLACE_EXISTING = 0x00000001
_MOVEFILE_WRITE_THROUGH = 0x00000008
_REMOVED_PREFIX = ".dryml-removed-"
USES_WRITE_THROUGH = True


def _native_error(code: int, source: str, destination: str) -> OSError:
    """Return a native Windows error annotated with caller-visible paths."""

    import ctypes

    error = ctypes.WinError(code)
    error.filename = source
    error.filename2 = destination
    return error


def _extended_length_path(path: str) -> str:
    """Convert an absolute Win32 path to the private extended namespace."""

    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return f"\\\\?\\UNC\\{path[2:]}"
    return f"\\\\?\\{path}"


def _move_file_ex(source: str, destination: str, flags: int) -> None:
    """Invoke typed ``MoveFileExW`` and retain its native error subclass."""

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file_ex = kernel32.MoveFileExW
    move_file_ex.argtypes = (
        wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    )
    move_file_ex.restype = wintypes.BOOL
    if not move_file_ex(
            _extended_length_path(source),
            _extended_length_path(destination),
            flags):
        raise _native_error(ctypes.get_last_error(), source, destination)


def sync_file(path) -> None:
    """Flush one regular file through a writable Windows descriptor."""

    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_directory(path) -> None:
    """Reject POSIX-style directory barriers on Windows."""

    raise AssertionError("Windows publication must use write-through moves.")


def publish_path(source, destination, *, replace: bool) -> None:
    """Publish a same-volume path through ``MoveFileExW`` write-through."""

    flags = _MOVEFILE_WRITE_THROUGH
    if replace:
        flags |= _MOVEFILE_REPLACE_EXISTING
    _move_file_ex(
        os.path.abspath(os.fsdecode(os.fspath(source))),
        os.path.abspath(os.fsdecode(os.fspath(destination))),
        flags,
    )


def remove_file(path, cleanup_directory, *, missing_ok: bool) -> bool:
    """Move a logical name to a unique ignored tombstone before cleanup."""

    source = os.path.abspath(os.fsdecode(os.fspath(path)))
    cleanup = os.path.abspath(os.fsdecode(os.fspath(cleanup_directory)))
    tombstone = os.path.join(cleanup, f"{_REMOVED_PREFIX}{uuid4().hex}")
    try:
        publish_path(source, tombstone, replace=False)
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    try:
        cleanup_removed_file(tombstone)
    except OSError:
        pass
    return True


def cleanup_removed_file(path) -> None:
    """Best-effort unlink one already non-authoritative tombstone."""

    os.unlink(path)
