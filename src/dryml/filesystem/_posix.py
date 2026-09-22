"""Private POSIX filesystem publication primitives."""

from __future__ import annotations

import ctypes
import errno
import os
import sys

from .errors import FilesystemCapabilityError


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
USES_WRITE_THROUGH = False
_UNSUPPORTED_RENAME_ERRNOS = {
    errno.ENOSYS,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    errno.EOPNOTSUPP,
}
_UNSUPPORTED_DIRECTORY_SYNC_ERRNOS = {
    errno.ENOSYS,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    errno.EOPNOTSUPP,
}


def sync_file(path) -> None:
    """Flush one regular file through its descriptor."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_directory(path) -> None:
    """Flush one directory or report the missing persistence capability."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in _UNSUPPORTED_DIRECTORY_SYNC_ERRNOS:
            raise FilesystemCapabilityError(
                "The filesystem cannot open a directory persistence barrier."
            ) from error
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno in _UNSUPPORTED_DIRECTORY_SYNC_ERRNOS:
                raise FilesystemCapabilityError(
                    "The filesystem cannot persist directory entries."
                ) from error
            raise
    finally:
        os.close(descriptor)


def _native_error(source, destination) -> OSError:
    code = ctypes.get_errno()
    if code in _UNSUPPORTED_RENAME_ERRNOS:
        return FilesystemCapabilityError(
            "The host lacks native atomic no-replace rename support."
        )
    return OSError(code, os.strerror(code), source, None, destination)


def _rename_noreplace(source, destination) -> None:
    """Rename without replacement through OS-enforced exclusion."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise FilesystemCapabilityError(
                "Linux atomic no-replace rename is unavailable."
            )
        renameat2.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD, source_bytes, _AT_FDCWD, destination_bytes,
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise FilesystemCapabilityError(
                "macOS atomic exclusive rename is unavailable."
            )
        renamex_np.argtypes = (
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, _RENAME_EXCL)
    else:
        raise FilesystemCapabilityError(
            "This POSIX host has no supported atomic no-replace rename "
            "adapter."
        )
    if result != 0:
        raise _native_error(source, destination)


def publish_path(source, destination, *, replace: bool) -> None:
    """Rename one path without any cross-filesystem copy fallback."""

    if replace:
        os.replace(source, destination)
    else:
        _rename_noreplace(source, destination)


def remove_file(path, cleanup_directory, *, missing_ok: bool) -> bool:
    """Unlink one file; ``cleanup_directory`` is reserved for Windows."""

    try:
        os.unlink(path)
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    return True


def cleanup_removed_file(path) -> None:
    """Remove a non-authoritative tombstone after logical deletion."""

    os.unlink(path)
