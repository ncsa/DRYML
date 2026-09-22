"""Narrow native Windows primitive for durable same-volume Store moves."""

from __future__ import annotations

import os
import tempfile
from uuid import uuid4


_IS_WINDOWS = os.name == "nt"
_MOVEFILE_REPLACE_EXISTING = 0x00000001
_MOVEFILE_WRITE_THROUGH = 0x00000008


def is_windows() -> bool:
    """Return whether Store publication must use the native Windows adapter."""

    return _IS_WINDOWS


def _extended_length_path(path: str) -> str:
    """Convert an absolute Win32 path to the documented extended namespace.

    Args:
        path: Absolute drive, UNC, or already-prefixed Unicode path.

    Returns:
        The equivalent extended-namespace drive or UNC path.
    """

    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return f"\\\\?\\UNC\\{path[2:]}"
    return f"\\\\?\\{path}"


def _move_file_ex(source: str, destination: str, flags: int) -> None:
    """Invoke typed ``MoveFileExW`` with extended paths and native failures.

    Args:
        source: Absolute existing source path.
        destination: Absolute final path.
        flags: Exact ``MOVEFILE_*`` flag mask.

    Raises:
        OSError: From the native Windows error code when the move returns false.

    Side Effects:
        Moves the source according to ``flags`` through the Unicode Win32 API.
    """

    import ctypes
    from ctypes import wintypes

    move_file_ex = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
    move_file_ex.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
    move_file_ex.restype = wintypes.BOOL
    if not move_file_ex(
            _extended_length_path(source),
            _extended_length_path(destination),
            flags,
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def move_file_write_through(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        replace: bool,
) -> None:
    """Durably move one same-volume file or directory through Win32.

    Args:
        source: Existing sibling temporary file or directory.
        destination: Final path on the same local volume.
        replace: Whether an existing destination file may be replaced.

    Raises:
        OSError: If Windows rejects or cannot durably complete the move.

    Side Effects:
        Atomically changes the visible name and requests write-through metadata
        publication before returning. Cross-volume copy fallback is never enabled.
    """

    flags = _MOVEFILE_WRITE_THROUGH
    if replace:
        flags |= _MOVEFILE_REPLACE_EXISTING
    _move_file_ex(
        os.path.abspath(os.fsdecode(os.fspath(source))),
        os.path.abspath(os.fsdecode(os.fspath(destination))),
        flags,
    )


def makedirs_write_through(path: str | os.PathLike[str]) -> None:
    """Durably create every missing component of an absolute directory path.

    Args:
        path: Directory chain that must exist before publication.

    Raises:
        OSError: If creation or write-through installation fails for any reason
            other than a verified destination-exists race.

    Side Effects:
        Creates private sibling directories and installs each through a
        write-through move. Concurrently installed real directories are accepted.
    """

    destination = os.path.abspath(os.fsdecode(os.fspath(path)))
    missing = []
    current = destination
    while not os.path.exists(current):
        missing.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    for directory in reversed(missing):
        parent = os.path.dirname(directory) or "."
        temporary = tempfile.mkdtemp(prefix=".store-dir-", dir=parent)
        try:
            move_file_write_through(temporary, directory, replace=False)
        except FileExistsError:
            if os.path.isdir(directory):
                try:
                    os.rmdir(temporary)
                except FileNotFoundError:
                    pass
                continue
            try:
                os.rmdir(temporary)
            except OSError:
                pass
            raise
        except OSError:
            try:
                os.rmdir(temporary)
            except OSError:
                pass
            raise


def unlink_file_write_through(
        path: str | os.PathLike[str],
        *,
        tombstone_prefix: str,
) -> bool:
    """Durably remove one logical name through a unique sibling tombstone.

    Args:
        path: File whose recognized logical name must become absent.
        tombstone_prefix: Prefix that the owning reader contract ignores.

    Returns:
        ``True`` when the logical name was moved, or ``False`` when it was
        already absent.

    Raises:
        OSError: If Windows cannot durably move the logical name.

    Side Effects:
        Write-through moves ``path`` to a unique non-authoritative sibling, then
        best-effort unlinks only that tombstone. Cleanup failure leaves the
        caller's recognized name durably absent.
    """

    source = os.path.abspath(os.fsdecode(os.fspath(path)))
    parent = os.path.dirname(source) or "."
    tombstone = os.path.join(parent, f"{tombstone_prefix}{uuid4().hex}")
    try:
        move_file_write_through(source, tombstone, replace=False)
    except FileNotFoundError:
        return False
    try:
        os.unlink(tombstone)
    except OSError:
        pass
    return True
