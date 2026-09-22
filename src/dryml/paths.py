"""Dependency-light host path and file-URI normalization."""

from __future__ import annotations

import os
from pathlib import Path


_PathInput = str | bytes | os.PathLike[str] | os.PathLike[bytes]


def _path(path: _PathInput) -> str:
    """Return one text path using the host surrogateescape conversion."""

    return os.fsdecode(os.fspath(path))


def absolute_path(path: _PathInput) -> Path:
    """Return an absolute lexical path without resolving symlinks.

    Args:
        path: Text, bytes, or ``os.PathLike`` host path. Relative paths use the
            current working directory at call time.

    Returns:
        An absolute :class:`pathlib.Path`. ``..`` components are normalized by
        the host path library, but symlinks are not resolved.

    Raises:
        TypeError: If ``path`` is not path-like.

    Side Effects:
        None. The path need not exist.
    """

    return Path(os.path.abspath(_path(path)))


def local_path_key(path: _PathInput) -> str:
    """Return a normalized real-path key for host-local comparisons.

    Args:
        path: Text, bytes, or ``os.PathLike`` host path, whether or not it
            exists.

    Returns:
        ``normcase(realpath(path))`` as text. The key is process-host local; it
        is neither persistent identity nor proof that hard links identify
        distinct files.

    Raises:
        TypeError: If ``path`` is not path-like.

    Side Effects:
        Resolves path spellings through host filesystem path semantics without
        opening or modifying the target.
    """

    return os.path.normcase(os.path.realpath(_path(path)))


def to_file_uri(path: _PathInput) -> str:
    """Return a correctly escaped absolute ``file`` URI for a host path.

    Args:
        path: Text, bytes, or ``os.PathLike`` host path accepted by
            :func:`absolute_path`.

    Returns:
        The standard-library :meth:`pathlib.Path.as_uri` representation.

    Raises:
        TypeError: If ``path`` is not path-like.
        ValueError: If the host cannot represent the path as an absolute file
            URI.

    Side Effects:
        None. The path need not exist.
    """

    return absolute_path(path).as_uri()


__all__ = ["absolute_path", "local_path_key", "to_file_uri"]
