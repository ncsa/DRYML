"""Dependency-light cross-platform filesystem publication guarantees.

The public functions select native POSIX or Windows mechanics internally.
Consumers provide paths and policy such as replacement or a safe cleanup
directory; they never branch on the host or call backend internals.
"""

from __future__ import annotations

import errno
import os
import stat

from .errors import FilesystemCapabilityError, FilesystemError

if os.name == "nt":  # pragma: no cover - selected on Windows hosts.
    from . import _windows as _backend
else:  # pragma: no branch - the backend is fixed for one interpreter.
    from . import _posix as _backend


def _path(value):
    """Return one text path using the host surrogateescape conversion."""

    return os.fsdecode(os.fspath(value))


def _require_bool(name: str, value) -> None:
    """Reject truthy option substitutes before any filesystem I/O."""

    if type(value) is not bool:
        raise TypeError(f"{name} must be bool.")


def _require_regular_file(path):
    """Require a regular file without following a final symlink."""

    evidence = os.lstat(path)
    if not stat.S_ISREG(evidence.st_mode):
        if stat.S_ISDIR(evidence.st_mode):
            raise IsADirectoryError(
                errno.EISDIR, os.strerror(errno.EISDIR), path,
            )
        raise FilesystemError(f"Path is not a regular file: {path!r}.")
    return evidence


def _require_directory(path):
    """Require a directory without following a final symlink."""

    evidence = os.lstat(path)
    if not stat.S_ISDIR(evidence.st_mode):
        raise NotADirectoryError(
            errno.ENOTDIR, os.strerror(errno.ENOTDIR), path,
        )
    return evidence


def _require_existing_directory(path):
    """Require a directory while following an existing final symlink."""

    evidence = os.stat(path)
    if not stat.S_ISDIR(evidence.st_mode):
        raise NotADirectoryError(
            errno.ENOTDIR, os.strerror(errno.ENOTDIR), path,
        )
    return evidence


def _require_distinct(source, destination) -> None:
    """Reject lexical and existing-inode aliases before publication."""

    source_absolute = os.path.normcase(os.path.abspath(source))
    destination_absolute = os.path.normcase(os.path.abspath(destination))
    if source_absolute == destination_absolute:
        raise ValueError("staged_path and destination must be distinct paths.")
    try:
        same = os.path.samestat(os.stat(source), os.stat(destination))
    except FileNotFoundError:
        same = False
    if same:
        raise ValueError(
            "staged_path and destination must identify distinct files."
        )


def sync_file(path) -> None:
    """Flush one existing regular file's content and metadata.

    Args:
        path: Text, bytes, or ``os.PathLike`` path to a regular non-symlink
            file.

    Returns:
        ``None`` after the native file barrier completes.

    Raises:
        TypeError: If ``path`` is not path-like.
        FileNotFoundError: If the file is absent.
        IsADirectoryError: If the path names a directory.
        FilesystemError: If the path has another unsupported shape.
        OSError: For native open, permission, or persistence failures.

    Side Effects:
        Opens and flushes the file. It does not rename, copy, or remove it.
    """

    native = _path(path)
    _require_regular_file(native)
    _backend.sync_file(native)


def ensure_directory(path) -> None:
    """Create and persist every missing component of a directory path.

    Args:
        path: Text, bytes, or ``os.PathLike`` directory path. Existing
            directory components, including symlinks resolving to directories,
            are accepted; an existing non-directory is rejected.

    Returns:
        ``None`` when the complete directory chain exists and all newly created
        parent entries have crossed the platform persistence barrier.

    Raises:
        FileExistsError: If the final path exists as a non-directory.
        NotADirectoryError: If an intermediate component is not a directory.
        FilesystemCapabilityError: If required native publication or directory
            persistence is unavailable.
        OSError: For native permission and I/O failures.

    Side Effects:
        Creates missing directories. A failure may occur after one or more are
        visible and never implies rollback.
    """

    native = _path(path)
    absolute = os.path.abspath(native)
    missing = []
    current = absolute
    while True:
        try:
            _require_existing_directory(current)
            break
        except FileNotFoundError:
            missing.append(current)
            parent = os.path.dirname(current)
            if parent == current:
                raise
            current = parent
        except NotADirectoryError:
            if current == absolute:
                raise FileExistsError(
                    errno.EEXIST, os.strerror(errno.EEXIST), absolute,
                )
            raise
    for directory in reversed(missing):
        parent = os.path.dirname(directory) or os.curdir
        if _backend.USES_WRITE_THROUGH:
            import tempfile

            temporary = tempfile.mkdtemp(
                prefix=".dryml-directory-", dir=parent,
            )
            try:
                _backend.publish_path(temporary, directory, replace=False)
            except FileExistsError:
                try:
                    _require_existing_directory(directory)
                except BaseException:
                    try:
                        os.rmdir(temporary)
                    except OSError:
                        pass
                    raise
                try:
                    os.rmdir(temporary)
                except FileNotFoundError:
                    pass
            except BaseException:
                try:
                    os.rmdir(temporary)
                except OSError:
                    pass
                raise
        else:
            try:
                os.mkdir(directory)
            except FileExistsError:
                _require_existing_directory(directory)
                continue
            _backend.sync_directory(parent)


def _sync_publication_parents(source, destination) -> None:
    """Persist rename metadata in each changed POSIX directory."""

    if _backend.USES_WRITE_THROUGH:
        return
    source_parent = os.path.dirname(os.path.abspath(source)) or os.curdir
    destination_parent = (
        os.path.dirname(os.path.abspath(destination)) or os.curdir
    )
    _backend.sync_directory(destination_parent)
    if os.path.normcase(source_parent) != os.path.normcase(destination_parent):
        _backend.sync_directory(source_parent)


def publish_file(staged_path, destination, *, replace: bool = False) -> None:
    """Flush and atomically publish one staged regular file.

    Args:
        staged_path: Existing regular non-symlink file containing complete
            bytes.
        destination: Final path on the same filesystem or Windows volume.
        replace: Exact boolean permitting atomic replacement of an existing
            destination file. It is validated before file access.

    Returns:
        ``None`` after publication and required persistence barriers complete.

    Raises:
        TypeError: If ``replace`` is not an exact boolean.
        ValueError: If both paths are lexical or existing hard-link aliases.
        FileExistsError: If ``replace`` is false and the destination exists.
        FilesystemCapabilityError: If atomic no-clobber publication or required
            persistence barriers are unavailable.
        OSError: For native shape, permission, cross-volume, or I/O failures.

    Side Effects:
        Flushes and then renames ``staged_path``. Success consumes staging.
        There is no copy fallback. Failure can occur after the destination is
        visible, so an exception never promises rollback and callers must
        reconcile destination state when that distinction matters.
    """

    _require_bool("replace", replace)
    source = _path(staged_path)
    target = _path(destination)
    _require_regular_file(source)
    _require_distinct(source, target)
    _backend.sync_file(source)
    _backend.publish_path(source, target, replace=replace)
    _sync_publication_parents(source, target)


def _sync_tree(root) -> None:
    """Flush one complete staged tree exactly once before publication."""

    def raise_walk_error(error):
        raise error

    for directory, directories, files in os.walk(
            root, topdown=False, onerror=raise_walk_error):
        for name in files:
            sync_file(os.path.join(directory, name))
        for name in directories:
            _require_directory(os.path.join(directory, name))
        if not _backend.USES_WRITE_THROUGH:
            _backend.sync_directory(directory)


def publish_directory(staged_path, destination) -> None:
    """Flush and atomically publish a staged tree at an absent destination.

    Args:
        staged_path: Existing complete non-symlink directory tree.
        destination: Absent final path on the same filesystem or Windows
            volume.

    Returns:
        ``None`` after tree flushing, exclusive publication, and persistence.

    Raises:
        ValueError: If source and destination are the same lexical path.
        FileExistsError: If any file or directory already occupies
            destination.
        FilesystemCapabilityError: If native no-clobber publication or required
            persistence barriers are unavailable.
        OSError: For native shape, permission, cross-volume, or I/O failures.

    Side Effects:
        Traverses the staged tree once to flush files and directories, then
        renames it without replacement. Success consumes staging. There is no
        copy fallback; failure can occur after visible publication and does not
        imply rollback.
    """

    source = _path(staged_path)
    target = _path(destination)
    _require_directory(source)
    _require_distinct(source, target)
    source_absolute = os.path.normcase(os.path.abspath(source))
    target_absolute = os.path.normcase(os.path.abspath(target))
    try:
        target_within_source = (
            os.path.commonpath((source_absolute, target_absolute))
            == source_absolute
        )
    except ValueError:
        target_within_source = False
    if target_within_source:
        raise ValueError("destination must not be within staged_path.")
    _sync_tree(source)
    _backend.publish_path(source, target, replace=False)
    _sync_publication_parents(source, target)


def remove_file(path, *, cleanup_directory, missing_ok: bool = False) -> bool:
    """Durably remove one logical file name.

    Args:
        path: Regular file name to remove.
        cleanup_directory: Existing non-symlink directory safe for
            unrecognized native deletion tombstones. It must share the source
            filesystem.
        missing_ok: Return ``False`` rather than raising when ``path`` is
            absent. This option must be an exact boolean.

    Returns:
        ``True`` after logical removal, or ``False`` only for an allowed
        missing source.

    Raises:
        TypeError: If ``missing_ok`` is not an exact boolean.
        FileNotFoundError: If the source is absent and ``missing_ok`` is
            false.
        NotADirectoryError: If ``cleanup_directory`` is not a directory.
        FilesystemCapabilityError: If the logical-removal persistence barrier
            is unavailable.
        OSError: For native permission, cross-volume, or I/O failures.

    Side Effects:
        POSIX unlinks and persists the parent entry. Windows write-through
        moves the name to a unique tombstone before best-effort cleanup.
        Tombstone cleanup errors are ignored because logical authority is
        already absent.
        Other failures may happen after visible removal and never imply
        rollback.

    Validation Order:
        ``missing_ok`` is validated before filesystem access. Source absence or
        shape is then resolved before inspecting ``cleanup_directory``. A
        missing source therefore returns ``False`` when allowed, even if the
        cleanup directory is absent. For a present source, cleanup shape and
        same-device evidence are validated before mutation.
    """

    _require_bool("missing_ok", missing_ok)
    native = _path(path)
    try:
        source_evidence = _require_regular_file(native)
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    cleanup = _path(cleanup_directory)
    cleanup_evidence = _require_directory(cleanup)
    if source_evidence.st_dev != cleanup_evidence.st_dev:
        raise OSError(
            errno.EXDEV,
            "cleanup_directory must share the source filesystem",
            native,
            None,
            cleanup,
        )
    removed = _backend.remove_file(native, cleanup, missing_ok=missing_ok)
    if removed and not _backend.USES_WRITE_THROUGH:
        parent = os.path.dirname(os.path.abspath(native)) or os.curdir
        _backend.sync_directory(parent)
    return removed


__all__ = [
    "FilesystemCapabilityError",
    "FilesystemError",
    "ensure_directory",
    "publish_directory",
    "publish_file",
    "remove_file",
    "sync_file",
]
