# Filesystem Publication

`dryml.filesystem` is DRYML's dependency-light cross-platform concern for local
filesystem persistence and atomic publication. Consumers call the same API on
every host; POSIX and Windows native mechanics remain private.

Every path parameter accepts text, bytes, or `os.PathLike`. The public boundary
normalizes with `os.fspath` and `os.fsdecode`, so backend calls receive one
consistent text shape while undecodable POSIX bytes round-trip through Python's
filesystem `surrogateescape` handling.
This preserves the path representation, not permission to create every byte
sequence: native filesystem naming restrictions still apply. Filesystems that
reject non-UTF-8 names report their native `EILSEQ` error unchanged.

`sync_file(path)` flushes an existing regular file. `ensure_directory(path)`
creates missing components and persists each new parent entry. Like
`os.makedirs(..., exist_ok=True)`, it follows existing directory symlinks,
including the final component. Publication staging sources and removal cleanup
directories retain their stricter non-symlink shape checks.
`publish_file(staged_path, destination, replace=False)` flushes and atomically
renames one staged file. `publish_directory(staged_path, destination)` flushes a
complete staged tree and atomically installs it only when the destination is
absent. Successful publication consumes staging. File replacement is explicit;
file and directory no-clobber publication is enforced by the native rename
operation, including competing trusted publishers.
`replace` must be an exact boolean and is validated before staging is flushed.

Publication never falls back to copying across filesystems or volumes. POSIX
uses atomic replacement or a native exclusive rename and then persists changed
directory entries. Windows uses same-volume `MoveFileExW` with
`MOVEFILE_WRITE_THROUGH`, adding `MOVEFILE_REPLACE_EXISTING` only for explicit
file replacement. If the host lacks the required no-replace primitive or
persistence barrier, DRYML raises `FilesystemCapabilityError` rather than
weakening the contract.
Only host errnos identifying an unsupported directory persistence operation are
translated to that capability error. Operational failures such as I/O errors,
descriptor exhaustion, and storage exhaustion retain their native `OSError`.

`remove_file(path, cleanup_directory=..., missing_ok=False)` durably removes a
logical file name. The caller supplies an existing same-filesystem directory
where Windows may place an unrecognized tombstone. Tombstone cleanup is
best-effort only after the logical name is absent. POSIX directly unlinks and
persists the containing directory.
`missing_ok` must be an exact boolean and is checked before filesystem access.
Source absence or shape is resolved before the cleanup directory is inspected:
an allowed missing source returns `False` even when cleanup is absent. For a
present source, DRYML verifies cleanup directory shape and matching `st_dev`
evidence before mutation on every platform.

Native `FileExistsError`, `FileNotFoundError`, `PermissionError`, and
cross-device `OSError` values remain available to callers. An error can occur
after publication or removal is visible, especially while completing a
persistence barrier. Errors therefore never promise rollback; callers whose
domain distinguishes pre- and post-publication failures must reconcile the
destination. These guarantees are conditional on the documented local
filesystem rename, file-flush, directory-persistence, and Windows write-through
semantics. They are not a sandbox or a proof against hostile races or power-loss
behavior beyond those host guarantees.
