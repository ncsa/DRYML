# Host Paths

`dryml.paths` provides small dependency-light conversions for host-local paths.
Every path parameter accepts text, bytes, or `os.PathLike` and normalizes with
`os.fspath` and `os.fsdecode`. Undecodable POSIX bytes therefore round-trip via
Python's filesystem `surrogateescape` handling.

`absolute_path(path)` returns an absolute lexical `pathlib.Path`. It normalizes
relative and `..` spelling through the host path library but does not resolve
symlinks. `local_path_key(path)` returns `normcase(realpath(path))` as text for
host-local comparison and resource lookup. It is not persistent identity and
does not distinguish or prove hard-link identity; domains that require physical
evidence must retain their inode/device checks.

`to_file_uri(path)` first obtains an absolute lexical path and then delegates to
the standard library's `Path.as_uri()`. Spaces, reserved characters, percent
signs, and Unicode are escaped according to the standard library rather than by
interpolating a `file://` string. SQLite connection ownership uses this URI when
adding its `mode=ro` query so reserved path characters cannot become SQLite URI
syntax.

These functions do not define Store schemas, transport paths, or portable
cross-host identities. Private native adapters may use extended Win32 path
spellings, but those spellings are not part of this public concern.
