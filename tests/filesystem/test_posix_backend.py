"""Private POSIX publication error construction."""

import errno

import pytest

from dryml.filesystem import _posix as posix_backend


@pytest.mark.parametrize(
    ("code", "error_type"),
    ((errno.EEXIST, FileExistsError), (errno.EXDEV, OSError)),
)
def test_native_rename_error_retains_errno_and_both_filenames(
        monkeypatch, code, error_type):
    monkeypatch.setattr(posix_backend.ctypes, "get_errno", lambda: code)

    error = posix_backend._native_error("source", "destination")

    assert isinstance(error, error_type)
    assert error.errno == code
    assert error.filename == "source"
    assert error.filename2 == "destination"


@pytest.mark.parametrize(
    ("operation", "code"),
    (("open", errno.EMFILE), ("fsync", errno.EIO), ("fsync", errno.ENOSPC)),
)
def test_directory_sync_propagates_operational_errors_without_wrapping(
        monkeypatch, operation, code):
    failure = OSError(code, "injected operational failure", "directory")
    if operation == "open":
        monkeypatch.setattr(posix_backend.os, "open", lambda *_args: (_ for _ in ()).throw(failure))
    else:
        monkeypatch.setattr(posix_backend.os, "open", lambda *_args: 7)
        monkeypatch.setattr(posix_backend.os, "fsync", lambda _fd: (_ for _ in ()).throw(failure))
        monkeypatch.setattr(posix_backend.os, "close", lambda _fd: None)

    with pytest.raises(OSError) as raised:
        posix_backend.sync_directory("directory")

    assert raised.value is failure
    assert raised.value.errno == code
    assert raised.value.__cause__ is None


def test_directory_sync_wraps_only_unsupported_errno_with_cause(monkeypatch):
    failure = OSError(errno.EINVAL, "directory fsync unsupported", "directory")
    monkeypatch.setattr(posix_backend.os, "open", lambda *_args: 7)
    monkeypatch.setattr(posix_backend.os, "fsync", lambda _fd: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(posix_backend.os, "close", lambda _fd: None)

    with pytest.raises(posix_backend.FilesystemCapabilityError) as raised:
        posix_backend.sync_directory("directory")

    assert raised.value.__cause__ is failure
    assert raised.value.__cause__.errno == errno.EINVAL
