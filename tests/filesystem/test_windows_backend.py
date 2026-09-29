"""Private Windows mechanics owned by the public filesystem concern."""

import ctypes
import errno

import pytest

from dryml.filesystem import _windows as windows_backend


def test_write_through_move_selects_exact_flags_and_propagates_failure(
        tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(
        windows_backend,
        "_move_file_ex",
        lambda source, destination, flags: observed.append(
            (source, destination, flags),
        ),
    )
    source = tmp_path / "source"
    destination = tmp_path / "destination"

    windows_backend.publish_path(source, destination, replace=False)
    windows_backend.publish_path(source, destination, replace=True)

    write_through = windows_backend._MOVEFILE_WRITE_THROUGH
    replace_existing = windows_backend._MOVEFILE_REPLACE_EXISTING
    assert observed == [
        (str(source), str(destination), write_through),
        (str(source), str(destination), write_through | replace_existing),
    ]

    monkeypatch.setattr(
        windows_backend,
        "_move_file_ex",
        lambda *_args: (_ for _ in ()).throw(
            PermissionError("native sharing denial"),
        ),
    )
    with pytest.raises(PermissionError, match="sharing denial"):
        windows_backend.publish_path(source, destination, replace=True)


def test_extended_length_paths_preserve_drive_unc_and_prefixed_names():
    convert = windows_backend._extended_length_path
    assert convert(r"C:\store\record") == r"\\?\C:\store\record"
    assert convert(r"\\server\share\record") == (
        r"\\?\UNC\server\share\record"
    )
    assert convert(r"\\?\C:\store\record") == r"\\?\C:\store\record"


def test_native_move_error_retains_subclass_code_and_supplied_paths(monkeypatch):
    native = FileExistsError(errno.EEXIST, "already exists")
    monkeypatch.setattr(ctypes, "WinError", lambda _code: native, raising=False)

    error = windows_backend._native_error(
        183, r"C:\store\source", r"C:\store\destination",
    )

    assert error is native
    assert isinstance(error, FileExistsError)
    assert error.errno == errno.EEXIST
    assert error.filename == r"C:\store\source"
    assert error.filename2 == r"C:\store\destination"
