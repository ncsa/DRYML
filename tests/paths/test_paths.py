"""Public host-local path normalization contracts."""

import os
from pathlib import Path
from urllib.parse import unquote, unquote_to_bytes, urlparse

import pytest

from dryml.paths import absolute_path, local_path_key, to_file_uri


class BytesPath:
    """PathLike fixture returning raw filesystem bytes."""

    def __init__(self, path: bytes):
        self.path = path

    def __fspath__(self) -> bytes:
        return self.path


def test_absolute_path_is_lexical_and_does_not_resolve_symlinks(
        tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    result = absolute_path(Path("link") / "child")

    assert result == tmp_path / "link" / "child"
    assert result.is_absolute()


def test_local_path_key_matches_normcase_realpath_for_text_and_pathlike(
        tmp_path):
    path = tmp_path / "directory" / ".." / "target"
    expected = os.path.normcase(os.path.realpath(path))
    assert local_path_key(path) == expected
    assert isinstance(local_path_key(path), str)


def test_file_uri_uses_stdlib_escaping_and_absolute_path(tmp_path):
    path = tmp_path / "space # percent % unicode-é"
    uri = to_file_uri(path)
    parsed = urlparse(uri)

    assert parsed.scheme == "file"
    assert "%20" in uri and "%23" in uri and "%25" in uri
    assert unquote(parsed.path).endswith("space # percent % unicode-é")


@pytest.mark.parametrize("as_pathlike", [False, True], ids=["bytes", "pathlike"])
@pytest.mark.parametrize(
    "suffix",
    [
        b"portable",
        pytest.param(
            b"\xff",
            marks=pytest.mark.skipif(
                os.name != "posix",
                reason="undecodable raw-byte names are a POSIX contract",
            ),
        ),
    ],
    ids=["portable", "undecodable-posix"],
)
def test_path_utilities_preserve_bytes(tmp_path, as_pathlike, suffix):
    raw = os.path.join(os.fsencode(tmp_path), b"path-" + suffix)
    path = BytesPath(raw) if as_pathlike else raw

    absolute = absolute_path(path)
    key = local_path_key(path)
    uri = to_file_uri(path)

    assert os.fsencode(absolute) == raw
    assert os.fsencode(key) == os.path.normcase(os.path.realpath(raw))
    assert unquote_to_bytes(urlparse(uri).path).endswith(b"/path-" + suffix)
