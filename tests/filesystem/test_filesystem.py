"""Public cross-platform filesystem publication contracts."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

import dryml.filesystem as filesystem
from dryml.filesystem import _windows as windows_backend


class BytesPath:
    """PathLike fixture returning raw filesystem bytes."""

    def __init__(self, path: bytes):
        self.path = path

    def __fspath__(self) -> bytes:
        return self.path


def _staged_file(root: Path, name: str, payload: bytes = b"payload") -> Path:
    path = root / name
    path.write_bytes(payload)
    return path


def test_publish_file_no_clobber_consumes_staging_and_preserves_bytes(
        tmp_path):
    staged = _staged_file(tmp_path, "staged", b"\x00 exact bytes \xff")
    destination = tmp_path / "published name"

    filesystem.publish_file(staged, destination)

    assert destination.read_bytes() == b"\x00 exact bytes \xff"
    assert not staged.exists()


def test_bytes_paths_and_unicode_names_preserve_native_path_fidelity(tmp_path):
    staged = tmp_path / "staged-é"
    staged.write_bytes(b"\x00\xff native")
    destination = tmp_path / "published-é"

    filesystem.publish_file(
        os.fsencode(staged), os.fsencode(destination), replace=False,
    )

    assert destination.read_bytes() == b"\x00\xff native"
    assert not staged.exists()


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
def test_bytes_and_bytes_pathlike_work_for_every_operation(
        tmp_path, as_pathlike, suffix):
    def path(value):
        return BytesPath(value) if as_pathlike else value

    root = os.fsencode(tmp_path)
    ensured = os.path.join(root, b"directory-" + suffix, b"nested")
    filesystem.ensure_directory(path(ensured))
    assert os.path.isdir(ensured)

    sync_target = os.path.join(root, b"sync-" + suffix)
    with open(sync_target, "wb") as target:
        target.write(b"sync")
    filesystem.sync_file(path(sync_target))

    staged_file = os.path.join(root, b"staged-file-" + suffix)
    published_file = os.path.join(root, b"published-file-" + suffix)
    with open(staged_file, "wb") as target:
        target.write(b"file")
    filesystem.publish_file(
        path(staged_file), path(published_file),
    )
    assert not os.path.exists(staged_file)
    with open(published_file, "rb") as source:
        assert source.read() == b"file"

    staged_directory = os.path.join(root, b"staged-directory-" + suffix)
    published_directory = os.path.join(root, b"published-directory-" + suffix)
    os.mkdir(staged_directory)
    payload = b"payload-" + suffix
    with open(os.path.join(staged_directory, payload), "wb") as target:
        target.write(b"directory")
    filesystem.publish_directory(
        path(staged_directory), path(published_directory),
    )
    assert not os.path.exists(staged_directory)
    with open(
            os.path.join(published_directory, payload), "rb",
    ) as source:
        assert source.read() == b"directory"

    assert filesystem.remove_file(
        path(published_file),
        cleanup_directory=path(root),
    )
    assert not os.path.exists(published_file)


def test_windows_directory_creation_normalizes_bytes_before_native_adapter(
        tmp_path, monkeypatch):
    observed = []

    def move(source, destination, _flags):
        observed.append((source, destination))
        os.rename(source, destination)

    monkeypatch.setattr(filesystem, "_backend", windows_backend)
    monkeypatch.setattr(windows_backend, "_move_file_ex", move)
    destination = os.path.join(
        os.fsencode(tmp_path), b"windows-bytes", b"nested",
    )

    filesystem.ensure_directory(BytesPath(destination))

    assert os.path.isdir(destination)
    assert observed
    assert all(
        isinstance(source, str) and isinstance(target, str)
        for source, target in observed
    )


@pytest.mark.parametrize("existing", [b"", b"existing"])
def test_publish_file_no_clobber_retains_existing_destination(
        tmp_path, existing):
    staged = _staged_file(tmp_path, "staged", b"new")
    destination = _staged_file(tmp_path, "destination", existing)

    with pytest.raises(FileExistsError) as raised:
        filesystem.publish_file(staged, destination)

    assert raised.value.errno == errno.EEXIST
    assert raised.value.filename == os.fspath(staged)
    assert raised.value.filename2 == os.fspath(destination)
    assert destination.read_bytes() == existing
    assert staged.read_bytes() == b"new"


@pytest.mark.parametrize("populate", [False, True], ids=["empty", "nonempty"])
def test_publish_directory_no_clobber_retains_existing_destination(
        tmp_path, populate):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "new").write_bytes(b"new")
    destination = tmp_path / "destination"
    destination.mkdir()
    if populate:
        (destination / "existing").write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        filesystem.publish_directory(staged, destination)

    assert staged.joinpath("new").read_bytes() == b"new"
    assert destination.is_dir()
    assert destination.joinpath("existing").exists() is populate


def test_publish_directory_consumes_complete_staging_tree(tmp_path):
    staged = tmp_path / "staged"
    (staged / "empty" / "nested").mkdir(parents=True)
    (staged / "bytes").write_bytes(b"\x00\xff")
    destination = tmp_path / "destination"

    filesystem.publish_directory(staged, destination)

    assert not staged.exists()
    assert destination.joinpath("bytes").read_bytes() == b"\x00\xff"
    assert destination.joinpath("empty", "nested").is_dir()


def test_publish_directory_rejects_destination_within_staging(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()

    with pytest.raises(ValueError, match="within"):
        filesystem.publish_directory(staged, staged / "destination")

    assert staged.is_dir()


def test_competing_file_publishers_have_one_winner_without_overwrite(tmp_path):
    destination = tmp_path / "destination"
    barrier = threading.Barrier(2)
    outcomes = []

    def publish(index):
        staged = _staged_file(tmp_path, f"staged-{index}", str(index).encode())
        barrier.wait()
        try:
            filesystem.publish_file(staged, destination)
        except FileExistsError:
            outcomes.append(("exists", index, staged.exists()))
        else:
            outcomes.append(("published", index, staged.exists()))

    workers = [
        threading.Thread(target=publish, args=(index,))
        for index in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)

    assert not any(worker.is_alive() for worker in workers)
    assert sorted(item[0] for item in outcomes) == ["exists", "published"]
    winner = next(
        index for status, index, _exists in outcomes
        if status == "published"
    )
    assert destination.read_bytes() == str(winner).encode()
    assert next(
        exists for status, _index, exists in outcomes
        if status == "exists"
    )


def test_competing_directory_publishers_have_one_winner(tmp_path):
    destination = tmp_path / "destination"
    barrier = threading.Barrier(2)
    outcomes = []

    def publish(index):
        staged = tmp_path / f"staged-{index}"
        staged.mkdir()
        (staged / "winner").write_text(str(index), encoding="ascii")
        barrier.wait()
        try:
            filesystem.publish_directory(staged, destination)
        except FileExistsError:
            outcomes.append(("exists", index, staged.exists()))
        else:
            outcomes.append(("published", index, staged.exists()))

    workers = [
        threading.Thread(target=publish, args=(index,))
        for index in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)

    assert sorted(item[0] for item in outcomes) == ["exists", "published"]
    winner = next(
        index for status, index, _exists in outcomes
        if status == "published"
    )
    assert (
        destination.joinpath("winner").read_text(encoding="ascii")
        == str(winner)
    )


def test_publish_rejects_same_path_and_hardlink_alias(tmp_path):
    staged = _staged_file(tmp_path, "staged")
    with pytest.raises(ValueError, match="distinct"):
        filesystem.publish_file(staged, staged, replace=True)

    alias = tmp_path / "alias"
    try:
        os.link(staged, alias)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")
    with pytest.raises(ValueError, match="distinct"):
        filesystem.publish_file(staged, alias, replace=True)
    assert staged.exists() and alias.exists()


def test_publish_never_falls_back_to_cross_volume_copy(tmp_path, monkeypatch):
    staged = _staged_file(tmp_path, "staged")
    destination = tmp_path / "destination"

    def cross_volume(_source, _destination, *, replace):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(filesystem._backend, "publish_path", cross_volume)
    with pytest.raises(OSError) as raised:
        filesystem.publish_file(staged, destination)

    assert raised.value.errno == errno.EXDEV
    assert staged.exists()
    assert not destination.exists()


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_publish_file_rejects_non_bool_replace_before_flush_or_mutation(
        tmp_path, monkeypatch, value):
    staged = _staged_file(tmp_path, "staged", b"new")
    destination = _staged_file(tmp_path, "destination", b"existing")
    calls = []
    monkeypatch.setattr(
        filesystem._backend,
        "sync_file",
        lambda _path: calls.append("flush"),
    )

    with pytest.raises(TypeError, match="replace"):
        filesystem.publish_file(staged, destination, replace=value)

    assert calls == []
    assert staged.read_bytes() == b"new"
    assert destination.read_bytes() == b"existing"


def test_native_publication_failure_after_visible_move_does_not_imply_rollback(
        tmp_path, monkeypatch):
    staged = _staged_file(tmp_path, "staged")
    destination = tmp_path / "destination"
    original = filesystem._backend.publish_path

    def fail_after_visible(source, target, *, replace):
        original(source, target, replace=replace)
        raise OSError("publication completion uncertain")

    monkeypatch.setattr(
        filesystem._backend, "publish_path", fail_after_visible,
    )
    with pytest.raises(OSError, match="completion uncertain"):
        filesystem.publish_file(staged, destination)

    assert destination.read_bytes() == b"payload"
    assert not staged.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory barrier")
def test_post_publication_directory_barrier_failure_does_not_imply_rollback(
        tmp_path, monkeypatch):
    staged = _staged_file(tmp_path, "staged")
    destination = tmp_path / "destination"
    original = filesystem._backend.sync_directory

    def fail_after_visible(path):
        if Path(path) == tmp_path:
            raise OSError("persistence completion uncertain")
        return original(path)

    monkeypatch.setattr(
        filesystem._backend, "sync_directory", fail_after_visible,
    )
    with pytest.raises(OSError, match="completion uncertain"):
        filesystem.publish_file(staged, destination)

    assert destination.read_bytes() == b"payload"
    assert not staged.exists()


def test_shape_and_permission_failures_retain_native_errors(
        tmp_path, monkeypatch):
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(IsADirectoryError) as is_directory:
        filesystem.sync_file(directory)
    assert is_directory.value.errno == errno.EISDIR
    assert is_directory.value.filename == os.fspath(directory)
    staged_file = _staged_file(tmp_path, "file")
    with pytest.raises(NotADirectoryError) as not_directory:
        filesystem.publish_directory(staged_file, tmp_path / "published")
    assert not_directory.value.errno == errno.ENOTDIR
    assert not_directory.value.filename == os.fspath(staged_file)

    def denied(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(filesystem._backend, "sync_file", denied)
    with pytest.raises(PermissionError, match="denied"):
        filesystem.sync_file(staged_file)

    staged_directory = tmp_path / "staged-directory"
    staged_directory.mkdir()
    monkeypatch.setattr(
        filesystem._backend,
        "publish_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("directory publication denied"),
        ),
    )
    with pytest.raises(PermissionError, match="directory publication denied"):
        filesystem.publish_directory(
            staged_directory, tmp_path / "published-directory",
        )


def test_remove_file_missing_policy_and_shape_validation(tmp_path):
    cleanup = tmp_path / "cleanup"
    cleanup.mkdir()
    missing = tmp_path / "missing"
    assert not filesystem.remove_file(
        missing, cleanup_directory=cleanup, missing_ok=True,
    )
    with pytest.raises(FileNotFoundError):
        filesystem.remove_file(
            missing, cleanup_directory=cleanup, missing_ok=False,
        )
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(IsADirectoryError):
        filesystem.remove_file(
            directory, cleanup_directory=cleanup, missing_ok=False,
        )

    target = _staged_file(tmp_path, "target")
    assert filesystem.remove_file(
        target, cleanup_directory=cleanup, missing_ok=False,
    )
    assert not target.exists()


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_remove_file_rejects_non_bool_missing_ok_before_mutation(
        tmp_path, monkeypatch, value):
    cleanup = tmp_path / "cleanup"
    cleanup.mkdir()
    target = _staged_file(tmp_path, "target")
    calls = []
    monkeypatch.setattr(
        filesystem._backend,
        "remove_file",
        lambda *_args, **_kwargs: calls.append("remove"),
    )

    with pytest.raises(TypeError, match="missing_ok"):
        filesystem.remove_file(
            target, cleanup_directory=cleanup, missing_ok=value,
        )

    assert calls == []
    assert target.read_bytes() == b"payload"


def test_remove_file_missing_source_precedes_cleanup_validation(tmp_path):
    missing_source = tmp_path / "missing-source"
    missing_cleanup = tmp_path / "missing-cleanup"

    assert not filesystem.remove_file(
        missing_source,
        cleanup_directory=missing_cleanup,
        missing_ok=True,
    )
    with pytest.raises(FileNotFoundError) as raised:
        filesystem.remove_file(
            missing_source,
            cleanup_directory=missing_cleanup,
            missing_ok=False,
        )

    assert raised.value.filename == os.fspath(missing_source)


def test_remove_file_rejects_cross_filesystem_cleanup_before_mutation(
        tmp_path, monkeypatch):
    target = _staged_file(tmp_path, "target")
    cleanup = tmp_path / "cleanup"
    cleanup.mkdir()
    original_lstat = filesystem.os.lstat
    calls = []

    def lstat(path):
        result = original_lstat(path)
        return SimpleNamespace(
            st_mode=result.st_mode,
            st_dev=2 if path == os.fspath(cleanup) else 1,
        )

    monkeypatch.setattr(filesystem.os, "lstat", lstat)
    monkeypatch.setattr(
        filesystem._backend,
        "remove_file",
        lambda *_args, **_kwargs: calls.append("remove"),
    )

    with pytest.raises(OSError) as raised:
        filesystem.remove_file(target, cleanup_directory=cleanup)

    assert raised.value.errno == errno.EXDEV
    assert raised.value.filename == os.fspath(target)
    assert raised.value.filename2 == os.fspath(cleanup)
    assert calls == []
    assert target.read_bytes() == b"payload"


def test_ensure_directory_publishes_nested_components_and_rejects_file(
        tmp_path):
    target = tmp_path / "one" / "two"
    filesystem.ensure_directory(target)
    assert target.is_dir()
    filesystem.ensure_directory(target)

    file_path = _staged_file(tmp_path, "file")
    with pytest.raises(FileExistsError) as exists:
        filesystem.ensure_directory(file_path)
    assert exists.value.errno == errno.EEXIST
    assert exists.value.filename == os.fspath(file_path)


def test_ensure_directory_follows_existing_directory_symlinks(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    filesystem.ensure_directory(link)
    filesystem.ensure_directory(link / "nested")

    assert (real / "nested").is_dir()


def test_filesystem_and_paths_import_without_core_or_optional_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; import dryml.filesystem, dryml.paths; "
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name == 'dryml.core' or name.startswith('dryml.core.') "
            "or name in {'numpy', 'torch', 'tensorflow', 'jax'})))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "[]"
