import multiprocessing
import os
from pathlib import Path

import dill
import pytest

import dryml.filesystem as filesystem
from dryml.core.store.dir import DirStore
from dryml.core.store.records import StoreFormatRecord
from dryml.core.store.store import StoreAuthorityError


def _bootstrap_format_worker(root, ready, start, paused, resume, outcomes):
    """Race initial format replacement while one writer retains its temporary file."""
    target = os.path.join(os.path.abspath(root), "store-format.record")
    original_publish = filesystem.publish_file

    def pause_format_replacement(source, destination, *, replace=False):
        if os.path.abspath(destination) == target:
            paused.set()
            if not resume.wait(10):
                raise TimeoutError("format publication was not resumed")
        return original_publish(source, destination, replace=replace)

    filesystem.publish_file = pause_format_replacement
    ready.set()
    start.wait(10)
    try:
        store = DirStore(root)
        outcomes.put(("ok", Path(store.store_format_path).read_bytes()))
    except BaseException as error:  # pragma: no cover - asserted through parent result.
        outcomes.put((type(error).__name__, str(error)))


def test_new_store_has_only_the_current_format_gate(tmp_path):
    store = DirStore(tmp_path / "store")

    assert Path(store.store_format_path).is_file()
    assert not (Path(store.base_dir) / "objects").exists()
    assert Path(store._bootstrap_lock_path).is_file()
    assert {entry.name for entry in Path(store.base_dir).iterdir()} == {"store-format.record"}


def test_spawned_bootstrap_serializes_root_creation_and_format_publication(tmp_path):
    """Spawned constructors wait through a visible format temporary and share one gate."""
    ctx = multiprocessing.get_context("spawn")
    ready_one, ready_two, start = ctx.Event(), ctx.Event(), ctx.Event()
    paused, resume, outcomes = ctx.Event(), ctx.Event(), ctx.Queue()
    root = os.fspath(tmp_path / "source" / "store")
    first = ctx.Process(
        target=_bootstrap_format_worker,
        args=(root, ready_one, start, paused, resume, outcomes),
    )
    second = ctx.Process(
        target=_bootstrap_format_worker,
        args=(root, ready_two, start, paused, resume, outcomes),
    )
    first.start()
    second.start()
    try:
        assert ready_one.wait(10) and ready_two.wait(10)
        start.set()
        assert paused.wait(10)
        entries = tuple(Path(root).iterdir())
        assert not Path(root, "store-format.record").exists()
        assert len(entries) == 1 and entries[0].name.startswith(".store-")
    finally:
        resume.set()
        first.join(20)
        second.join(20)

    assert first.exitcode == second.exitcode == 0
    expected = StoreFormatRecord().to_bytes()
    assert [outcomes.get(timeout=2) for _ in range(2)] == [("ok", expected)] * 2
    readers = [DirStore(root) for _ in range(3)]
    assert [Path(store.store_format_path).read_bytes() for store in readers] == [expected] * 3
    assert {entry.name for entry in Path(root).iterdir()} == {"store-format.record"}


def test_failed_initial_format_publication_leaves_an_empty_root_recoverable(tmp_path, monkeypatch):
    """A failed initial replacement leaves no partial authority for a later constructor."""
    root = tmp_path / "source" / "store"
    target = os.fspath(root / "store-format.record")
    original_publish = filesystem.publish_file

    def fail_format_replacement(source, destination, *, replace=False):
        if os.path.abspath(destination) == target:
            raise OSError("interrupted format replacement")
        return original_publish(source, destination, replace=replace)

    with monkeypatch.context() as patch:
        patch.setattr(filesystem, "publish_file", fail_format_replacement)
        with pytest.raises(OSError, match="interrupted format replacement"):
            DirStore(root)

    assert root.is_dir()
    assert tuple(root.iterdir()) == ()
    store = DirStore(root)
    assert Path(store.store_format_path).read_bytes() == StoreFormatRecord().to_bytes()


def test_nonempty_ungated_store_fails_without_mutation(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    retired = root / "objects"
    retired.mkdir()

    with pytest.raises(StoreAuthorityError, match="store-format"):
        DirStore(root)

    assert retired.is_dir()


def test_previous_store_format_is_rejected_without_mutation(tmp_path):
    root = tmp_path / "previous"
    root.mkdir()
    gate = root / "store-format.record"
    payload = {
        "schema": "store-format",
        "version": 1,
        "format_version": 1,
    }
    original = (
        b"DRYML-STORE-RECORD/store-format/1\n"
        + dill.dumps(payload, protocol=5)
    )
    gate.write_bytes(original)

    with pytest.raises(StoreAuthorityError, match="Malformed Store record"):
        DirStore(root)

    assert gate.read_bytes() == original
