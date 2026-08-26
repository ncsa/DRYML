import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil

import pytest

from dryml.core2 import Object, Repo
from dryml.core2.store.zip import ZipStore, ZipStoreConflictError


_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "cdef_v1" / "store-fixture" / "zip-store.zip"


class ZipAtomicObject(Object):
    def __init__(self, value="value"):
        super().__init__()
        self.value = value


def _archive_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_v1_archive(path: Path):
    shutil.copy2(_FIXTURE_PATH, path)
    store = ZipStore(path)
    legacy = tuple(store.hydrate_index())[0]
    store.close()
    return legacy


def _concurrent_zip_commit(path: str, value: str, mode: str, ready, start, results) -> None:
    """Stage one root or alias in an independently opened archive and report its commit."""

    store = None
    try:
        store = ZipStore(path)
        ready.set()
        if not start.wait(timeout=30):
            results.put(("error", "start timeout"))
            return
        obj = ZipAtomicObject(value)
        if mode == "root":
            store.save_object(obj)
        else:
            store.write_aliases({value: obj.definition})
        store.commit()
        results.put(("published", mode, obj.definition.stable_hash()))
    except ZipStoreConflictError:
        results.put(("conflict", mode, None))
    except BaseException as exc:
        results.put(("error", mode, repr(exc)))
    finally:
        if store is not None:
            store.close()


def test_zipstore_hydrates_v1_fixture_and_read_only_flush_preserves_archive(tmp_path):
    path = tmp_path / "legacy.zip"
    legacy = _copy_v1_archive(path)
    before = _archive_digest(path)

    store = ZipStore(path)
    repo = Repo(stores=store)
    assert store.read_definition(legacy) == legacy
    assert tuple(store.hydrate_index()) == (legacy,)
    assert repo.get_alias("primary") == legacy
    assert repo.get_alias("secondary") == legacy
    repo.flush()
    repo.close(flush=True)

    assert _archive_digest(path) == before


def test_path_backed_zip_publication_keeps_previous_archive_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "objects.zip"
    store = ZipStore(path)
    repo = Repo(stores=store)
    first = ZipAtomicObject("first", repo=repo)
    repo.save_object(first)
    repo.flush()
    before = _archive_digest(path)

    second = ZipAtomicObject("second", repo=repo)
    repo.save_object(second)
    original_replace = os.replace

    def fail_archive_replace(source, destination):
        if os.fspath(destination) == os.fspath(path):
            raise OSError("injected archive replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_archive_replace)
    with pytest.raises(OSError, match="archive replacement"):
        repo.flush()
    assert _archive_digest(path) == before

    monkeypatch.setattr(os, "replace", original_replace)
    repo.flush()
    reopened = ZipStore(path)
    assert set(reopened.hydrate_index()) == {first.definition, second.definition}
    reopened.close()


def test_failed_extracted_root_publication_restores_archive_dirty_state(tmp_path, monkeypatch):
    path = tmp_path / "failed-root.zip"
    store = ZipStore(path)
    obj = ZipAtomicObject("failed-root")
    object_dir = store.object_dir(obj.definition)
    original_replace = os.replace

    def fail_root_replace(source, destination):
        if os.fspath(destination) == object_dir:
            raise OSError("injected extracted-root failure")
        return original_replace(source, destination)

    monkeypatch.setattr("dryml.core2.store.store.os.replace", fail_root_replace)

    with pytest.raises(OSError, match="extracted-root"):
        store.save_object(obj)

    assert not store._archive_dirty
    assert not path.exists()
    store.close()


def test_stale_zip_handle_cannot_erase_newer_roots_and_can_retry_after_reopen(tmp_path):
    path = tmp_path / "concurrent-roots.zip"
    first_store = ZipStore(path)
    stale_store = ZipStore(path)
    first = ZipAtomicObject("first")
    stale = ZipAtomicObject("stale")

    first_store.save_object(first)
    first_store.commit()
    after_first = _archive_digest(path)

    stale_store.save_object(stale)
    with pytest.raises(ZipStoreConflictError, match="reopen the Store"):
        stale_store.commit()
    assert _archive_digest(path) == after_first

    stale_store.close()
    retry_store = ZipStore(path)
    retry_store.save_object(stale)
    retry_store.commit()
    assert set(retry_store.hydrate_index()) == {first.definition, stale.definition}

    first_store.close()
    retry_store.close()


def test_stale_read_only_zip_commit_neither_conflicts_nor_rewrites(tmp_path):
    path = tmp_path / "read-only-stale.zip"
    initial_store = ZipStore(path)
    initial = ZipAtomicObject("initial")
    initial_store.save_object(initial)
    initial_store.commit()
    initial_store.close()

    read_only_store = ZipStore(path)
    writer_store = ZipStore(path)
    updated = ZipAtomicObject("updated")
    writer_store.save_object(updated)
    writer_store.commit()
    after_writer = _archive_digest(path)

    read_only_store.commit()
    assert _archive_digest(path) == after_writer

    read_only_store.close()
    writer_store.close()


def test_stale_zip_handle_cannot_erase_newer_aliases_and_can_retry_after_reopen(tmp_path):
    path = tmp_path / "concurrent-aliases.zip"
    target = ZipAtomicObject("alias-target").definition
    first_store = ZipStore(path)
    stale_store = ZipStore(path)

    first_store.write_aliases({"first": target})
    first_store.commit()
    after_first = _archive_digest(path)

    stale_store.write_aliases({"stale": target})
    with pytest.raises(ZipStoreConflictError, match="reopen the Store"):
        stale_store.commit()
    assert _archive_digest(path) == after_first

    stale_store.close()
    retry_store = ZipStore(path)
    aliases = retry_store.read_aliases()
    retry_store.write_aliases({**aliases, "stale": target})
    retry_store.commit()
    assert retry_store.read_aliases() == {"first": target, "stale": target}

    first_store.close()
    retry_store.close()


@pytest.mark.parametrize("mode", ["root", "alias"])
def test_path_backed_zip_publication_serializes_across_processes(tmp_path, mode):
    path = tmp_path / f"processes-{mode}.zip"
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    ready = [ctx.Event(), ctx.Event()]
    processes = [
        ctx.Process(
            target=_concurrent_zip_commit,
            args=(str(path), value, mode, ready[index], start, results),
        )
        for index, value in enumerate(("first", "second"))
    ]
    for process in processes:
        process.start()
    try:
        for event in ready:
            assert event.wait(timeout=30)
        start.set()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        outcomes = [results.get(timeout=1) for _ in processes]
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert sorted(status for status, _, _ in outcomes) == ["conflict", "published"]
    published_hash = next(value for status, _, value in outcomes if status == "published")
    reopened = ZipStore(path)
    if mode == "root":
        assert {cdef.stable_hash() for cdef in reopened.hydrate_index()} == {published_hash}
    else:
        assert {cdef.stable_hash() for cdef in reopened.read_aliases().values()} == {published_hash}
    reopened.close()
