import hashlib
import os
from pathlib import Path
import shutil

import pytest

from dryml.core2 import Object, Repo
from dryml.core2.store.zip import ZipStore


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
