import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

from dryml.core2 import Object, Repo
from dryml.core2.store.zip import ZipStore
from dryml.core2.utils.general import pickle_save, unpickler


_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "cdef_v1" / "manifest.json"


class ZipAtomicObject(Object):
    def __init__(self, value="value"):
        super().__init__()
        self.value = value


def _fixture_v1_definition():
    manifest = json.loads(_FIXTURE_PATH.read_text())
    return unpickler(base64.b64decode(manifest["payload"]))["object_root"]


def _archive_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_v1_archive(path: Path):
    store = ZipStore(path)
    legacy = _fixture_v1_definition()
    root = Path(store.object_dir(legacy))
    root.mkdir(parents=True, exist_ok=True)
    pickle_save(legacy, root / "def.pkl")
    pickle_save(legacy, Path(store.base_dir) / "def.pkl")
    pickle_save({"legacy": legacy}, Path(store.base_dir) / "aliases.pkl")
    store._mark_authority_dirty()
    store.commit()
    store.close()
    return legacy


def test_zipstore_hydrates_v1_fixture_and_read_only_flush_preserves_archive(tmp_path):
    path = tmp_path / "legacy.zip"
    legacy = _seed_v1_archive(path)
    before = _archive_digest(path)

    store = ZipStore(path)
    repo = Repo(stores=store)
    assert store.read_definition(legacy) == legacy
    assert tuple(store.hydrate_index()) == (legacy,)
    assert repo.get_alias("legacy") == legacy
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
