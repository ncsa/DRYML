import base64
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from dryml.core2 import ConcreteDefinition, Object, Repo
from dryml.core2.query.model import QueryIndexError
from dryml.core2.store.dir import DirStore
from dryml.core2.utils.general import pickle_save, unpickler


_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "cdef_v1" / "manifest.json"
_STORE_FIXTURE_ROOT = _FIXTURE_PATH.parent / "store-fixture"


class AtomicStoreObject(Object):
    def __init__(self, value="value"):
        super().__init__()
        self.value = value


class FailingAtomicStoreObject(AtomicStoreObject):
    def save_state_to_dir_imp(self, dest_dir, revision=None):
        raise RuntimeError("injected state write failure")


class LegacyDefaultObject(Object):
    def __init__(self, value=3):
        super().__init__()
        self.value = value


class MultiFileAtomicStoreObject(Object):
    def __init__(self, name="state"):
        super().__init__()
        self.name = name
        self.left = "left-v1"
        self.right = "right-v1"

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        Path(dest_dir, "left.txt").write_text(self.left)
        Path(dest_dir, "right.txt").write_text(self.right)

    def restore_state_from_dir_imp(self, src_dir, revision=None):
        self.left = Path(src_dir, "left.txt").read_text()
        self.right = Path(src_dir, "right.txt").read_text()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _fixture_v1_definition():
    manifest = json.loads(_FIXTURE_PATH.read_text())
    return unpickler(base64.b64decode(manifest["payload"]))["object_root"]


def _write_root(store: DirStore, cdef, *, fanout=None, digest=None):
    digest = cdef.stable_hash() if digest is None else digest
    fanout = digest[:2] if fanout is None else fanout
    path = Path(store.object_root_dir) / fanout / digest / "def.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle_save(cdef, path)
    return path


def test_v1_fixture_hydration_and_read_only_flush_preserve_authoritative_tree(tmp_path):
    fixture_manifest = json.loads((_STORE_FIXTURE_ROOT / "manifest.json").read_text())
    for relative, expected in fixture_manifest["files"].items():
        assert hashlib.sha256((_STORE_FIXTURE_ROOT / relative).read_bytes()).hexdigest() == expected

    shutil.copytree(_STORE_FIXTURE_ROOT / "dir-store", tmp_path / "store")
    store = DirStore(tmp_path / "store", query_index="memory")
    legacy = store.read_main_def()
    before = _tree_digest(Path(store.base_dir))

    reopened = DirStore(store.base_dir, query_index="memory")
    repo = Repo(stores=reopened)
    assert tuple(reopened.hydrate_index()) == (legacy,)
    assert reopened.read_definition(legacy) == legacy
    assert repo.get_alias("primary") == legacy
    assert repo.get_alias("secondary") == legacy
    repo.flush()
    repo.close(flush=True)

    assert _tree_digest(Path(store.base_dir)) == before


def test_read_only_multistore_hydration_does_not_copy_main_definition(tmp_path):
    default_store = DirStore(tmp_path / "default", query_index="memory")
    source_store = DirStore(tmp_path / "source", query_index="memory")
    legacy = _fixture_v1_definition()
    source_store.write_main_def(legacy)
    before_default = _tree_digest(Path(default_store.base_dir))
    before_source = _tree_digest(Path(source_store.base_dir))

    repo = Repo(stores=[default_store, source_store])
    assert repo.main_def == legacy
    repo.flush()

    assert default_store.read_main_def() is None
    assert _tree_digest(Path(default_store.base_dir)) == before_default
    assert _tree_digest(Path(source_store.base_dir)) == before_source


@pytest.mark.parametrize(
    "fanout,digest",
    [
        ("00", "a" * 64),
        ("aa", "a" * 63),
        ("zz", "z" * 64),
    ],
)
def test_hydration_rejects_invalid_root_path_before_catalog_mutation(tmp_path, fanout, digest):
    store = DirStore(tmp_path / "store", query_index="memory")
    _write_root(store, AtomicStoreObject("valid").definition, fanout=fanout, digest=digest)

    with pytest.raises(QueryIndexError):
        tuple(store.hydrate_index())


def test_hydration_rejects_changed_definition_and_duplicate_location(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    first = AtomicStoreObject("first").definition
    second = AtomicStoreObject("second").definition
    _write_root(store, second, digest=first.stable_hash())

    with pytest.raises(QueryIndexError, match="wrong stable-hash"):
        tuple(store.hydrate_index())

    changed_root = Path(store.object_root_dir) / first.stable_hash()[:2] / first.stable_hash()
    changed_root.mkdir(parents=True, exist_ok=True)
    pickle_save(first, changed_root / "def.pkl")
    duplicate = Path(store.object_root_dir) / "ff" / first.stable_hash() / "def.pkl"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    pickle_save(first, duplicate)
    with pytest.raises(QueryIndexError):
        tuple(store.hydrate_index())


def test_v1_materialization_uses_current_omitted_default_without_inference():
    legacy = ConcreteDefinition._from_persisted_record(LegacyDefaultObject, (), {})
    old_defaults = LegacyDefaultObject.__init__.__defaults__
    LegacyDefaultObject.__init__.__defaults__ = (4,)
    try:
        loaded = Repo().load_or_build(legacy, restore_state=False, cache="none")
    finally:
        LegacyDefaultObject.__init__.__defaults__ = old_defaults

    assert loaded.value == 4


def test_new_root_is_not_visible_when_state_or_final_replace_fails(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    failed = FailingAtomicStoreObject("state", repo=repo)

    with pytest.raises(RuntimeError, match="state write"):
        repo.save_object(failed)
    assert not store.has(failed.definition)

    obj = AtomicStoreObject("replace", repo=repo)
    final_root = store.object_dir(obj.definition)
    original_replace = os.replace

    def fail_final_replace(source, destination):
        if os.fspath(destination) == final_root:
            raise OSError("injected root publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_final_replace)
    with pytest.raises(OSError, match="root publication"):
        repo.save_object(obj)
    assert not store.has(obj.definition)

    monkeypatch.setattr(os, "replace", original_replace)
    repo.save_object(obj)
    assert store.has(obj.definition)


def test_existing_root_state_switches_as_one_generation(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="memory")
    obj = MultiFileAtomicStoreObject()
    store.save_object(obj)
    object_dir = Path(store.object_dir(obj.definition))
    before = _tree_digest(object_dir)

    obj.left = "left-v2"
    obj.right = "right-v2"
    original_replace = os.replace

    def fail_pointer_replace(source, destination):
        if os.fspath(destination) == os.fspath(object_dir / ".state-current.pkl"):
            raise OSError("injected state generation publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_pointer_replace)
    with pytest.raises(OSError, match="state generation"):
        store.save_object(obj)

    assert not (object_dir / ".state-current.pkl").exists()
    assert (object_dir / "left.txt").read_text() == "left-v1"
    assert (object_dir / "right.txt").read_text() == "right-v1"
    assert _tree_digest(object_dir) == before

    monkeypatch.setattr(os, "replace", original_replace)
    store.save_object(obj)
    restored = MultiFileAtomicStoreObject()
    store.restore_object(restored)

    assert restored.left == "left-v2"
    assert restored.right == "right-v2"


def test_new_root_interruption_after_replace_retains_root_and_dirty_token(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    obj = AtomicStoreObject("published-before-interrupt")
    object_dir = store.object_dir(obj.definition)
    original_replace = os.replace

    def interrupt_after_replace(source, destination):
        result = original_replace(source, destination)
        if os.fspath(destination) == object_dir:
            raise KeyboardInterrupt("injected after root replacement")
        return result

    monkeypatch.setattr(os, "replace", interrupt_after_replace)

    with pytest.raises(KeyboardInterrupt, match="root replacement"):
        store.save_object(obj)

    assert store.read_definition(obj.definition) == obj.definition
    assert store.query_index_is_dirty()


def test_existing_root_interruption_after_pointer_retains_generation_and_token(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    obj = MultiFileAtomicStoreObject("interrupted-pointer")
    store.save_object(obj)
    store.clear_query_index_dirty()
    obj.left = "left-v2"
    obj.right = "right-v2"
    pointer_path = os.path.join(store.object_dir(obj.definition), ".state-current.pkl")
    original_replace = os.replace

    def interrupt_after_replace(source, destination):
        result = original_replace(source, destination)
        if os.fspath(destination) == pointer_path:
            raise KeyboardInterrupt("injected after pointer replacement")
        return result

    monkeypatch.setattr(os, "replace", interrupt_after_replace)

    with pytest.raises(KeyboardInterrupt, match="pointer replacement"):
        store.save_object(obj)

    restored = MultiFileAtomicStoreObject("interrupted-pointer")
    store.restore_object(restored)
    assert restored.left == "left-v2"
    assert restored.right == "right-v2"
    assert store.query_index_is_dirty()


def test_alias_and_main_references_publish_only_after_explicit_mutation(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = AtomicStoreObject("references", repo=repo)
    repo.save_object(obj)
    repo.set_main_def(obj.definition)
    repo.set_alias("root", obj.definition)
    original_replace = os.replace

    def fail_reference_replace(source, destination):
        if os.fspath(destination) in {
            str(Path(store.base_dir) / "def.pkl"),
            str(Path(store.base_dir) / "aliases.pkl"),
        }:
            raise OSError("injected reference publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_reference_replace)
    with pytest.raises(OSError, match="reference publication"):
        repo.flush()
    assert not (Path(store.base_dir) / "def.pkl").exists()
    assert not (Path(store.base_dir) / "aliases.pkl").exists()

    monkeypatch.setattr(os, "replace", original_replace)
    repo.flush()
    assert store.read_main_def() == obj.definition
    assert store.read_aliases() == {"root": obj.definition}
