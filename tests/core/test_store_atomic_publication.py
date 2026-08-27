import base64
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import threading
import time
import zipfile

import pytest

from dryml.core import ConcreteDefinition, Object, Repo
from dryml.core.cdef_identity import V1_IDENTITY_VERSION
from dryml.core.query.model import QueryIndexError
from dryml.core.repo import load_alias, load_object
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError
from dryml.core.store.zip import ZipStore
from dryml.core.utils.general import pickle_save, unpickler


_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "cdef_v1" / "manifest.json"
_STORE_FIXTURE_ROOT = _FIXTURE_PATH.parent / "store-fixture"
_V1_FIXTURE_PRODUCER = "85ea268860091f96b97fa9031ac813beb369c749"


def _retired_module() -> str:
    """Build the unsupported former namespace without retaining it as source text."""

    return "dryml.core" + "2"


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


class BlockingRestoreAtomicStoreObject(MultiFileAtomicStoreObject):
    def __init__(self, ready_path, release_path, name="state"):
        super().__init__(name)
        self.ready_path = str(ready_path)
        self.release_path = str(release_path)

    def restore_state_from_dir_imp(self, src_dir, revision=None):
        super().restore_state_from_dir_imp(src_dir, revision=revision)
        Path(self.ready_path).touch()
        deadline = time.monotonic() + 30
        while not Path(self.release_path).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("reader release was not signalled")
            time.sleep(0.01)


def _restore_while_blocked(store_path, ready_path, release_path, name, results):
    """Restore one state generation while retaining the Store reader lease."""

    try:
        store = DirStore(store_path, query_index="memory")
        restored = BlockingRestoreAtomicStoreObject(ready_path, release_path, name)
        store.restore_object(restored)
        results.put(("ok", restored.left, restored.right))
    except BaseException as exc:
        results.put(("error", repr(exc)))


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _generation_names(store, cdef):
    generations_dir = Path(store.object_dir(cdef), ".state-generations")
    if not generations_dir.exists():
        return set()
    return {path.name for path in generations_dir.iterdir() if path.is_dir()}


def _wait_for_path(path: Path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _fixture_v1_definition():
    manifest = json.loads(_FIXTURE_PATH.read_text())
    payload = base64.b64decode(manifest["payload"], validate=True)
    assert hashlib.sha256(payload).hexdigest() == manifest["payload_sha256"]
    assert payload[:2] == b"\x80\x05"
    return unpickler(payload)["object_root"]


def _load_v1_store_fixture_manifest():
    manifest = json.loads((_STORE_FIXTURE_ROOT / "manifest.json").read_text())
    assert manifest["format"] == "pre-v2-store-fixture/v1"
    assert manifest["producer"] == {
        "commit": _V1_FIXTURE_PRODUCER,
        "dill": "unrecorded",
        "pickle_protocol": 5,
        "python": "3.8 (declared by dev-env.yaml)",
        "runtime": "unrecorded",
    }
    assert _tree_digest(_STORE_FIXTURE_ROOT / "dir-store") == manifest["dir_store_tree_sha256"]
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((_STORE_FIXTURE_ROOT / relative).read_bytes()).hexdigest() == expected
    return manifest


def _write_root(store: DirStore, cdef, *, fanout=None, digest=None):
    digest = cdef.stable_hash() if digest is None else digest
    fanout = digest[:2] if fanout is None else fanout
    path = Path(store.object_root_dir) / fanout / digest / "def.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle_save(cdef, path)
    return path


def test_historical_fixture_hydration_rejects_without_changing_authoritative_tree(tmp_path):
    """An old Store fails closed and leaves its authoritative files untouched."""

    _load_v1_store_fixture_manifest()

    shutil.copytree(_STORE_FIXTURE_ROOT / "dir-store", tmp_path / "store")
    store_path = tmp_path / "store"
    before = _tree_digest(store_path)

    with pytest.raises(ModuleNotFoundError, match=_retired_module()):
        DirStore(store_path, query_index="memory")

    assert _tree_digest(store_path) == before


@pytest.mark.parametrize("store_type", ["directory", "zip"])
def test_historical_fixture_materialization_fails_without_authoritative_rewrite(tmp_path, store_type):
    """Retired persisted globals are not remapped through main or alias routes."""

    _load_v1_store_fixture_manifest()
    if store_type == "directory":
        path = tmp_path / "store"
        shutil.copytree(_STORE_FIXTURE_ROOT / "dir-store", path)
        digest = lambda: _tree_digest(path)
    else:
        path = tmp_path / "store.zip"
        shutil.copy2(_STORE_FIXTURE_ROOT / "zip-store.zip", path)
        digest = lambda: hashlib.sha256(path.read_bytes()).hexdigest()
    before = digest()
    with pytest.raises(ModuleNotFoundError, match=_retired_module()):
        store = DirStore(path, query_index="memory") if store_type == "directory" else ZipStore(path)
        Repo(stores=store)
    assert digest() == before


def test_read_only_multistore_hydration_does_not_copy_main_definition(tmp_path):
    default_store = DirStore(tmp_path / "default", query_index="memory")
    source_store = DirStore(tmp_path / "source", query_index="memory")
    legacy = AtomicStoreObject("main-definition").definition
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


def test_repeated_state_saves_reclaim_inactive_generations(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    obj = MultiFileAtomicStoreObject("bounded-generations")
    store.save_object(obj)

    for generation in range(1, 6):
        obj.left = f"left-v{generation + 1}"
        obj.right = f"right-v{generation + 1}"
        store.save_object(obj)
        assert len(_generation_names(store, obj.definition)) == 1

    restored = MultiFileAtomicStoreObject("bounded-generations")
    store.restore_object(restored)
    assert (restored.left, restored.right) == ("left-v6", "right-v6")


def test_reader_can_finish_before_writer_reclaims_previous_generation(tmp_path):
    store_path = tmp_path / "store"
    ready_path = tmp_path / "reader-ready"
    release_path = tmp_path / "reader-release"
    store = DirStore(store_path, query_index="memory")
    obj = BlockingRestoreAtomicStoreObject(str(ready_path), str(release_path), "reader-lease")
    store.save_object(obj)
    obj.left = "left-v2"
    obj.right = "right-v2"

    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    reader = ctx.Process(
        target=_restore_while_blocked,
        args=(str(store_path), str(ready_path), str(release_path), "reader-lease", results),
    )
    reader.start()
    writer_errors = []
    writer = threading.Thread(target=lambda: _save_or_capture(store, obj, writer_errors))
    try:
        assert _wait_for_path(ready_path)
        writer.start()
        time.sleep(0.2)
        assert writer.is_alive()
        release_path.touch()
        writer.join(timeout=10)
        assert not writer.is_alive()
        assert not writer_errors
        assert results.get(timeout=10) == ("ok", "left-v1", "right-v1")
        reader.join(timeout=10)
        assert reader.exitcode == 0
    finally:
        release_path.touch()
        if writer.is_alive():
            writer.join(timeout=10)
        if reader.is_alive():
            reader.terminate()
            reader.join(timeout=10)

    assert len(_generation_names(store, obj.definition)) == 1
    restored = BlockingRestoreAtomicStoreObject(str(ready_path), str(release_path), "reader-lease")
    store.restore_object(restored)
    assert (restored.left, restored.right) == ("left-v2", "right-v2")


def _save_or_capture(store, obj, errors):
    """Save in a test thread while preserving an assertion-friendly error value."""

    try:
        store.save_object(obj)
    except BaseException as exc:
        errors.append(exc)


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


def test_interruption_before_pointer_publication_removes_unreachable_generation(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="memory")
    obj = MultiFileAtomicStoreObject("before-pointer")
    store.save_object(obj)
    obj.left = "left-v2"
    obj.right = "right-v2"
    pointer_path = os.path.join(store.object_dir(obj.definition), ".state-current.pkl")
    original_replace = os.replace

    def interrupt_before_pointer_replace(source, destination):
        if os.fspath(destination) == pointer_path:
            raise KeyboardInterrupt("injected before pointer replacement")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt_before_pointer_replace)
    with pytest.raises(KeyboardInterrupt, match="before pointer"):
        store.save_object(obj)

    restored = MultiFileAtomicStoreObject("before-pointer")
    store.restore_object(restored)
    assert (restored.left, restored.right) == ("left-v1", "right-v1")
    assert not _generation_names(store, obj.definition)


def test_interruption_after_pointer_publication_remains_recoverable_and_is_reclaimed(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="memory")
    obj = MultiFileAtomicStoreObject("after-pointer")
    store.save_object(obj)
    obj.left = "left-v2"
    obj.right = "right-v2"
    store.save_object(obj)
    obj.left = "left-v3"
    obj.right = "right-v3"
    pointer_path = os.path.join(store.object_dir(obj.definition), ".state-current.pkl")
    original_replace = os.replace

    def interrupt_after_pointer_replace(source, destination):
        result = original_replace(source, destination)
        if os.fspath(destination) == pointer_path:
            raise KeyboardInterrupt("injected after pointer replacement")
        return result

    monkeypatch.setattr(os, "replace", interrupt_after_pointer_replace)
    with pytest.raises(KeyboardInterrupt, match="after pointer"):
        store.save_object(obj)

    restored = MultiFileAtomicStoreObject("after-pointer")
    store.restore_object(restored)
    assert (restored.left, restored.right) == ("left-v3", "right-v3")
    assert len(_generation_names(store, obj.definition)) == 2

    monkeypatch.setattr(os, "replace", original_replace)
    obj.left = "left-v4"
    obj.right = "right-v4"
    store.save_object(obj)
    assert len(_generation_names(store, obj.definition)) == 1


def test_zipstore_archives_only_the_active_state_generation(tmp_path):
    path = tmp_path / "states.zip"
    store = ZipStore(path)
    obj = MultiFileAtomicStoreObject("zip-generations")
    store.save_object(obj)
    for generation in range(1, 4):
        obj.left = f"left-v{generation + 1}"
        obj.right = f"right-v{generation + 1}"
        store.save_object(obj)
    store.commit()

    with zipfile.ZipFile(path) as archive:
        state_members = [
            name for name in archive.namelist()
            if "/.state-generations/" in name
        ]
    generations = {Path(name).parts[-2] for name in state_members}
    assert len(generations) == 1

    reopened = ZipStore(path)
    restored = MultiFileAtomicStoreObject("zip-generations")
    reopened.restore_object(restored)
    assert (restored.left, restored.right) == ("left-v4", "right-v4")
    reopened.close()


def test_zipstore_does_not_ship_unreachable_generation_after_pointer_interruption(tmp_path, monkeypatch):
    path = tmp_path / "interrupted-states.zip"
    store = ZipStore(path)
    obj = MultiFileAtomicStoreObject("zip-interrupted-generations")
    store.save_object(obj)
    obj.left = "left-v2"
    obj.right = "right-v2"
    store.save_object(obj)
    obj.left = "left-v3"
    obj.right = "right-v3"
    pointer_path = os.path.join(store.object_dir(obj.definition), ".state-current.pkl")
    original_replace = os.replace

    def interrupt_after_pointer_replace(source, destination):
        result = original_replace(source, destination)
        if os.fspath(destination) == pointer_path:
            raise KeyboardInterrupt("injected ZipStore pointer interruption")
        return result

    monkeypatch.setattr(os, "replace", interrupt_after_pointer_replace)
    with pytest.raises(KeyboardInterrupt, match="ZipStore pointer"):
        store.save_object(obj)
    assert len(_generation_names(store, obj.definition)) == 2

    store.commit()
    with zipfile.ZipFile(path) as archive:
        state_members = [
            name for name in archive.namelist()
            if "/.state-generations/" in name
        ]
    assert len({Path(name).parts[-2] for name in state_members}) == 1
    store.close()


@pytest.mark.parametrize(
    ("destination", "reference_name"),
    [("aliases.pkl", "alias"), ("def.pkl", "main definition")],
)
def test_reference_publication_failure_preserves_seeded_authority_and_retries(
        tmp_path, monkeypatch, destination, reference_name):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    previous = AtomicStoreObject("previous", repo=repo)
    replacement = AtomicStoreObject("replacement", repo=repo)
    repo.save_object(previous)
    repo.save_object(replacement)
    store.write_main_def(previous.definition)
    store.write_aliases({"root": previous.definition})

    repo = Repo(stores=store)
    repo.set_main_def(replacement.definition)
    repo.set_alias("root", replacement.definition, save_live=False)
    reference_path = Path(store.base_dir) / destination
    before = reference_path.read_bytes()
    original_replace = os.replace
    attempted = []

    def fail_reference_replace(source, destination):
        attempted.append(os.fspath(destination))
        if os.fspath(destination) == os.fspath(reference_path):
            raise OSError(f"injected {reference_name} publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_reference_replace)
    with pytest.raises(OSError, match=f"{reference_name} publication"):
        repo.flush()
    assert os.fspath(reference_path) in attempted
    assert reference_path.read_bytes() == before
    if destination == "aliases.pkl":
        assert store.read_aliases() == {"root": previous.definition}
        assert store.read_main_def() == previous.definition
    else:
        assert store.read_aliases() == {"root": replacement.definition}
        assert store.read_main_def() == previous.definition

    def record_reference_replace(source, destination):
        attempted.append(os.fspath(destination))
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", record_reference_replace)
    attempted.clear()
    repo.flush()
    assert os.fspath(reference_path) in attempted
    if destination == "def.pkl":
        assert os.fspath(Path(store.base_dir) / "aliases.pkl") not in attempted
    assert store.read_main_def() == replacement.definition
    assert store.read_aliases() == {"root": replacement.definition}


@pytest.mark.parametrize("store_type", ["directory", "zip"])
def test_reference_writers_reject_malformed_payloads_without_changing_bytes_or_caches(tmp_path, store_type):
    path = tmp_path / ("store" if store_type == "directory" else "store.zip")
    store = DirStore(path, query_index="memory") if store_type == "directory" else ZipStore(path)
    valid = AtomicStoreObject("valid-reference").definition
    store.write_main_def(valid)
    store.write_aliases({"valid": valid})
    store.commit()

    main_path = Path(store.base_dir) / "def.pkl"
    aliases_path = Path(store.base_dir) / "aliases.pkl"
    before_main = main_path.read_bytes()
    before_aliases = aliases_path.read_bytes()
    before_archive = path.read_bytes() if store_type == "zip" else None
    repo = Repo(stores=store)
    invalid_alias_object = AtomicStoreObject("invalid-save-alias")

    with pytest.raises(StoreAuthorityError):
        store.write_main_def(object())
    with pytest.raises(StoreAuthorityError):
        store.write_aliases({"": valid})
    with pytest.raises(StoreAuthorityError):
        store.write_aliases({"valid": object()})
    with pytest.raises(TypeError):
        repo.set_main_def(object())
    with pytest.raises(ValueError):
        repo.set_alias("", valid)
    with pytest.raises(ValueError):
        repo.save_object(invalid_alias_object, alias="")

    assert store.read_main_def() == valid
    assert store.read_aliases() == {"valid": valid}
    assert repo.main_def == valid
    assert repo.aliases() == {"valid": valid}
    assert not store.has(invalid_alias_object.definition)
    assert main_path.read_bytes() == before_main
    assert aliases_path.read_bytes() == before_aliases

    store.commit()
    if before_archive is not None:
        assert path.read_bytes() == before_archive
