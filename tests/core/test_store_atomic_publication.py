"""Native format-4 publication, byte completeness and legacy-consumer refusal.

The previous V3 snapshot/alias cases deliberately cut over to Node/state/name
authority. Unmigrated Repo/reference writers remain explicitly unsupported.
"""

import os
from pathlib import Path

import pytest

import dryml.filesystem as filesystem
from dryml.core import Definition, Object, Repo, StateRef
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord, MainRefRecord, NodeNameRecord
from dryml.core.store.authority import StoreMutationError
from dryml.core.store.store import StoreAuthorityError, StoreCapabilityError


class AtomicRecordObject(Object):
    def __init__(self, value=1):
        self.value = value


def creation(value=1):
    return Definition(AtomicRecordObject, value=value).concretize()


def test_immutable_creation_collision_is_idempotent_only_after_full_validation(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    digest = store.install_creation(creation())
    assert store.install_creation(creation()) == digest
    path = Path(store._bucket_path(digest), "def.pkl")
    path.write_bytes(b"not a record")
    with pytest.raises(StoreAuthorityError, match="Malformed Store record"):
        store.install_creation(creation())
    assert path.read_bytes() == b"not a record"


def test_direct_membership_read_validates_only_requested_authority(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    first, other = (store.publish_node(creation(value)) for value in (1, 2))
    Path(store._node_path("membership/recipe", other.digest)).write_bytes(b"corrupt unrelated")
    assert store.read_node(first.digest).graph_equal(creation())
    assert store.read_node("0" * 64) is None
    Path(store._node_path("definitions/roots", first.digest)).unlink()
    with pytest.raises(StoreAuthorityError, match="Missing or mismatched"):
        store.read_node(first.digest)


def test_short_record_write_never_publishes_truncated_authority(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    before = store.authority_cut()
    original = os.fdopen
    class ShortWriter:
        def __init__(self, file):
            self.file = file
        def __enter__(self):
            self.file.__enter__()
            return self
        def __exit__(self, *args):
            return self.file.__exit__(*args)
        def write(self, data):
            return self.file.write(data[:-1])
    monkeypatch.setattr(os, "fdopen", lambda *a, **kw: ShortWriter(original(*a, **kw)))
    with pytest.raises(StoreMutationError, match="incomplete"):
        store.install_creation(creation())
    assert store.authority_cut() == before
    assert not Path(store.base_dir, "states").exists()
    assert not Path(store.base_dir, "mutation-prepare.record").exists()


def test_publication_delegates_to_native_api_after_durable_intent(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    observed = []
    original = filesystem.publish_file
    def publish(source, destination, *, replace=False):
        relative = str(Path(destination).relative_to(store.base_dir))
        observed.append(relative)
        if relative.endswith("def.pkl"):
            pending = store.pending_mutations()
            assert len(pending) == 1 and pending[0].intent["status"] == "pending"
            assert pending[0].intent["transitions"]
        return original(source, destination, replace=replace)
    monkeypatch.setattr(filesystem, "publish_file", publish)
    digest = store.install_creation(creation())
    header_index = observed.index(str(Path("states", digest[:2], digest, "def.pkl")))
    assert observed.index("mutation-prepare.record") < observed.index("authority.record") < header_index
    assert any(name.startswith("mutations/") for name in observed[:header_index])
    assert store.authority_cut().pending == ()


def test_dirstore_can_publish_beneath_symlinked_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    store = DirStore(link / "store", query_index="none")
    digest = store.install_creation(creation())
    assert store.read_creation(digest).graph_equal(creation())
    assert (real / "store").is_dir()


def test_filesystem_capability_failure_retains_actionable_cause(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    def fail(*args, **kwargs):
        raise filesystem.FilesystemCapabilityError("exclusive rename unavailable")
    monkeypatch.setattr(filesystem, "publish_file", fail)
    with pytest.raises(StoreMutationError, match="exclusive rename unavailable") as raised:
        store.install_creation(creation())
    assert isinstance(raised.value.__cause__, StoreCapabilityError)
    assert not Path(store.base_dir, "states").exists()


def test_failed_header_replace_has_replayable_metadata_without_recapture(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    original = filesystem.publish_file
    def fail(source, destination, *, replace=False):
        if Path(destination).name == "def.pkl":
            raise OSError("injected header replacement failure")
        return original(source, destination, replace=replace)
    with monkeypatch.context() as patch:
        patch.setattr(filesystem, "publish_file", fail)
        with pytest.raises(StoreMutationError, match="replacement") as raised:
            store.install_creation(creation())
    monkeypatch.setattr(store, "_prepare_node_roots", lambda *a, **kw: pytest.fail("repair prepared a graph"))
    cut = store.replay_mutation(raised.value.mutation_id)
    assert not cut.pending
    from dryml.core.node_views import CREATION_VIEW, view_digest
    digest = view_digest(creation(), CREATION_VIEW)
    assert store.read_creation(digest).graph_equal(creation())


@pytest.mark.parametrize("failure", [OSError, PermissionError])
def test_mutable_name_replacement_keeps_previous_complete_record(tmp_path, monkeypatch, failure):
    store = DirStore(tmp_path / "store", query_index="none")
    first, second = (store.publish_node(creation(value)) for value in (1, 2))
    previous, replacement = (NodeNameRecord("main", "recipe", r.digest) for r in (first, second))
    store.compare_and_set_node_name(previous, expected=None)
    original = filesystem.publish_file
    def fail(source, destination, *, replace=False):
        if Path(destination) == Path(store.base_dir, "names", "main.record"):
            raise failure("injected name replacement denial")
        return original(source, destination, replace=replace)
    monkeypatch.setattr(filesystem, "publish_file", fail)
    with pytest.raises(StoreMutationError, match="denial"):
        store.compare_and_set_node_name(replacement, expected=previous)
    assert store.read_node_name("main") == previous


@pytest.mark.parametrize("operation", ["definition", "main", "metadata", "delete-metadata", "snapshot", "Repo", "query", "index"])
def test_legacy_publication_and_coverage_are_refused_without_effects(tmp_path, operation):
    store = DirStore(tmp_path / "store", query_index="none")
    before = store.authority_cut()
    actions = {
        "definition": lambda: store.write_definition_record(DefinitionRecord(creation())),
        "main": lambda: store.write_main_ref(MainRefRecord("1" * 64)),
        "metadata": lambda: store.write_metadata(AtomicRecordObject().object_ref, {"x": 1}),
        "delete-metadata": lambda: store.delete_metadata(AtomicRecordObject().object_ref),
        "snapshot": lambda: store.publish_snapshot(StateRef(AtomicRecordObject().object_ref, {}), evidence=None, local_states={}),
        "Repo": lambda: Repo(store).save_object(AtomicRecordObject()),
        "query": store.query,
        "index": lambda: DirStore(store.base_dir, query_index="sqlite").open_query_index(),
    }
    with pytest.raises(StoreCapabilityError):
        actions[operation]()
    assert store.authority_cut() == before
    assert not Path(store.base_dir, "mutations").exists()
    assert not Path(store.base_dir, "states").exists()
