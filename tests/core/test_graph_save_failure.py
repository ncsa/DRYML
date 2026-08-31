from pathlib import Path

import pytest

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore


class FailingState(Serializable):
    calls = 0

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).calls += 1
        Path(dest_dir, "partial.txt").write_text("partial")
        raise RuntimeError("serializer failed")


class GoodState(Serializable):
    def __init__(self, value=1):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value.txt").write_text(str(self.value))


class EmptyState(Serializable):
    calls = []

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).calls.append(Path(dest_dir).is_dir() and not any(Path(dest_dir).iterdir()))


def test_hook_failure_leaves_no_state_ref_or_last_hash(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = FailingState(repo=repo)

    with pytest.raises(Exception, match="local state publication"):
        obj.save(repo=repo)

    assert obj._last_state_hash is None
    assert not (tmp_path / "store" / "state-refs").exists()


def test_alias_replacement_failure_preserves_completed_state_ref(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = GoodState(repo=repo)
    first = obj.save(repo=repo, alias="latest")

    def fail_alias(record):
        raise OSError("alias replacement failed")

    monkeypatch.setattr(store, "write_object_alias", fail_alias)
    obj.value = 2
    with pytest.raises(OSError, match="alias replacement"):
        obj.save(repo=repo, alias="latest", deep_capture=True)

    assert store.read_state_ref_record(first.digest()).state_ref == first


def test_empty_payload_hook_receives_an_empty_data_root_and_publishes_exact_state(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    EmptyState.calls = []

    state = EmptyState(repo=repo).save(repo=repo)

    path = next(iter(state.states))
    directory = Path(store.open_local_state(state.object.definition.graph_hash(), state.states[path]))
    assert EmptyState.calls == [True]
    assert (directory / "data").is_dir()
    assert not any((directory / "data").iterdir())


def test_contended_node_fails_before_hooks_and_state_ref_publication(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = EmptyState(repo=repo)
    EmptyState.calls = []
    obj._save_load_reservation.acquire()
    try:
        with pytest.raises(Exception, match="already reserved"):
            obj.save(repo=repo)
    finally:
        obj._save_load_reservation.release()

    assert EmptyState.calls == []
    assert obj._last_state_hash is None
    assert not (tmp_path / "store" / "state-refs").exists()


@pytest.mark.parametrize("failure", ["hook", "install"])
def test_failed_local_state_publication_cleans_staging_before_state_ref(tmp_path, monkeypatch, failure):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = FailingState(repo=repo) if failure == "hook" else GoodState(repo=repo)
    if failure == "install":
        monkeypatch.setattr(store, "install_local_state", lambda source, manifest: (_ for _ in ()).throw(OSError("install failed")))

    with pytest.raises(Exception, match="local state publication"):
        obj.save(repo=repo)

    staging = Path(store.base_dir, ".staging")
    assert not staging.exists() or not any(staging.iterdir())
    assert not (Path(store.base_dir) / "state-refs").exists()
    assert obj._last_state_hash is None
