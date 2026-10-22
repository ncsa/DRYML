from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Repo, Serializable
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.node_prepare import NodePreparation


def saved_digest(node):
    return NodePreparation((node,)).digest(node, domain="dryml-saved-root/v1")


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

    with pytest.raises(Exception, match="snapshot-local state preparation"):
        obj.save(repo=repo)

    assert obj._last_state_hash is None
    assert obj.last_state_ref is None
    assert not tuple(Path(store.base_dir, "graphs").rglob("*.record"))


def test_alias_replacement_failure_preserves_completed_state_ref(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = GoodState(repo=repo)
    first = obj.save(repo=repo, alias="latest")

    original = store._atomic_write
    def fail_alias(path, payload):
        if str(path).endswith("/names/latest.record"):
            raise OSError("alias replacement failed")
        return original(path, payload)

    monkeypatch.setattr(store, "_atomic_write", fail_alias)
    obj.value = 2
    with pytest.raises(RepoSaveError, match="publication") as raised:
        obj.save(repo=repo, alias="latest", deep_capture=True)
    assert isinstance(raised.value.__cause__, OSError)
    assert "alias replacement" in str(raised.value.__cause__)
    assert raised.value.report is not None

    assert store.read_node(saved_digest(first), kind="saved").graph_equal(first)
    assert obj.last_state_ref != first
    assert store.read_node(saved_digest(obj.last_state_ref), kind="saved").graph_equal(obj.last_state_ref)


def test_snapshot_install_failure_preserves_the_prior_receipt(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = GoodState(repo=repo)
    first = obj.save(repo=repo)
    obj.value = 2

    original = store._atomic_write
    def fail_snapshot(path, payload):
        if "/graphs/" in str(path):
            raise OSError("snapshot install failed")
        return original(path, payload)
    monkeypatch.setattr(store, "_atomic_write", fail_snapshot)
    with pytest.raises(RepoSaveError, match="publication") as raised:
        obj.save(repo=repo, deep_capture=True)
    assert isinstance(raised.value.__cause__, OSError)
    assert "snapshot install failed" in str(raised.value.__cause__)
    assert raised.value.report is not None

    assert obj.last_state_ref == first


def test_derived_index_failure_propagates_after_installing_the_receipt(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = GoodState(repo=repo)

    index = store.open_query_index()
    index.maintain_store()
    original = index.maintain_store
    def fail_index(*args, **kwargs):
        if store.authority_cut().pending:
            raise RuntimeError("index registration failed")
        return original(*args, **kwargs)
    monkeypatch.setattr(index, "maintain_store", fail_index)
    with pytest.raises(RepoSaveError, match="publication") as raised:
        obj.save(repo=repo)
    assert isinstance(raised.value.__cause__, OSError)
    assert "index registration failed" in str(raised.value.__cause__)
    assert raised.value.report is not None

    assert store.read_node(saved_digest(obj.last_state_ref), kind="saved").graph_equal(obj.last_state_ref)


def test_unconfirmed_readback_does_not_return_complete_save(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    obj = GoodState(repo=repo)
    monkeypatch.setattr(store, "read_node", lambda *args, **kwargs: None)
    with pytest.raises(RepoSaveError, match="read-back") as caught:
        repo.save_object(obj)
    assert caught.value.report is not None
    assert obj.last_state_ref is None


def test_empty_payload_hook_receives_an_empty_data_root_and_publishes_exact_state(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    EmptyState.calls = []

    state = EmptyState(repo=repo).save(repo=repo)

    with store.read_state(state.state.key) as source:
        directory = Path(source.data_directory)
        assert EmptyState.calls == [True]
        assert directory.is_dir()
        assert not any(directory.iterdir())


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
    assert not tuple(Path(store.base_dir, "graphs").rglob("*.record"))


@pytest.mark.parametrize("failure", ["hook", "snapshot"])
def test_failed_local_state_publication_cleans_staging_before_state_ref(tmp_path, monkeypatch, failure):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = FailingState(repo=repo) if failure == "hook" else GoodState(repo=repo)
    if failure == "snapshot":
        original = store._atomic_write
        def fail_snapshot(path, payload):
            if "/graphs/" in str(path):
                raise OSError("snapshot install failed")
            return original(path, payload)
        monkeypatch.setattr(store, "_atomic_write", fail_snapshot)

    expected = "snapshot-local state preparation" if failure == "hook" else "Save publication failed"
    with pytest.raises(Exception, match=expected):
        obj.save(repo=repo)

    staging = Path(store.base_dir, ".staging")
    if failure == "hook":
        assert not staging.exists() or {p.name for p in staging.iterdir()} <= {".leases"}
        assert not store._staging_leases
    else:
        assert store.pending_mutations()
        assert obj.last_state_ref is None
    assert not tuple(Path(store.base_dir, "graphs").rglob("*.record"))
    assert (obj._last_state_hash is None) is (failure == "hook")
