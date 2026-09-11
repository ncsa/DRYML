from pathlib import Path

import pytest

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.object import Pickleable
from dryml.core.repo import RepoSaveError
from dryml.core.utils.general import pickle_load
from dryml.core.store.dir import DirStore


class ReceiptValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class ReceiptPickleable(Pickleable):
    def __init__(self, value):
        self.value = value


class ReceiptRoot(Object):
    def __init__(self, children):
        self.children = children


class ReceiptPendingParent(Serializable):
    def __init__(self, child):
        self.child = child

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "parent").write_text("parent", encoding="ascii")


def test_receipt_starts_empty_is_read_only_and_is_excluded_from_pickle_payload(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    obj = ReceiptPickleable("first", repo=repo)

    assert obj.last_state_ref is None
    with pytest.raises(AttributeError):
        obj.last_state_ref = None

    state = obj.save(repo=repo)
    payload = tmp_path / "payload"
    payload.mkdir()
    obj.save_state_to_dir(payload, codec="pkl")

    assert obj.last_state_ref == state
    assert "_last_state_ref" not in pickle_load(payload / "heavy.pkl")


def test_successful_save_keeps_the_last_complete_receipt_across_unsaved_mutation(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    obj = ReceiptValue(1, repo=repo)

    state = obj.save(repo=repo)
    obj.value = 2

    assert obj.last_state_ref == state


@pytest.mark.parametrize("reuse_live", ["matching", "greedy", "never"])
def test_exact_load_attaches_only_its_requested_root_receipt(tmp_path, reuse_live):
    source = Repo(DirStore(tmp_path / "store"))
    root = ReceiptRoot(ReceiptValue(7, repo=source), repo=source)
    state = source.save_object(root)
    root._last_state_ref = None

    loaded = source.load_state_ref(state, reuse_live=reuse_live)

    assert loaded.last_state_ref == state
    assert loaded.children.last_state_ref is None


def test_receipt_preserves_equal_child_identity_and_shared_alias_topology(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = ReceiptValue(3, repo=repo)
    second = ReceiptValue(3, repo=repo)
    root = ReceiptRoot([first, second, first], repo=repo)

    state = root.save(repo=repo)
    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert root.last_state_ref == state
    assert first.last_state_ref is None
    assert second.last_state_ref is None
    assert first.object_id != second.object_id
    assert loaded.children[0] is loaded.children[2]
    assert loaded.children[0] is not loaded.children[1]
    assert loaded.children[0].object_id != loaded.children[1].object_id
    assert loaded.last_state_ref == state
    assert loaded.children[0].last_state_ref is None
    assert loaded.children[1].last_state_ref is None


def test_recursive_child_receipt_survives_an_enclosing_root_failure(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child_ref = repo.declare_object(ReceiptValue(5).definition)
    parent_ref = repo.declare_object(
        Definition(ReceiptPendingParent, child_ref).concretize(repo=repo)
    )
    parent = repo.build_object_ref(parent_ref)
    installed = []
    original = store.write_state_ref_record

    def fail_parent_record(record):
        installed.append(record.state_ref)
        if len(installed) == 2:
            raise OSError("parent record install failed")
        return original(record)

    monkeypatch.setattr(store, "write_state_ref_record", fail_parent_record)
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(parent, deep_capture=True)

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "parent record install failed"
    assert parent.last_state_ref is None
    assert parent.child.last_state_ref == installed[0]
