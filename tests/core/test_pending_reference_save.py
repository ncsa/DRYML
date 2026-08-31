from pathlib import Path

import pytest

from dryml.core import Definition, Repo, Serializable
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore


class PendingValue(Serializable):
    captures = 0

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).captures += 1
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")


class PendingParent(Serializable):
    captures = 0

    def __init__(self, child):
        self.child = child

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).captures += 1
        Path(dest_dir, "parent").write_text("parent", encoding="ascii")


class FailingPendingParent(PendingParent):
    def __init__(self, child):
        raise RuntimeError("parent construction failed")


class CountingPendingParent(PendingParent):
    constructions = 0

    def __init__(self, child):
        type(self).constructions += 1
        super().__init__(child)


def test_pending_declaration_save_completes_its_claim_and_captures_once(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = repo.declare_object(PendingValue(1).definition)
    PendingValue.captures = 0

    obj = repo.build_object_ref(reference)
    state = repo.save_object(obj, deep_capture=True)

    assert PendingValue.captures == 1
    assert state.object == reference
    assert store.read_claim_record(reference.digest()).status == "completed"


def test_nested_pending_declaration_completes_before_parent_and_is_adopted_once(tmp_path):
    child_store = DirStore(tmp_path / "child")
    parent_store = DirStore(tmp_path / "parent")
    repo = Repo([child_store, parent_store])
    child_reference = repo.declare_object(PendingValue(1).definition, store=child_store)
    parent_reference = repo.declare_object(
        Definition(PendingParent, child_reference).concretize(repo=repo),
        store=parent_store,
    )
    PendingValue.captures = 0
    PendingParent.captures = 0

    parent = repo.build_object_ref(parent_reference, store=parent_store)
    state = repo.save_object(parent, store=parent_store, deep_capture=True)

    assert child_store.read_claim_record(child_reference.digest()).status == "completed"
    assert parent_store.read_claim_record(parent_reference.digest()).status == "completed"
    assert PendingValue.captures == 1
    assert PendingParent.captures == 1
    child_path = next(path for path, object_id in state.object.objects.items() if object_id == child_reference.object_id)
    assert parent_store.validate_local_state(
        state.object.at(child_path).definition, state.states[child_path]
    )


def test_nested_constructor_failure_releases_only_acquired_claims_in_reverse_order(tmp_path, monkeypatch):
    child_store = DirStore(tmp_path / "child")
    parent_store = DirStore(tmp_path / "parent")
    repo = Repo([child_store, parent_store])
    child = repo.declare_object(PendingValue(1).definition, store=child_store)
    parent = repo.declare_object(
        Definition(FailingPendingParent, child).concretize(repo=repo), store=parent_store
    )
    released = []
    original = repo._abandon_claim

    def record_release(lease):
        released.append(lease.object_ref)
        return original(lease)

    monkeypatch.setattr(repo, "_abandon_claim", record_release)

    with pytest.raises(RepoLoadError, match="parent construction failed"):
        repo.build_object_ref(parent, store=parent_store)

    assert released == [parent, child]
    assert child_store.read_claim_record(child.digest()).status == "available"
    assert parent_store.read_claim_record(parent.digest()).status == "available"


def test_federated_pending_adoption_reports_child_declaration_store(tmp_path):
    child_store = DirStore(tmp_path / "child")
    parent_store = DirStore(tmp_path / "parent")
    repo = Repo([child_store, parent_store])
    child = repo.declare_object(PendingValue(1).definition, store=child_store)
    parent = repo.declare_object(
        Definition(PendingParent, child).concretize(repo=repo), store=parent_store
    )

    live = repo.build_object_ref(parent, store=parent_store)
    state, report = repo.save_object(
        live, store=parent_store, deep_capture=True, federated=True, report_stores=True
    )

    child_path = next(path for path, object_id in state.object.objects.items() if object_id == child.object_id)
    assert report.state_stores[child_path] is child_store
    assert child_store in report.required_stores
    assert parent_store in report.required_stores


def test_active_nested_claim_rejects_parent_before_any_constructor_runs(tmp_path):
    child_store = DirStore(tmp_path / "child")
    parent_store = DirStore(tmp_path / "parent")
    repo = Repo([child_store, parent_store])
    child = repo.declare_object(PendingValue(1).definition, store=child_store)
    parent = repo.declare_object(
        Definition(CountingPendingParent, child).concretize(repo=repo), store=parent_store
    )
    CountingPendingParent.constructions = 0
    lease = repo._acquire_claim(child, child_store)

    with pytest.raises(RepoLoadError, match="active first-construction claim"):
        repo.build_object_ref(parent, store=parent_store)

    assert CountingPendingParent.constructions == 0
    assert parent_store.read_claim_record(parent.digest()).status == "available"
    assert repo._abandon_claim(lease)
