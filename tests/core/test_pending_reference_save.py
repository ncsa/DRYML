from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, ObjectId, Repo, SaveRouting, Selector, Serializable
from dryml.core.repo import RepoLoadError, RepoSaveError
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


class FailingPendingValue(PendingValue):
    def save_state_to_dir_imp(self, dest_dir, *, codec):
        raise RuntimeError("child save failed")


def test_pending_declaration_save_completes_its_claim_and_captures_once(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = Definition(PendingValue, 1).with_identity(ObjectId()).concretize()
    PendingValue.captures = 0
    obj = repo.build(reference)
    with repo.reserve_state_graph(obj) as reservation:
        state = repo.save_object(obj, reservation=reservation)

    assert PendingValue.captures == 1
    assert state.object_id == reference.object_id
    assert state.creation_view().graph_equal(reference.creation_view())
    assert store.query().saved().one().graph_equal(state)
    assert not reservation.active


def test_nested_constructor_failure_releases_only_acquired_claims_in_reverse_order(tmp_path, monkeypatch):
    child_store = DirStore(tmp_path / "child")
    parent_store = DirStore(tmp_path / "parent")
    repo = Repo([child_store, parent_store])
    child = Definition(PendingValue, 1).with_identity(ObjectId())
    parent = Definition(FailingPendingParent, child).with_identity(ObjectId()).concretize()
    with pytest.raises(RepoLoadError, match="parent construction failed"):
        repo.build(parent)
    from dryml.core.state import reserve
    with reserve(parent, (), tuple(parent.identity_spec().values())) as reservation:
        assert reservation.active
    assert child_store.query().saved().count() == 0
    assert parent_store.query().saved().count() == 0
    assert repo.build(child.concretize(), cache="none").object_id == child.object_id


def test_explicit_store_rejects_excluded_pending_declaration_before_capture(tmp_path):
    child_store = DirStore(tmp_path / "child")
    parent_store = DirStore(tmp_path / "parent")
    repo = Repo([child_store, parent_store])
    child = Definition(PendingValue, 1).with_identity(ObjectId())
    parent = Definition(PendingParent, child).with_identity(ObjectId()).concretize()
    child_store.publish_node(child.concretize())
    before = child_store.authority_cut()
    live = repo.build(parent)
    PendingValue.captures = 0
    PendingParent.captures = 0

    state = repo.save_object(live, store=parent_store)
    assert PendingValue.captures == PendingParent.captures == 1
    assert child_store.authority_cut() == before
    assert child_store.query().saved().count() == 0
    assert parent_store.query().sel(state).saved().one().graph_equal(state)
    from dryml.core import DefinitionPath, Parameter
    child_state = state.at(DefinitionPath((Parameter("child"),)))
    assert parent_store.query().sel(child_state).saved().one().graph_equal(child_state)
    for local in state.state_spec().values():
        with parent_store.read_state(local.key) as read:
            assert Path(read.data_directory).is_dir()


def test_dependency_save_failure_abandons_enclosing_claims(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = Definition(FailingPendingValue, 1).with_identity(ObjectId())
    parent = Definition(PendingParent, child).with_identity(ObjectId()).concretize()
    live = repo.build(parent)
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(live, deep_capture=True)
    assert "child save failed" in str(raised.value.__cause__)
    assert store.query().saved().count() == 0
    with repo.reserve_state_graph(live) as reservation:
        assert reservation.active


def test_routing_rejects_an_excluded_pending_declaration_before_capture(tmp_path):
    """Recipe membership does not constrain later native state-save routing."""
    declaration_store = DirStore(tmp_path / "declaration")
    routed_store = DirStore(tmp_path / "routed")
    repo = Repo(
        [declaration_store, routed_store],
        save_routing=SaveRouting(((Selector(PendingValue), routed_store),)),
    )
    reference = Definition(PendingValue, 1).with_identity(ObjectId()).concretize()
    declaration_store.publish_node(reference)
    before = declaration_store.authority_cut()
    live = repo.build(reference)
    PendingValue.captures = 0

    state = repo.save_object(live)
    assert PendingValue.captures == 1
    assert declaration_store.authority_cut() == before
    assert declaration_store.query().saved().count() == 0
    assert routed_store.query().saved().one().graph_equal(state)


def test_routed_claim_generation_fences_snapshot_membership_and_completion(tmp_path, monkeypatch):
    """An inactive reservation cannot publish through a new owner's save window."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    live = PendingValue(1, repo=repo)
    with repo.reserve_state_graph(live) as stale:
        assert stale.active
    PendingValue.captures = 0
    with repo.reserve_state_graph(live) as current:
        with pytest.raises(RepoSaveError, match="reservation|active"):
            repo.save_object(live, reservation=stale)
        assert current.active
        assert PendingValue.captures == 0
        assert store.query().saved().count() == 0
        state = repo.save_object(live, reservation=current)
    assert PendingValue.captures == 1
    assert store.query().saved().one().graph_equal(state)


def test_route_order_remains_a_then_b_while_declaration_b_claim_completes_first(tmp_path, monkeypatch):
    """Native save batches preserve route order and both independently readable replicas."""

    first = DirStore(tmp_path / "first")
    declaration = DirStore(tmp_path / "declaration")
    repo = Repo(
        [first, declaration],
        save_routing=SaveRouting(
            ((Selector(PendingValue), first), (Selector(PendingValue), declaration)),
            match_mode="all",
        ),
    )
    reference = Definition(PendingValue, 1).with_identity(ObjectId()).concretize()
    declaration.publish_node(reference)
    live = repo.build(reference)
    observed = []
    original = first.publish_save_batch

    def observe_first(*args, **kwargs):
        observed.append(declaration.query().saved().count())
        return original(*args, **kwargs)

    monkeypatch.setattr(first, "publish_save_batch", observe_first)
    state, report = repo.save_object(live, deep_capture=True, report_stores=True)

    snapshots = [item for item in report.publications if item.phase == "snapshot"]
    assert [item.store for item in snapshots] == [first, declaration]
    assert observed == [0]
    assert first.query().saved().one().graph_equal(state)
    assert declaration.query().saved().one().graph_equal(state)
