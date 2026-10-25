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


class CountingPendingParent(PendingParent):
    constructions = 0

    def __init__(self, child):
        type(self).constructions += 1
        super().__init__(child)


class FailingPendingValue(PendingValue):
    def save_state_to_dir_imp(self, dest_dir, *, codec):
        raise RuntimeError("child save failed")


class FailSecondSnapshotStore(DirStore):
    """Inject one replica failure after another Store has completed a claim."""

    fail = False

    def publish_snapshot(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("second replica failed")
        return super().publish_snapshot(*args, **kwargs)


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


def test_nested_pending_declaration_completes_before_parent_and_is_adopted_once(tmp_path):
    child_store = DirStore(tmp_path / "child")
    parent_store = DirStore(tmp_path / "parent")
    repo = Repo(
        [child_store, parent_store],
        save_routing=SaveRouting(
            ((Selector(PendingValue), child_store), (Selector(PendingParent), parent_store)),
        ),
    )
    child_reference = repo.declare_object(PendingValue(1).definition, store=child_store)
    parent_reference = repo.declare_object(
        Definition(PendingParent, child_reference).concretize(repo=repo),
        store=parent_store,
    )
    PendingValue.captures = 0
    PendingParent.captures = 0

    parent = repo.build_object_ref(parent_reference, store=parent_store)
    state = repo.save_object(parent, deep_capture=True)

    assert child_store.read_claim_record(child_reference.digest()).status == "completed"
    assert parent_store.read_claim_record(parent_reference.digest()).status == "completed"
    assert PendingValue.captures == 1
    assert PendingParent.captures == 1
    child_path = next(path for path, object_id in state.object.objects.items() if object_id == child_reference.object_id)
    child_state = state.at(child_path)
    assert child_store.validate_local_state(child_state, ()).state_hash == state.states[child_path]


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


def test_derived_index_failure_clears_completed_live_claim(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = repo.declare_object(PendingValue(1).definition)
    live = repo.build_object_ref(reference)

    monkeypatch.setattr(
        repo._query_index,
        "register_saved_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("index registration failed")
        ),
    )

    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.save_object(live, deep_capture=True)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "index registration failed" in str(raised.value.__cause__)
    assert raised.value.report is not None

    assert store.read_claim_record(reference.digest()).status == "completed"
    assert live._claim_lease is None
    assert live._claim_leases == ()
    assert live._pending_claim_dependencies == ()


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


def test_completed_claim_survives_a_later_routed_replica_failure(tmp_path):
    """A later replica error cannot abandon an already completed claim generation."""
    declaration_store = DirStore(tmp_path / "declaration")
    failing_store = FailSecondSnapshotStore(tmp_path / "failing")
    repo = Repo(
        [declaration_store, failing_store],
        save_routing=SaveRouting(
            ((Selector(PendingValue), declaration_store), (Selector(PendingValue), failing_store)),
            match_mode="all",
        ),
    )
    reference = repo.declare_object(PendingValue(1).definition, store=declaration_store)
    live = repo.build_object_ref(reference, store=declaration_store)
    failing_store.fail = True

    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.save_object(live, deep_capture=True)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "second replica failed" in str(raised.value.__cause__)
    assert raised.value.report is not None

    claim = declaration_store.read_claim_record(reference.digest())
    assert claim.status == "completed"
    assert declaration_store.read_state_ref_record(claim.state_ref_digest) is not None
    assert live._claim_lease is None


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


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("after", [False, True], ids=["before", "after"])
def test_claim_control_flow_preserves_generation_evidence(
        tmp_path, monkeypatch, error_type, after):
    """A failure after claim replacement keeps the generation's completed authority."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    reference = repo.declare_object(PendingValue(1).definition)
    live = repo.build_object_ref(reference)
    original = repo._mark_initial_state_ref_complete

    def complete_then_fail(state_ref, destination, lease):
        if not after:
            raise error_type("claim interruption")
        original(state_ref, destination, lease)
        raise error_type("claim interruption")

    monkeypatch.setattr(repo, "_mark_initial_state_ref_complete", complete_then_fail)
    with pytest.raises(error_type, match="claim interruption") as raised:
        repo.save_object(live, deep_capture=True)

    report = raised.value.report
    claim_publication = next(item for item in report.publications if item.phase == "claim")
    assert claim_publication.status == ("completed" if after else "failed")
    assert store.read_claim_record(reference.digest()).status == (
        "completed" if after else "available"
    )
    assert report.snapshots == ()
    if after:
        assert store.read_state_ref_record(claim_publication.state_ref.digest()) is not None
        assert live._claim_lease is None
        assert live._claim_leases == ()
    assert live.last_state_ref is None


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
