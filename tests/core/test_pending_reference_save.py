from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, Repo, SaveRouting, Selector, Serializable
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import ClaimRecord


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


def test_explicit_store_rejects_excluded_pending_declaration_before_capture(tmp_path):
    child_store = DirStore(tmp_path / "child")
    parent_store = DirStore(tmp_path / "parent")
    repo = Repo([child_store, parent_store])
    child = repo.declare_object(PendingValue(1).definition, store=child_store)
    parent = repo.declare_object(
        Definition(PendingParent, child).concretize(repo=repo), store=parent_store
    )

    live = repo.build_object_ref(parent, store=parent_store)
    PendingValue.captures = 0
    PendingParent.captures = 0

    with pytest.raises(RepoSaveError, match="declaration Store"):
        repo.save_object(live, store=parent_store, deep_capture=True)

    assert PendingValue.captures == 0
    assert PendingParent.captures == 0
    assert child_store.read_claim_record(child.digest()).status == "available"
    assert parent_store.read_claim_record(parent.digest()).status == "available"


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
    child = repo.declare_object(FailingPendingValue(1).definition)
    parent = repo.declare_object(
        Definition(PendingParent, child).concretize(repo=repo)
    )
    live = repo.build_object_ref(parent)

    with pytest.raises(RepoSaveError, match="snapshot-local state preparation failed"):
        repo.save_object(live, deep_capture=True)

    assert store.read_claim_record(child.digest()).status == "available"
    assert store.read_claim_record(parent.digest()).status == "available"


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
    """Every pending declaration Store must be selected before user hooks run."""
    declaration_store = DirStore(tmp_path / "declaration")
    routed_store = DirStore(tmp_path / "routed")
    repo = Repo(
        [declaration_store, routed_store],
        save_routing=SaveRouting(((Selector(PendingValue), routed_store),)),
    )
    reference = repo.declare_object(PendingValue(1).definition, store=declaration_store)
    live = repo.build_object_ref(reference, store=declaration_store)
    PendingValue.captures = 0

    with pytest.raises(RepoSaveError, match="declaration Store"):
        repo.save_object(live, deep_capture=True)

    assert PendingValue.captures == 0
    assert declaration_store.read_claim_record(reference.digest()).status == "available"


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
    """A successor generation cannot be completed through a stale save window."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    reference = repo.declare_object(PendingValue(1).definition)
    live = repo.build_object_ref(reference)
    original_mark = repo._mark_initial_state_ref_complete

    def replace_generation(state_ref, destination, lease):
        destination.write_claim_record(ClaimRecord(
            lease.object_ref.digest(), lease.generation + 1, "claimed", "successor", 1000,
        ))
        original_mark(state_ref, destination, lease)

    monkeypatch.setattr(repo, "_mark_initial_state_ref_complete", replace_generation)

    with pytest.raises(RepoSaveError):
        repo.save_object(live, deep_capture=True)

    claim = store.read_claim_record(reference.digest())
    assert claim.generation == 2
    assert claim.status == "claimed"


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
    """A declaration replica fences its claim before earlier logical route work."""

    first = DirStore(tmp_path / "first")
    declaration = DirStore(tmp_path / "declaration")
    repo = Repo(
        [first, declaration],
        save_routing=SaveRouting(
            ((Selector(PendingValue), first), (Selector(PendingValue), declaration)),
            match_mode="all",
        ),
    )
    reference = repo.declare_object(PendingValue(1).definition, store=declaration)
    live = repo.build_object_ref(reference, store=declaration)
    observed = []
    original = first.publish_snapshot

    def observe_first(*args, **kwargs):
        claim = declaration.read_claim_record(reference.digest())
        observed.append(claim.status if claim is not None else None)
        return original(*args, **kwargs)

    monkeypatch.setattr(first, "publish_snapshot", observe_first)
    state, report = repo.save_object(live, deep_capture=True, report_stores=True)

    snapshots = [item for item in report.publications if item.phase == "snapshot"]
    assert [item.store for item in snapshots] == [first, declaration]
    assert observed == ["claimed"]
    assert declaration.read_claim_record(reference.digest()).status == "completed"
    assert first.read_state_ref_record(state.digest()).state_ref == state
    assert declaration.read_state_ref_record(state.digest()).state_ref == state
