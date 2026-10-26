"""Focused authority-selection proofs for the shared core signature interpreter."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import ConcreteDefinition, ConcreteNode, Definition, Node, Object, ObjectId, Repo, SavedNode, Serializable
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.repo import RepoLoadError
from dryml.core.quoted import QuotedDef, SelectorSpec
from dryml.core.selector import Selector
from dryml.core.signatures import (
    AutoRef,
    Mat,
    Ref,
    ReferenceSelection,
    SignatureError,
    compile_signature,
    signature_context,
)
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError


class SignatureCounter(Serializable):
    """Minimal stateful fixture whose state receipt changes with ``value``."""

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class SignatureStateless(Object):
    """Minimal stateless Object fixture for automatic authority selection."""

    def __init__(self, value=0):
        self.value = value


class SignatureWrapper(Object):
    """Stateless root whose materializing child controls automatic selection."""

    def __init__(self, child):
        self.child = child


class SignatureObjectParameter(SignatureCounter):
    """Stateful fixture whose ordinary constructor data is named ``object``."""

    def __init__(self, object=None):
        self.value = 0


def _plan(annotation):
    """Compile one single-slot signature with the supplied role annotation."""

    def target(value):
        return value

    target.__annotations__["value"] = annotation
    return compile_signature(target)


def _pair_plan(annotation):
    """Compile two ordinary slots with the same supplied annotation."""

    def target(first, second):
        return first, second

    target.__annotations__.update(first=annotation, second=annotation)
    return compile_signature(target)


def _stateful_reference(repo: Repo):
    """Build a bound Node and save it; selection itself performs neither action."""

    reference = Definition(SignatureCounter, 1).with_identity(ObjectId()).concretize()
    live = reference.build(repo=repo)
    state = repo.save_object(live)
    return reference, live, state


def _replicate_snapshot(source: DirStore, target: DirStore, state) -> None:
    """Copy selected full keys through an ordinary current-Node save."""
    repo = Repo((source, target))
    live = repo.load(state)
    replica = repo.save(live, store=target, states=state.state_spec(), source_store=source)
    assert replica.graph_equal(state)


def test_automatic_and_exact_live_projections_do_not_query_snapshots(tmp_path):
    """Automatic Ref uses local graph/receipt facts while exact live views stay local."""

    repo = Repo(DirStore(tmp_path / "store"))
    stateless = SignatureStateless()
    stateful = SignatureCounter(1)

    assert _plan(Ref[AutoRef]).prepare_args((stateless,), {}).authority["value"] == stateless.definition
    assert _plan(Ref[Node]).prepare_args((stateful,), {}).authority["value"] == stateful.definition
    assert _plan(Ref[ConcreteNode]).prepare_args((stateful,), {}).authority["value"] == stateful.definition
    with pytest.raises(SignatureError, match="unavailable"):
        _plan(Ref[SavedNode]).prepare_args((stateful,), {})

    reference, live, state = _stateful_reference(repo)
    live.value = 99
    assert _plan(Ref[Node]).prepare_args((live,), {}).authority["value"] == live.definition
    assert _plan(Ref[SavedNode]).prepare_args((state,), {}).authority["value"] is state
    assert repo.describe(live).object_view().graph_equal(reference.object_view())


def test_automatic_selection_counts_stateful_descendants_but_not_ref_data(tmp_path):
    """Automatic selection follows materializing topology without loading payloads."""

    repo = Repo(DirStore(tmp_path / "store"))
    stateful = SignatureCounter(2, repo=repo)
    materializing = SignatureWrapper(stateful, repo=repo)
    reference_only = SignatureWrapper(
        DefLink.finalized(EdgeKind.REF, stateful.definition), repo=repo
    )

    described = repo.describe(materializing)
    assert _plan(Ref[Node]).prepare_args((described,), {}).authority["value"] is described
    with pytest.raises(SignatureError, match="unavailable"):
        _plan(Ref[SavedNode]).prepare_args((described,), {}, repo=repo)
    assert _plan(Ref[AutoRef]).prepare_args((reference_only,), {}).authority["value"] == reference_only.definition


@pytest.mark.parametrize("ephemeral_root", (False, True))
def test_automatic_saved_live_receipt_is_metadata_only(tmp_path, monkeypatch, ephemeral_root):
    """Saved live roots and owned descendants select their receipt without IO/hooks."""
    from dryml.core import function, require_saved

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    child = SignatureCounter(3)
    live = SignatureWrapper(child) if ephemeral_root else child
    saved = repo.save(live)
    child.value = 99

    def forbidden(*args, **kwargs):
        pytest.fail("automatic Ref performed IO, capture, or restore")

    for name in ("save", "save_object", "load", "describe"):
        monkeypatch.setattr(repo, name, forbidden)
    monkeypatch.setattr(repo.default_store, "authority_cut", forbidden)
    monkeypatch.setattr(repo.default_store, "read_states", forbidden)
    monkeypatch.setattr(SignatureCounter, "save_state_to_dir_imp", forbidden)
    monkeypatch.setattr(SignatureCounter, "restore_state_from_dir_imp", forbidden)

    @function
    def select(value: Ref[AutoRef]) -> Ref[AutoRef]:
        return value

    with signature_context(repo=repo):
        assert select(live) is saved
    boundary = _plan(Ref[AutoRef]).prepare_args((live,), {}, repo=repo)
    assert boundary.authority["value"] is saved
    assert boundary.deliver_args()[0][0] is saved
    assert require_saved(saved) is saved
    assert saved.object_id is None if ephemeral_root else saved.object_id == child.object_id
    with pytest.raises(SignatureError, match="unavailable"):
        _plan(Ref[SavedNode]).prepare_args((live,), {}, repo=repo)


def test_automatic_receipt_checks_projection_not_constructor_object(tmp_path):
    """An ordinary ``object`` parameter cannot impersonate a receipt's authority."""

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    live = SignatureObjectParameter()
    saved = repo.save(live)
    assert _plan(Ref[AutoRef]).prepare_args((live,), {}).authority["value"] is saved

    identity = repo.describe(live).object_view()
    unrelated = Definition(SignatureObjectParameter, identity).concretize().with_realization(
        identities={"$": live.object_id}, states={"$": saved.state},
    )
    assert unrelated.object is identity
    live._last_state_ref = unrelated
    selected = _plan(Ref[AutoRef]).prepare_args((live,), {}).deliver_args()[0][0]
    assert isinstance(selected, ConcreteNode)
    assert selected.object_id == live.object_id
    assert selected.state is None
    assert not selected.object_projection_equal(unrelated)
    with pytest.raises(SignatureError, match="unavailable"):
        _plan(Ref[SavedNode]).prepare_args((live,), {})

    live._last_state_ref = saved.without_state()
    incomplete = _plan(Ref[AutoRef]).prepare_args((live,), {}).authority["value"]
    assert incomplete.state is None and incomplete.object_id == live.object_id


def test_state_selection_deduplicates_replicas_and_rejects_ambiguity(tmp_path):
    """Explicit query dedupes replicas; a Node signature never invents a checkpoint."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(first)
    reference, live, state = _stateful_reference(repo)
    _replicate_snapshot(first, second, state)
    replicas = Repo((first, second))

    selected = replicas.query().saved().sel(reference).one()
    assert _plan(Ref[SavedNode]).prepare_args((selected,), {}, repo=replicas).authority["value"].graph_equal(state)
    with signature_context(repo=replicas):
        with pytest.raises(SignatureError, match="unavailable"):
            _plan(Ref[SavedNode]).prepare_args((reference,), {})
    live.value = 2
    second_state = repo.save_object(live, deep_capture=True)
    _replicate_snapshot(first, second, second_state)
    from dryml.core.query.model import QueryCardinalityError
    with pytest.raises(QueryCardinalityError):
        replicas.query().saved().sel(reference).one()
    for annotation in (Ref[ConcreteNode | SavedNode], Ref[SavedNode | ConcreteNode]):
        assert _plan(annotation).prepare_args((state,), {}, repo=replicas).authority["value"] is state


@pytest.mark.parametrize("status", ("available", "claimed", "completed"))
def test_declaration_claim_statuses_remain_reference_eligible(tmp_path, status, monkeypatch):
    """Recipe-only, live-reserved and saved Nodes remain inert reference metadata."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = Definition(SignatureCounter, 1).with_identity(ObjectId()).concretize()
    record = store.publish_node(reference)
    from contextlib import ExitStack
    with ExitStack() as lifetime:
        if status in {"claimed", "completed"}:
            live = reference.build(repo=repo)
            if status == "claimed":
                lifetime.enter_context(repo.reserve_state_graph(live))
            else:
                reference = repo.save(live)
        before = store.authority_cut()
        constructions = repo._num_constructions
        monkeypatch.setattr(store, "read_states", lambda *_a, **_kw: pytest.fail("Ref read payloads"))
        result = _plan(Ref[ConcreteNode]).prepare_args((reference,), {}, repo=repo)
        assert result.authority["value"] is reference
        assert store.authority_cut() == before
        assert repo._num_constructions == constructions
    assert store.read_node(record.digest).object_id == reference.object_id


def test_retained_live_claim_remains_reference_eligible_without_acquisition(tmp_path):
    """A real reservation remains owned while Ref inspects supplied Node metadata."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = Definition(SignatureCounter, 1).with_identity(ObjectId()).concretize()
    live = reference.build(repo=repo)
    with repo.reserve_state_graph(live) as reservation:
        selected = _plan(Ref[ConcreteNode]).prepare_args((reference,), {}, repo=repo)
        assert selected.authority["value"] is reference
        assert reservation.active and live.last_state_ref is None


def test_explicit_selection_and_definition_strengthening_guards(tmp_path):
    """Slot selections disambiguate only declared topology and Definition does not add defaults."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    cdef = SignatureCounter(1).definition
    first = cdef.with_identity(ObjectId())
    second = cdef.with_identity(ObjectId())
    store.publish_node(first)
    store.publish_node(second)
    plan = _plan(Ref[ConcreteNode])
    assert plan.prepare_args((cdef,), {}, repo=repo).authority["value"] is cdef
    selected = plan.prepare_args(
        (second,), {}, repo=repo, selections={"value": ReferenceSelection(second, store)}
    )
    assert selected.authority["value"] == second
    unrelated = Definition(SignatureCounter, 2).concretize()
    with pytest.raises(SignatureError, match="unavailable"):
        plan.prepare_args((cdef,), {}, repo=repo, selections={"value": unrelated})

    assert _plan(Ref[ConcreteDefinition]).prepare_args((Definition(SignatureCounter, 1),), {}).authority["value"].graph_equal(cdef)
    with pytest.raises(SignatureError, match="unavailable"):
        _plan(Ref[ConcreteDefinition]).prepare_args((Definition(SignatureCounter),), {})


def test_materializing_selection_claims_only_the_pinned_replica(tmp_path):
    """Boundary delivery retains the selected declaration Store through admission."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    source = Repo(first)
    live_source = SignatureCounter(7, repo=source)
    reference = source.save(live_source)
    _replicate_snapshot(first, second, reference)
    repo = Repo((first, second))

    boundary = _plan(Mat[SignatureCounter]).prepare_args(
        (reference,), {}, repo=repo,
        selections={"value": ReferenceSelection(reference, second)},
    )
    live = boundary.deliver_args()[0][0]

    assert live.object_id == reference.object_id and live.value == 7
    assert live.definition.graph_equal(live_source.definition)
    assert live.last_state_ref.graph_equal(reference)


def test_materializing_selection_revalidates_stale_and_disconnected_stores(tmp_path, monkeypatch):
    """A prepared Store pin cannot be substituted after its authority changes."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(first)
    reference = repo.save(SignatureCounter(8, repo=repo))
    _replicate_snapshot(first, second, reference)
    connected = Repo((first, second))
    stale = _plan(Mat[SignatureCounter]).prepare_args(
        (reference,), {}, repo=connected,
        selections={"value": ReferenceSelection(reference, second)},
    )
    @contextmanager
    def unavailable(*args, **kwargs):
        raise StoreAuthorityError("selected source unavailable")
        yield
    monkeypatch.setattr(second, "read_states", unavailable)
    with pytest.raises(RepoLoadError, match="invalid source") as raised:
        stale.deliver_args()
    assert isinstance(raised.value.__cause__, StoreAuthorityError)
    assert connected._num_constructions == 0

    disconnected = DirStore(tmp_path / "disconnected")
    boundary = _plan(Mat[SignatureCounter]).prepare_args(
        (reference,), {}, repo=repo,
        selections={"value": ReferenceSelection(reference, disconnected)},
    )
    with pytest.raises((RepoLoadError, ValueError), match="connected"):
        boundary.deliver_args()
    assert connected._num_constructions == 0


def test_conflicting_replica_pins_fail_before_claims_but_same_domain_handles_agree(tmp_path):
    """Distinct Store domains conflict while duplicate handles name one authority."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    source = Repo(first)
    reference = source.save(SignatureCounter(9, repo=source))
    _replicate_snapshot(first, second, reference)
    repo = Repo((first, second))
    conflicting = _pair_plan(Mat[SignatureCounter]).prepare_args(
        (reference, reference), {}, repo=repo,
        selections={
            "first": ReferenceSelection(reference, first),
            "second": ReferenceSelection(reference, second),
        },
    )

    with pytest.raises(SignatureError, match="conflicting declaration Store"):
        conflicting.deliver_args()
    assert repo._num_constructions == 0

    shared_first = DirStore(tmp_path / "shared")
    shared_second = DirStore(tmp_path / "shared")
    shared_repo = Repo((shared_first, shared_second))
    shared_ref = Repo(shared_first).save(SignatureCounter(10))
    agreeing = _pair_plan(Mat[SignatureCounter]).prepare_args(
        (shared_ref, shared_ref), {}, repo=shared_repo,
        selections={
            "first": ReferenceSelection(shared_ref, shared_first),
            "second": ReferenceSelection(shared_ref, shared_second),
        },
    )
    left, right = agreeing.deliver_args()[0]
    assert left is right


def test_source_guards_and_quotation_data_are_preserved(tmp_path):
    """Live receipt and quotation restrictions survive composed authority conversion."""

    repo = Repo(DirStore(tmp_path / "store"))
    reference, live, state = _stateful_reference(repo)
    no_receipt = SignatureCounter(2, repo=repo)
    with pytest.raises(SignatureError, match="unavailable"):
        _plan(Ref[SavedNode]).prepare_args((no_receipt,), {}, repo=repo)
    assert _plan(Ref[SavedNode]).prepare_args((state,), {}, repo=repo).authority["value"] is state

    quoted = QuotedDef(Definition(SignatureCounter, 1))
    assert _plan(Ref[Node]).prepare_args((quoted,), {}).authority["value"] == quoted.value
    assert _plan(Ref[Node]).prepare_args((quoted,), {}).authority["value"].object_id is None
    selector_spec = SelectorSpec(Selector(Definition(SignatureCounter)))
    assert _plan(Ref[Selector]).prepare_args((selector_spec,), {}).authority["value"] == selector_spec.selector
    with pytest.raises(SignatureError, match="quoted input"):
        _plan(Mat[SignatureCounter]).prepare_args((quoted,), {})
    assert _plan(Ref[Node | ConcreteNode | SavedNode]).prepare_args(
        (state,), {}, repo=repo
    ).authority["value"] is state
    assert repo.describe(live).object_view().graph_equal(reference.object_view())


def test_state_only_and_store_failures_are_not_treated_as_absence(tmp_path, monkeypatch):
    """State-only records serve explicit refs, while corrupt/missing reads remain failures."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference, _, state = _stateful_reference(repo)
    state_only = DirStore(tmp_path / "state-only")
    state_only.publish_node(state, kind="saved")
    detached = Repo(state_only)
    assert _plan(Ref[SavedNode]).prepare_args((state,), {}, repo=detached).authority["value"] is state
    with pytest.raises(RepoLoadError, match="unavailable"):
        _plan(Mat[SignatureCounter]).prepare_args((state,), {}, repo=detached).deliver_args()
    @contextmanager
    def offline(*args, **kwargs):
        raise OSError("offline")
        yield
    monkeypatch.setattr(store, "read_states", offline)
    with pytest.raises(RepoLoadError, match="invalid source") as raised:
        _plan(Mat[SignatureCounter]).prepare_args((state,), {}, repo=repo).deliver_args()
    assert isinstance(raised.value.__cause__, OSError)


def test_evidence_fences_duplicate_handles_once_and_reject_unsupported(tmp_path, monkeypatch):
    """Metadata-only Ref does not acquire source fences, even with duplicate handles."""

    class CountingStore(DirStore):
        fences = 0

        @contextmanager
        def authority_read_fence(self):
            type(self).fences += 1
            with super().authority_read_fence():
                yield

    store = CountingStore(tmp_path / "store")
    repo = Repo((store, store))
    CountingStore.fences = 0
    reference = Definition(SignatureCounter, 1).with_identity(ObjectId()).concretize()
    selected = _plan(Ref[ConcreteNode]).prepare_args((reference,), {}, repo=repo)
    assert selected.authority["value"] is reference
    assert CountingStore.fences == 0

    unsupported = Repo(CountingStore(tmp_path / "unsupported"))
    monkeypatch.setattr(unsupported.default_store, "authority_fence_key",
                        lambda: pytest.fail("Ref must not acquire an availability fence"))
    assert _plan(Ref[ConcreteNode]).prepare_args((reference,), {}, repo=unsupported).authority["value"] is reference


def test_immutable_snapshot_store_uses_its_read_fence_not_a_writer_fence(tmp_path):
    """Read-only snapshot authority can select/restore without mutable fencing."""

    source = DirStore(tmp_path / "store", query_index="none")
    writable = Repo(source)
    reference, _, state = _stateful_reference(writable)

    class ImmutableSnapshotStore(DirStore):
        def authority_fence_key(self):
            return f"immutable:{self.base_dir}"

        @contextmanager
        def authority_read_fence(self):
            yield

        def writer_lock(self):
            raise AssertionError("immutable snapshot selection must not take a writer fence")

    readonly = Repo(ImmutableSnapshotStore.open_existing(tmp_path / "store", query_index="none"))

    assert _plan(Ref[SavedNode]).prepare_args((state,), {}, repo=readonly).authority["value"] is state
    assert _plan(Mat[SignatureCounter]).prepare_args((state,), {}, repo=readonly).deliver_args()[0][0].value == 1
