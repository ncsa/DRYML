"""Aggregate Mat delivery proofs for the core signature boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import ConcreteDefinition, Definition, Object, Repo, Serializable
from dryml.core.freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from dryml.core.repo import RepoLoadError
from dryml.core.reference_values import ObjectRef, StateRef
from dryml.core.signatures import Mat, Ref, SignatureError, compile_signature
from dryml.core.store.dir import DirStore
from dryml.core.store.records import (
    ClaimRecord,
    DeclarationRecord,
    DefinitionRecord,
)
from dryml.core.utils.graph.path import GraphPath, Parameter


class BoundaryValue(Serializable):
    """Stateful fixture recording construction and restoration effects."""

    constructions = 0
    restores = 0

    def __init__(self, value=3):
        type(self).constructions += 1
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        type(self).restores += 1
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class BoundaryPair(Object):
    """Container fixture retaining constructor argument identity."""

    constructions = 0

    def __init__(self, left, right):
        type(self).constructions += 1
        self.left = left
        self.right = right


class BoundaryOwner(Serializable):
    """Durable parent fixture retaining one materializing ObjectRef child."""

    def __init__(self, child, label):
        self.child = child
        self.label = label

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "label").write_text(self.label, encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.label = Path(src_dir, "label").read_text(encoding="ascii")


class FalseyBoundaryValue(BoundaryValue):
    """Stateful fixture whose valid live instances have false truthiness."""

    def __bool__(self):
        return False


def _plan(*annotations):
    """Compile one positional Mat test target from supplied annotations."""

    def target(*values):
        return values

    target.__annotations__ = {
        name: annotation for name, annotation in zip(("values",), annotations)
    }
    return compile_signature(target)


def _single(annotation):
    """Compile a normal single-value boundary without variadic ambiguity."""

    def target(value):
        return value

    target.__annotations__["value"] = annotation
    return compile_signature(target)


def test_mat_definition_applies_defaults_and_cdef_is_structural():
    """Mat Definition completes defaults while an exact CDef remains its recipe."""

    repo = Repo()
    definition = Definition(BoundaryValue)
    cdef = Definition(BoundaryValue, 7).concretize(repo=repo)

    defaulted = _single(Mat[Definition]).prepare_args((definition,), {}, repo=repo)
    structural = _single(Mat[ConcreteDefinition]).prepare_args((cdef,), {}, repo=repo)

    assert defaulted.deliver_args()[0][0].value == 3
    assert structural.deliver_args()[0][0].definition is cdef


def test_mat_preserves_cross_root_and_container_sharing_under_never(tmp_path):
    """One selected StateRef stays one object across roots and container aliases."""

    source = Repo(DirStore(tmp_path / "store"))
    state = source.save_object(BoundaryValue(4, repo=source))
    loaded = Repo(DirStore(tmp_path / "store"))

    result = loaded.materialize_boundary((state, [state, state]), reuse_live="never")

    assert result[0] is result[1][0] is result[1][1]
    assert result[0].value == 4


def test_aggregate_keeps_independent_private_cdefs_distinct_and_direct_live():
    """Private node identity, not equality, controls structural reuse and live delivery."""

    repo = Repo()
    first = Definition(BoundaryValue, 6).concretize(repo=repo)
    second = Definition(BoundaryValue, 6).concretize(repo=repo)
    live = BoundaryValue(8, repo=repo)

    realized = repo.materialize_boundary((first, second), cache="none")
    delivered = _single(Mat[Object]).prepare_args((live,), {}, repo=repo).deliver_args()[0][0]

    assert realized[0] is not realized[1]
    assert delivered is live
    assert delivered.value == 8


def test_aggregate_concretization_preserves_repeated_roots_and_nested_aliases():
    """One Definition identity remains shared across the complete boundary."""

    repo = Repo()
    shared = Definition(BoundaryValue, 5)
    parent = Definition(BoundaryPair, shared, shared)
    independent = Definition(BoundaryValue, 5)

    root, loaded_parent, separate = repo.materialize_boundary(
        (shared, parent, independent), cache="none"
    )

    assert root is loaded_parent.left is loaded_parent.right
    assert separate is not root
    assert separate.definition == root.definition


def test_materialization_rebuilds_mutable_frozen_and_plain_frozenset_containers(tmp_path):
    """Recursive Mat delivery preserves container families and reference aliases."""

    source = Repo(DirStore(tmp_path / "store"))
    state = source.save_object(BoundaryValue(4, repo=source))
    roots = (
        [state],
        (state,),
        {"value": state},
        {state},
        frozenset({state}),
        FrozenList((state,)),
        FrozenTuple((state,)),
        FrozenSet((state,)),
        FrozenDict({"value": state}),
    )

    realized = Repo(DirStore(tmp_path / "store")).materialize_boundary(
        roots, reuse_live="never"
    )
    values = tuple(
        item["value"] if isinstance(item, (dict, FrozenDict)) else next(iter(item))
        for item in realized
    )

    assert tuple(type(item) for item in realized) == (
        list, tuple, dict, set, frozenset,
        FrozenList, FrozenTuple, FrozenSet, FrozenDict,
    )
    assert all(value is values[0] for value in values)
    assert values[0].value == 4


def test_conflicting_state_demands_fail_before_any_restore(tmp_path):
    """Two exact roots for one ObjectId reject before a restoration hook runs."""

    repo = Repo(DirStore(tmp_path / "store"))
    value = BoundaryValue(1, repo=repo)
    first = repo.save_object(value)
    value.value = 2
    second = repo.save_object(value, deep_capture=True)
    BoundaryValue.restores = 0

    with pytest.raises(RepoLoadError, match="incompatible effective"):
        Repo(DirStore(tmp_path / "store")).materialize_boundary((first, second))

    assert BoundaryValue.restores == 0


@pytest.mark.parametrize("use_object_refs", (False, True))
def test_overlapping_descendant_state_demands_fail_before_effects(tmp_path, use_object_refs):
    """Explicit and ObjectRef-selected descendant conflicts share one preflight."""

    repo = Repo(DirStore(tmp_path / "store"))
    child = BoundaryValue(1, repo=repo)
    first = repo.save_object(BoundaryPair(child, "first", repo=repo), deep_capture=True)
    child.value = 2
    second = repo.save_object(BoundaryPair(child, "second", repo=repo), deep_capture=True)
    BoundaryValue.constructions = BoundaryValue.restores = 0
    BoundaryPair.constructions = 0
    roots = (first.object, second.object) if use_object_refs else (first, second)

    with pytest.raises(RepoLoadError, match="incompatible effective"):
        Repo(DirStore(tmp_path / "store")).materialize_boundary(roots)

    assert BoundaryValue.constructions == 0
    assert BoundaryValue.restores == 0
    assert BoundaryPair.constructions == 0


def test_enclosing_snapshot_controls_embedded_object_ref_state(tmp_path):
    """An ObjectRef seed does not override its enclosing exact snapshot action."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child_ref = repo.declare_object(Definition(BoundaryValue, 1).concretize(repo=repo))
    child = repo.build_object_ref(child_ref)
    child.value = 7
    repo.save_object(child, deep_capture=True)
    child.value = 9
    enclosing = repo.save_object(
        BoundaryPair(child, "parent", repo=repo), deep_capture=True,
    )

    loaded = Repo(DirStore(tmp_path / "store")).materialize_boundary(
        (enclosing,), reuse_live="never"
    )[0]

    assert loaded.left.value == 9


def test_ref_never_materializes_and_mat_forwards_reuse_policy(tmp_path):
    """Ref is metadata-only while Mat delegates selected exact state to Repo."""

    repo = Repo(DirStore(tmp_path / "store"))
    value = BoundaryValue(1, repo=repo)
    state = repo.save_object(value)
    value.value = 9  # Matching retains the accepted checkpoint-marked payload.
    BoundaryValue.restores = 0

    ref = _single(Ref[StateRef]).prepare_args((state,), {}, repo=repo)
    assert ref.deliver_args()[0][0] is state
    assert BoundaryValue.restores == 0

    mat = _single(Mat[StateRef]).prepare_args((state,), {}, repo=repo, reuse_live="never")
    fresh = mat.deliver_args()[0][0]
    assert fresh is not value
    assert fresh.value == 1


def test_materializing_boundary_is_one_shot_after_failed_admission(tmp_path):
    """A failed BoundaryPlan cannot reacquire claims or repeat effects."""

    repo = Repo(DirStore(tmp_path / "store"))
    reference = repo.declare_object(Definition(BoundaryValue, 8).concretize(repo=repo))
    boundary = _single(Mat[ObjectRef]).prepare_args((reference,), {}, repo=repo)
    repo.default_store.write_claim_record(
        repo.default_store.read_claim_record(reference.digest()).__class__(
            reference.digest(), 1, "claimed", "other", 10**12
        )
    )

    with pytest.raises(RepoLoadError):
        boundary.deliver_args()
    with pytest.raises(SignatureError, match="already been delivered"):
        boundary.deliver_args()


def test_object_ref_claim_is_admitted_once_and_transfers_at_existing_save(tmp_path):
    """Aggregate ObjectRef construction retains the claim until normal save completion."""

    repo = Repo(DirStore(tmp_path / "store"))
    reference = repo.declare_object(Definition(BoundaryValue, 11).concretize(repo=repo))
    boundary = _single(Mat[ObjectRef]).prepare_args((reference,), {}, repo=repo)

    live = boundary.deliver_args()[0][0]

    assert live.object_ref == reference
    assert repo.default_store.read_claim_record(reference.digest()).status == "claimed"
    repo.save_object(live)
    assert repo.default_store.read_claim_record(reference.digest()).status == "completed"


def test_materializing_object_ref_uses_unique_saved_authority_before_claim(tmp_path):
    """A completed ObjectRef selects its sole saved snapshot rather than rebuilding."""

    source = Repo(DirStore(tmp_path / "store"))
    reference = source.declare_object(Definition(BoundaryValue, 12).concretize(repo=source))
    saved = source.build_object_ref(reference)
    saved.value = 14
    source.save_object(saved, deep_capture=True)
    loaded = Repo(DirStore(tmp_path / "store"))

    boundary = _single(Mat[ObjectRef]).prepare_args((reference,), {}, repo=loaded)
    result = boundary.deliver_args()[0][0]

    assert result.object_ref == reference
    assert result.value == 14


def test_pending_parent_reuses_saved_nested_object_ref_without_reclaiming(tmp_path):
    """A pending parent loads its completed child from aggregate preflight."""

    path = tmp_path / "store"
    source = Repo(DirStore(path))
    child_ref = source.declare_object(
        Definition(BoundaryValue, 41).concretize(repo=source)
    )
    child = source.build_object_ref(child_ref)
    child.value = 42
    source.save_object(child, deep_capture=True)
    parent_ref = source.declare_object(
        Definition(BoundaryOwner, child_ref, "saved").concretize(repo=source)
    )

    loaded_repo = Repo(DirStore(path))
    parent = loaded_repo.materialize_boundary(
        (parent_ref,), reuse_live="never"
    )[0]

    assert parent.object_ref == parent_ref
    assert parent.child.object_ref == child_ref
    assert parent.child.value == 42
    assert loaded_repo.default_store.read_claim_record(child_ref.digest()).status == "completed"
    assert loaded_repo.default_store.read_claim_record(parent_ref.digest()).status == "claimed"


def test_pending_parent_reuses_live_nested_object_ref_without_reclaiming(tmp_path):
    """A pending parent receives the exact live child selected in preflight."""

    repo = Repo(DirStore(tmp_path / "store"))
    child_ref = repo.declare_object(
        Definition(BoundaryValue, 51).concretize(repo=repo)
    )
    child = repo.build_object_ref(child_ref)
    child_claim = repo.default_store.read_claim_record(child_ref.digest())
    parent_ref = repo.declare_object(
        Definition(BoundaryOwner, child_ref, "live").concretize(repo=repo)
    )

    parent = repo.materialize_boundary((parent_ref,))[0]
    retained_claim = repo.default_store.read_claim_record(child_ref.digest())

    assert parent.object_ref == parent_ref
    assert parent.child is child
    assert retained_claim.status == "claimed"
    assert retained_claim.generation == child_claim.generation
    assert retained_claim.owner == child_claim.owner


def test_pending_parent_uses_pinned_store_for_nested_claim(tmp_path):
    """Aggregate admission reuses one pinned nested claim across parent roots."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo((first, second))
    child_ref = repo.declare_object(
        Definition(BoundaryValue, 61).concretize(repo=repo), store=first
    )
    claim = first.read_claim_record(child_ref.digest())
    second.write_definition_record(
        DefinitionRecord(child_ref.definition), stored_root=False
    )
    second.write_claim_record(ClaimRecord(
        child_ref.digest(), claim.generation, claim.status
    ))
    second.write_declaration_record(DeclarationRecord(child_ref))
    parent_ref = repo.declare_object(
        Definition(BoundaryOwner, child_ref, "pinned").concretize(repo=repo),
        store=first,
    )

    parent, child = repo.materialize_boundary(
        (parent_ref, child_ref),
        declaration_stores={
            parent_ref.digest(): first,
            child_ref.digest(): second,
        },
    )

    assert parent.child is child
    assert parent._claim_lease.store is first
    assert child._claim_lease.store is second
    assert first.read_claim_record(child_ref.digest()).status == "available"
    assert second.read_claim_record(child_ref.digest()).status == "claimed"


def test_falsey_loaded_object_ref_result_does_not_fall_back_to_claim_build(tmp_path):
    """A valid falsey exact result remains selected without truthiness fallback."""

    source = Repo(DirStore(tmp_path / "store"))
    state = source.save_object(FalseyBoundaryValue(13, repo=source), deep_capture=True)

    result = Repo(DirStore(tmp_path / "store")).materialize_boundary(
        (state.object,), reuse_live="never"
    )[0]

    assert not result
    assert result.object_ref == state.object
    assert result.value == 13


@pytest.mark.parametrize("failure", (RuntimeError("later acquire"), KeyboardInterrupt()))
def test_later_claim_acquisition_failure_abandons_the_first_generation(tmp_path, monkeypatch, failure):
    """Every successful acquisition has cleanup before the next acquisition."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    references = tuple(
        repo.declare_object(Definition(BoundaryValue, value).concretize(repo=repo))
        for value in (21, 22)
    )
    original = repo._acquire_claim
    calls = 0

    def fail_second(reference, selected_store):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return original(reference, selected_store)

    monkeypatch.setattr(repo, "_acquire_claim", fail_second)
    BoundaryValue.constructions = 0
    with pytest.raises(type(failure), match=str(failure) if str(failure) else None):
        repo.materialize_boundary(references)

    assert BoundaryValue.constructions == 0
    assert all(store.read_claim_record(reference.digest()).status == "available" for reference in references)


def test_renewal_failure_runs_independent_reverse_claim_cleanups(tmp_path, monkeypatch):
    """One abandonment error cannot skip cleanup of another acquired generation."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    references = tuple(
        repo.declare_object(Definition(BoundaryValue, value).concretize(repo=repo))
        for value in (31, 32)
    )
    original_abandon = repo._abandon_claim
    abandoned = []

    def fail_one_abandon(lease):
        abandoned.append(lease)
        if len(abandoned) == 1:
            raise OSError("cleanup failed")
        return original_abandon(lease)

    monkeypatch.setattr(repo, "_renew_claim", lambda lease: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(repo, "_abandon_claim", fail_one_abandon)
    with pytest.raises(KeyboardInterrupt) as caught:
        repo.materialize_boundary(references)

    assert len(abandoned) == 2
    if hasattr(caught.value, "add_note"):
        assert any("cleanup failed" in note for note in caught.value.__notes__)
    assert sum(store.read_claim_record(reference.digest()).status == "available" for reference in references) == 1
    for lease in abandoned:
        original_abandon(lease)
