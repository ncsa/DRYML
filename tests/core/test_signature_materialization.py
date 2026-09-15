"""Aggregate Mat delivery proofs for the core signature boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from dryml.core import ConcreteDefinition, Definition, Object, Repo, Serializable
from dryml.core.repo import RepoLoadError
from dryml.core.reference_values import ObjectRef, StateRef
from dryml.core.signatures import Mat, Ref, SignatureError, compile_signature
from dryml.core.store.dir import DirStore


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

    def __init__(self, left, right):
        self.left = left
        self.right = right


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
