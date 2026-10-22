"""Aggregate Mat delivery proofs for the core signature boundary."""

from __future__ import annotations

from abc import abstractmethod, update_abstractmethods
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import ConcreteDefinition, Definition, Object, Repo, Serializable, SavedNode, Node, ObjectId, Template
from dryml.core.freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from dryml.core.repo import RepoLoadError
from dryml.core.signatures import Mat, Ref, SignatureError, compile_signature
from dryml.core.store.dir import DirStore
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


class AbstractableBoundaryValue(BoundaryValue):
    """Stateful fixture that can become abstract after snapshot publication."""


class BoundaryMetadata(Object):
    """Inert selected saved metadata held across a constructor Ref boundary."""

    def __init__(self, reference: Ref[SavedNode]):
        self.reference = reference


class BoundarySoftMetadata(Object):
    """Resolved but deliberately unnormalized reference metadata."""

    def __init__(self, reference: Ref[Node]):
        self.reference = reference


class BoundaryTemplate(Object):
    """Quoted saved metadata is exact future configuration, not owned state."""

    def __init__(self, recipe: Template[Node]):
        self.recipe = recipe


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


def test_typed_mat_target_owns_one_normalization_and_rejects_unresolved_before_hooks(monkeypatch):
    """Mat[Object subclass] checks its target and never constructs active symbols."""

    from dryml.core import Par
    from dryml.core.errors import UnresolvedDefinitionError

    repo = Repo()
    BoundaryValue.constructions = 0
    unresolved = _single(Mat[BoundaryValue]).prepare_args((Definition(BoundaryValue, Par("value")),), {}, repo=repo)
    with pytest.raises(UnresolvedDefinitionError):
        unresolved.deliver_args()
    assert BoundaryValue.constructions == 0
    with pytest.raises(SignatureError):
        _single(Mat[BoundaryValue]).prepare_args((Definition(BoundaryPair, 1, 2),), {}, repo=repo)
    with monkeypatch.context() as pure_selection:
        pure_selection.setattr(Definition, "concretize", lambda *a, **k: pytest.fail("selection concretized early"))
        defaulted = _single(Mat[BoundaryValue]).prepare_args((Definition(BoundaryValue),), {}, repo=repo)
    assert defaulted.authority["value"].parameters == {}
    delivered = defaulted.deliver_args()[0][0]
    assert delivered.value == 3
    assert BoundaryValue.constructions == 1


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
    assert separate.definition.graph_equal(root.definition)


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


def test_complementary_partial_views_compose_before_boundary_hooks(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    donor = BoundaryValue(1, repo=repo)
    donor.value = 9
    state = repo.save(donor).state
    recipe = Definition(BoundaryValue, 1).concretize()
    selected = recipe.with_state(state)
    BoundaryValue.restores = 0
    plain, restored = repo.materialize_boundary((recipe, selected), cache="none")
    assert plain is restored and restored.value == 9
    assert BoundaryValue.restores == 1


def test_overlapping_descendant_state_demands_fail_before_effects(tmp_path):
    """Saved descendant conflicts share one Node boundary preflight."""

    repo = Repo(DirStore(tmp_path / "store"))
    child = BoundaryValue(1, repo=repo)
    first = repo.save_object(BoundaryPair(child, "first", repo=repo), deep_capture=True)
    child.value = 2
    second = repo.save_object(BoundaryPair(child, "second", repo=repo), deep_capture=True)
    BoundaryValue.constructions = BoundaryValue.restores = 0
    BoundaryPair.constructions = 0
    roots = (first, second)

    with pytest.raises(RepoLoadError, match="incompatible effective"):
        Repo(DirStore(tmp_path / "store")).materialize_boundary(roots)

    assert BoundaryValue.constructions == 0
    assert BoundaryValue.restores == 0
    assert BoundaryPair.constructions == 0


def test_enclosing_snapshot_controls_embedded_object_ref_state(tmp_path):
    """Retained construction selection does not override saved realization state."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = Definition(BoundaryValue, 1).build(repo=repo)
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

    ref = _single(Ref[SavedNode]).prepare_args((state,), {}, repo=repo)
    assert ref.deliver_args()[0][0] is state
    assert BoundaryValue.restores == 0

    mat = _single(Mat[BoundaryValue]).prepare_args((state,), {}, repo=repo, reuse_live="never")
    fresh = mat.deliver_args()[0][0]
    assert fresh is not value
    assert fresh.value == 1


def test_ref_metadata_in_a_built_parent_retains_original_selection_without_restore(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    saved = repo.save(BoundaryValue(1, repo=repo))
    BoundaryValue.constructions = BoundaryValue.restores = 0
    parent = Definition(BoundaryMetadata, saved).build(repo=repo, cache="none")
    assert parent.reference is saved
    assert parent.graph_at(GraphPath((Parameter("reference"),))) is saved
    assert BoundaryValue.constructions == BoundaryValue.restores == 0


def test_ref_soft_node_build_never_fills_reference_defaults_or_allocates_it():
    metadata = Definition(BoundaryValue).with_identity(ObjectId())
    BoundaryValue.constructions = BoundaryValue.restores = 0
    parent = Definition(BoundarySoftMetadata, metadata).build(cache="none")
    assert parent.reference is metadata and not parent.reference.parameters
    assert BoundaryValue.constructions == BoundaryValue.restores == 0


def test_quoted_saved_metadata_build_and_save_never_restore_or_own_it(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    metadata = repo.save(BoundaryValue(1, repo=repo))
    BoundaryValue.constructions = BoundaryValue.restores = 0
    parent = Definition(BoundaryTemplate, metadata).build(repo=repo, cache="none")
    assert parent.recipe is metadata
    saved = repo.save(parent)
    assert not saved.state_spec()
    assert BoundaryValue.constructions == BoundaryValue.restores == 0


def test_materializing_boundary_is_one_shot_after_failed_admission(tmp_path):
    """A failed BoundaryPlan cannot repeat effects after missing-state admission."""

    repo = Repo(DirStore(tmp_path / "store"))
    reference = repo.save(BoundaryValue(8, repo=repo))
    boundary = _single(Mat[BoundaryValue]).prepare_args((reference,), {}, repo=repo)
    Path(repo.default_store.state_data_path(reference.state.key), "value").unlink()

    with pytest.raises(RepoLoadError):
        boundary.deliver_args()
    with pytest.raises(SignatureError, match="already been delivered"):
        boundary.deliver_args()


def test_identity_bound_node_is_admitted_once_and_preserved_at_save(tmp_path):
    """Explicit identity requires no retired persisted declaration or claim."""

    from dryml.core import ObjectId
    repo = Repo(DirStore(tmp_path / "store"))
    identity = ObjectId()
    reference = Definition(BoundaryValue, 11).with_identity(identity).concretize(repo=repo)
    boundary = _single(Mat[BoundaryValue]).prepare_args((reference,), {}, repo=repo)

    live = boundary.deliver_args()[0][0]

    assert live.object_id == identity
    assert repo.save_object(live).object_id == identity


def test_materializing_decoded_saved_node_uses_selected_runtime_class(tmp_path, monkeypatch):
    """A decoded saved graph resolves its own admitted materializing target."""

    from dryml.core import materialization
    from dryml.core.cdef_identity import cdef_node_key

    source = Repo(DirStore(tmp_path / "store"))
    saved = BoundaryValue(12, repo=source)
    saved.value = 14
    reference = source.save_object(saved, deep_capture=True)
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    from dryml.core.node_prepare import NodePreparation
    decoded = decode_node_graph(encode_node_graph(NodePreparation((reference,))))
    loaded = Repo(DirStore(tmp_path / "store"))
    caller_node = cdef_node_key(reference)
    original_resolve = materialization._resolve_materialization_class
    calls = []

    def verify_selected_class_manifest(node):
        assert cdef_node_key(node) is not caller_node
        calls.append(node)
        return original_resolve(node)

    monkeypatch.setattr(
        materialization, "_resolve_materialization_class", verify_selected_class_manifest,
    )

    boundary = _single(Mat[BoundaryValue]).prepare_args((decoded,), {}, repo=loaded)
    result = boundary.deliver_args()[0][0]

    assert calls and result.object_id == reference.object_id
    assert result.value == 14


def test_aggregate_preflights_later_selected_snapshot_before_earlier_construction(tmp_path):
    """An abstract selected snapshot prevents earlier ordinary construction."""

    store = DirStore(tmp_path / "store")
    source = Repo(store)
    pending = Definition(BoundaryValue, 71).concretize(repo=source)
    selected = source.save_object(AbstractableBoundaryValue(72, repo=source))
    BoundaryValue.constructions = 0

    def required(self):
        """Return the newly required implementation result."""

    AbstractableBoundaryValue.required = abstractmethod(required)
    update_abstractmethods(AbstractableBoundaryValue)
    try:
        with pytest.raises(TypeError, match="required"):
            Repo(DirStore(tmp_path / "store")).materialize_boundary(
                (pending, selected), reuse_live="never",
            )
    finally:
        del AbstractableBoundaryValue.required
        update_abstractmethods(AbstractableBoundaryValue)

    assert BoundaryValue.constructions == 0


def test_partial_parent_reuses_selected_saved_child_once(tmp_path):
    """A new parent restores its selected child in the aggregate boundary."""

    path = tmp_path / "store"
    source = Repo(DirStore(path))
    child = BoundaryValue(41, repo=source)
    child.value = 42
    child_ref = source.save_object(child, deep_capture=True)
    parent_ref = Definition(BoundaryOwner, child_ref, "saved").concretize(repo=source)

    loaded_repo = Repo(DirStore(path))
    BoundaryValue.restores = 0
    parent, selected_child = loaded_repo.materialize_boundary((parent_ref, child_ref), reuse_live="never")

    assert parent.child is selected_child
    assert parent.child.object_id == child_ref.object_id
    assert parent.child.value == 42
    assert BoundaryValue.restores == 1


def test_partial_parent_retains_nested_live_node_cache_identity(tmp_path):
    """An unspecified child uses the existing live occurrence cache rule."""

    repo = Repo(DirStore(tmp_path / "store"))
    child_ref = Definition(BoundaryValue, 51).concretize(repo=repo)
    child = repo.build(child_ref)
    parent_ref = Definition(BoundaryOwner, child_ref, "live").concretize(repo=repo)

    parent = repo.materialize_boundary((parent_ref,))[0]
    assert parent.child is child


def test_full_key_source_override_selects_only_connected_available_source(tmp_path):
    """An explicit full-key source is used without a sibling or replica scan."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo((first, second))
    saved = repo.save(BoundaryValue(61, repo=repo), store=second)
    child = repo.load_state_ref(saved, reuse_live="never", source_stores={saved.state.key: second})
    assert child.value == 61
    with pytest.raises(RepoLoadError, match="unavailable"):
        repo.load_state_ref(saved, source_stores={saved.state.key: first})


def test_falsey_loaded_object_ref_result_does_not_fall_back_to_claim_build(tmp_path):
    """A valid falsey exact result remains selected without truthiness fallback."""

    source = Repo(DirStore(tmp_path / "store"))
    state = source.save_object(FalseyBoundaryValue(13, repo=source), deep_capture=True)

    result = Repo(DirStore(tmp_path / "store")).materialize_boundary(
        (state,), reuse_live="never"
    )[0]

    assert not result
    assert source.describe(result).object_projection_equal(state)
    assert result.value == 13


@pytest.mark.parametrize("failure", (RuntimeError("later acquire"), KeyboardInterrupt()))
def test_later_source_acquisition_failure_releases_earlier_sources(tmp_path, monkeypatch, failure):
    """Source admission interruption runs independent reader/lease cleanups."""

    from contextlib import contextmanager
    from dryml.core.store.retention import _held
    first, second = DirStore(tmp_path / "first"), DirStore(tmp_path / "second")
    repo = Repo((first, second))
    references = tuple(
        repo.save(BoundaryValue(value, repo=repo), store=store)
        for value, store in ((21, first), (22, second))
    )
    original = first.read_states
    released = []

    @contextmanager
    def tracked(keys):
        try:
            with original(keys) as sources:
                yield sources
        finally:
            released.append(True)

    @contextmanager
    def fail_second(keys):
        raise failure
        yield

    monkeypatch.setattr(first, "read_states", tracked)
    monkeypatch.setattr(second, "read_states", fail_second)
    BoundaryValue.constructions = 0
    with pytest.raises(RepoLoadError if isinstance(failure, Exception) else KeyboardInterrupt):
        repo.materialize_boundary(references)

    assert BoundaryValue.constructions == 0
    assert released == [True]
    assert not _held()


def test_post_hook_boundary_failure_invalidates_earlier_completed_root(tmp_path, monkeypatch):
    """A later root failure cannot leave earlier restored roots reusable."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    references = tuple(
        repo.save(BoundaryValue(value, repo=repo))
        for value in (31, 32)
    )
    restored = []
    original = BoundaryValue.restore_state_from_dir_imp

    def restore(target, directory, *, codec):
        restored.append(target)
        original(target, directory, codec=codec)
        if len(restored) == 2:
            raise RuntimeError("second root hook")

    monkeypatch.setattr(BoundaryValue, "restore_state_from_dir_imp", restore)
    with pytest.raises(RepoLoadError, match="fresh materialization"):
        repo.materialize_boundary(references, reuse_live="never")
    assert len(restored) == 2 and all(target._restore_failed for target in restored)
