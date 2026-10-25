from pathlib import Path

import pytest

from dryml.core import Definition, Object, ObjectId, QuotedDef, Ref, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter


class ScopeLeaf(Object):
    constructed = 0

    def __init__(self, value):
        super().__init__()
        type(self).constructed += 1
        self.value = value


class ScopeParent(Object):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right


def test_aliases_share_but_independent_equal_nodes_do_not():
    repo = Repo()
    shared = Definition(ScopeLeaf, "same")
    aliased = repo.load_or_build(Definition(ScopeParent, shared, shared))
    first = Definition(ScopeLeaf, "same")
    second = Definition(ScopeLeaf, "same")
    distinct = repo.load_or_build(Definition(ScopeParent, first, second), cache="none")

    assert aliased.left is aliased.right
    assert distinct.left is not distinct.right
    assert distinct.left.definition.graph_equal(distinct.right.definition)
    assert distinct.left._realization_scope == distinct.right._realization_scope
    assert aliased._realization_scope != distinct._realization_scope


class QuotedScopeValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class QuotedScopeParent(Serializable):
    def __init__(self, left, right, recipe: Ref[QuotedDef]):
        self.left = left
        self.right = right
        self.recipe = recipe


@pytest.mark.usefixtures("fixed_snapshot_environment")
@pytest.mark.parametrize("source", ["fresh", "declared", "saved"])
def test_realization_and_capture_preserve_quoted_owned_alias(tmp_path, source):
    from dryml.core.node import _owned_nodes

    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    child = Definition(QuotedScopeValue, 1).concretize()
    if source == "declared":
        child = child.with_identity(ObjectId())
    elif source == "saved":
        seed = repo.build(child)
        seed.value = 7
        child = repo.save_object(seed)
    recipe = Definition(QuotedScopeParent, child, child, child.quote()).concretize()
    if source == "declared":
        recipe = recipe.with_identity(ObjectId())
        store.publish_node(recipe)
    quote = recipe.parameters["recipe"]
    child_path = GraphPath((Parameter("left"),))

    live = repo.build(recipe, cache="none")
    described = repo.describe(live)
    assert live.left is live.right
    assert live.left.value == (7 if source == "saved" else 1)
    assert described.parameters["recipe"] is quote
    assert described.at(child_path).object_id == live.left.object_id
    assert described.creation_digest() == recipe.creation_digest()

    live.left.value = 11
    saved = repo.save_object(live)
    assert saved.parameters["recipe"] is quote
    assert saved.object_projection_equal(described)
    assert saved.at(child_path) is saved.at(GraphPath((Parameter("right"),)))
    assert saved.at(child_path).state is not None and saved.state is not None
    assert saved.creation_digest() == recipe.creation_digest()
    assert quote.value is child

    unpinned = saved.with_realization(states={path: None for path, _ in _owned_nodes(saved)})
    assert unpinned.parameters["recipe"] is quote
    assert all(node.state is None for _, node in _owned_nodes(unpinned))
    assert unpinned.object_projection_equal(saved)

    live.left.value = 19
    repo.apply_state(live, {child_path: saved.at(child_path).state})
    applied = repo.describe(live)
    assert live.left is live.right and live.left.value == 11
    assert applied.parameters["recipe"] is quote
    assert applied.at(child_path).state == saved.at(child_path).state
    assert applied.object_projection_equal(described)
    recaptured = repo.save_object(live)
    assert recaptured.parameters["recipe"] is quote
    assert recaptured.object_projection_equal(saved)

    live.left.value = 23
    repo.restore(live, saved)
    assert live.left is live.right and live.left.value == 11
    assert repo.describe(live).object_projection_equal(saved)
    loaded = repo.load_state_ref(saved, reuse_live="never")
    loaded_view = repo.describe(loaded)
    assert loaded.left is loaded.right and loaded.left.value == 11
    assert loaded_view.object_projection_equal(saved)
    assert loaded_view.parameters["recipe"].value.graph_equal(child)
    assert loaded_view.creation_digest() == recipe.creation_digest()
