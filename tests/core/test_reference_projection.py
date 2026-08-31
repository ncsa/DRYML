import pytest

from dryml.core import Definition, Mat, ObjectId, ObjectRef, Ref, StateRef
from dryml.core.object import Object, Serializable
from dryml.core.utils.graph.path import GraphPath, Parameter


class ReferenceLeaf(Serializable):
    def __init__(self, value=1):
        self.value = value


class ReferenceWrapper(Object):
    def __init__(self, child, child_alias=None):
        self.child = child
        self.child_alias = child_alias


def _state_hash(char="a"):
    return "pkl-" + char * 64


def test_object_ref_derives_primary_stateful_paths_and_rebases_subtrees():
    leaf = Definition(ReferenceLeaf).concretize()
    root = Definition(ReferenceWrapper, leaf).concretize()
    child_path = GraphPath((Parameter("child"),))
    object_ref = ObjectRef(root, {child_path: ObjectId(("test",))})

    projected = object_ref.at(child_path)
    assert projected.definition.graph_equal(leaf)
    assert tuple(projected.objects) == (GraphPath(),)
    assert projected.object_id == object_ref.objects[child_path]

    state = StateRef(object_ref, {child_path: _state_hash()})
    assert state.at(child_path).states == {GraphPath(): _state_hash()}


def test_reference_values_reject_missing_duplicate_and_malformed_state_entries():
    leaf = Definition(ReferenceLeaf).concretize()
    root = Definition(ReferenceWrapper, leaf, child_alias=leaf).concretize()
    child_path = GraphPath((Parameter("child"),))
    alias_path = GraphPath((Parameter("child_alias"),))
    object_id = ObjectId()

    with pytest.raises(ValueError, match="primary paths"):
        ObjectRef(root, {alias_path: object_id})
    with pytest.raises(ValueError, match="exactly"):
        StateRef(ObjectRef(root, {child_path: object_id}), {})
    with pytest.raises(ValueError, match="State hashes"):
        StateRef(ObjectRef(root, {child_path: object_id}), {child_path: "bad"})


def test_materializing_exact_references_expand_but_ref_edges_do_not():
    leaf = Definition(ReferenceLeaf).concretize()
    leaf_ref = ObjectRef(leaf, {GraphPath(): ObjectId(("imported",))})
    child_path = GraphPath((Parameter("child"),))

    imported = Definition(ReferenceWrapper, leaf_ref).concretize()
    expanded = ObjectRef(imported, {child_path: leaf_ref.object_id})
    assert expanded.at(child_path) == leaf_ref

    ref_only = Definition(ReferenceWrapper, Ref(leaf_ref)).concretize()
    assert ObjectRef(ref_only, {}).objects == {}
    with pytest.raises(ValueError, match="Ref-only"):
        ObjectRef(ref_only, {}).at(child_path)


def test_imported_object_ids_cannot_be_substituted_by_an_enclosing_reference():
    leaf = Definition(ReferenceLeaf).concretize()
    imported = ObjectRef(leaf, {GraphPath(): ObjectId(("imported",))})
    child_path = GraphPath((Parameter("child"),))
    enclosing = Definition(ReferenceWrapper, imported).concretize()

    with pytest.raises(ValueError, match="embedded ObjectId"):
        ObjectRef(enclosing, {child_path: ObjectId(("enclosing",))})


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("with_state", [False, True])
def test_imported_reference_projections_traverse_and_rebase_nested_topology(
    materialized, with_state
):
    leaf = Definition(ReferenceLeaf).concretize()
    local_child = GraphPath((Parameter("child"),))
    local_alias = GraphPath((Parameter("child_alias"),))
    object_id = ObjectId(("imported",))
    imported_definition = Definition(
        ReferenceWrapper, leaf, child_alias=leaf
    ).concretize()
    imported = ObjectRef(imported_definition, {local_child: object_id})
    imported_value = (
        StateRef(imported, {local_child: _state_hash("c")})
        if with_state
        else imported
    )
    if materialized:
        imported_value = Mat(imported_value)
    outer_definition = Definition(
        ReferenceWrapper, imported_value, child_alias=imported_value
    ).concretize()
    imported_path = GraphPath((Parameter("child"),))
    imported_alias_path = GraphPath((Parameter("child_alias"),))
    nested_path = imported_path.join(local_child)
    alias_path = imported_path.join(local_alias)
    outer = ObjectRef(outer_definition, {nested_path: object_id})
    state = StateRef(outer, {nested_path: _state_hash("d")})

    imported_projection = outer.at(imported_path)
    assert imported_projection.definition.graph_equal(imported_definition)
    assert imported_projection.objects == {local_child: object_id}
    assert state.at(imported_path).states == {local_child: _state_hash("d")}

    for path in (nested_path, alias_path, imported_alias_path.join(local_child)):
        projected = outer.at(path)
        assert projected.definition.graph_equal(leaf)
        assert projected.objects == {GraphPath(): object_id}
        assert state.at(path).states == {GraphPath(): _state_hash("d")}
