"""Focused recursive object-projection contract tests."""

import pytest

from dryml.core import Definition, ObjectId, ObjectRef, Ref, StateRef
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.object import Serializable
from dryml.core.symbol import ImportRef
from dryml.core.template import Par, Template
from dryml.core.utils.graph.path import GraphPath, Parameter


class ProjectionLeaf(Serializable):
    """Stateful leaf used to retain exact reference identity in projection tests."""

    def __init__(self, value):
        self.value = value


class ProjectionRoot(Serializable):
    """Stateful container with materializing, reference, and recipe fields."""

    def __init__(self, materialized, referenced: Ref[StateRef], recipe: Ref[Template]):
        self.materialized = materialized
        self.referenced = referenced
        self.recipe = recipe


class ProjectionCarrier(Serializable):
    """Stateful materializing bridge to a terminal exact reference field."""

    def __init__(self, reference: Ref[StateRef]):
        self.reference = reference


def _state_hash(char):
    return "pkl-" + char * 64


def _state_ref(value, namespace, state_hash):
    definition = Definition(ProjectionLeaf, value).concretize()
    object_ref = ObjectRef(definition, {GraphPath(): ObjectId((namespace,))})
    return StateRef(object_ref, {GraphPath(): state_hash})


def test_recursive_object_projection_preserves_ids_roles_sharing_and_codec_meaning():
    """Projection weakens nested state references without changing graph association data."""

    imported = _state_ref("child", "imported", _state_hash("a"))
    recipe = Template.from_value({"exact": imported, "width": Par("width")})
    root = Definition(
        ProjectionRoot,
        DefLink.finalized(EdgeKind.MATERIALIZE, imported),
        referenced=DefLink.finalized(EdgeKind.REF, imported),
        recipe=DefLink.finalized(EdgeKind.REF, recipe),
    ).concretize()
    root_path = GraphPath()
    child_path = GraphPath((Parameter("materialized"),))
    root_ref = ObjectRef(
        root,
        {root_path: ObjectId(("root",)), child_path: imported.object_id},
    )
    state = StateRef(
        root_ref,
        {root_path: _state_hash("b"), child_path: imported.states[GraphPath()]},
    )

    projected = state.object_projection()
    materialized = projected.definition.parameters["materialized"]
    referenced = projected.definition.parameters["referenced"]
    opaque_recipe = projected.definition.parameters["recipe"]

    assert projected.objects == state.object.objects
    assert root.object_projection().graph_equal(projected.definition)
    assert referenced.kind is EdgeKind.REF
    assert materialized is referenced.target
    assert isinstance(materialized, ObjectRef)
    assert opaque_recipe.target is recipe
    assert opaque_recipe.target.root["exact"] is imported
    assert projected.object_projection() == projected
    assert ObjectRef.from_data(projected.to_data()) == projected

    traversed = state.object_projection(traverse_refs=True)
    traversed_recipe = traversed.definition.parameters["recipe"].target
    assert traversed_recipe.root["width"] == Par("width")
    assert traversed_recipe.root["exact"] == imported.object


def test_projection_never_allocates_object_ids_or_resolves_symbols(monkeypatch):
    """Projection is pure graph rewriting rather than a construction boundary."""

    state = _state_ref("child", "imported", _state_hash("a"))

    def fail_allocation(self, namespace=None):
        raise AssertionError("object projection must not allocate ObjectIds")

    def fail_resolution(*args, **kwargs):
        raise AssertionError("object projection must not resolve symbols")

    monkeypatch.setattr(ObjectId, "__init__", fail_allocation)
    monkeypatch.setattr(ImportRef, "resolve", fail_resolution)

    assert state.object_projection().objects == state.object.objects


def test_reference_value_at_accepts_only_terminal_ref_data():
    """Exact Ref data is available only at the terminal declared graph boundary."""

    imported = _state_ref("child", "imported", _state_hash("a"))
    recipe = Template.from_value({"exact": imported})
    carrier_definition = Definition(
        ProjectionCarrier, reference=DefLink.finalized(EdgeKind.REF, imported)
    ).concretize()
    carrier_object = ObjectRef(
        carrier_definition, {GraphPath(): ObjectId(("carrier",))}
    )
    carrier = StateRef(carrier_object, {GraphPath(): _state_hash("c")})
    root = Definition(
        ProjectionRoot,
        DefLink.finalized(EdgeKind.MATERIALIZE, carrier),
        referenced=DefLink.finalized(EdgeKind.REF, imported),
        recipe=DefLink.finalized(EdgeKind.REF, recipe),
    ).concretize()
    root_path = GraphPath()
    child_path = GraphPath((Parameter("materialized"),))
    root_ref = ObjectRef(
        root,
        {root_path: ObjectId(("root",)), child_path: carrier.object_id},
    )
    state = StateRef(
        root_ref,
        {root_path: _state_hash("b"), child_path: carrier.states[GraphPath()]},
    )

    assert state.reference_value_at("referenced") is imported
    assert state.reference_value_at(
        GraphPath((Parameter("materialized"), Parameter("reference")))
    ) is imported
    with pytest.raises(ValueError, match="terminal Ref"):
        state.reference_value_at("materialized")
    with pytest.raises(ValueError, match="Ref boundary"):
        state.reference_value_at("referenced.value")
    with pytest.raises(ValueError, match="terminal Ref"):
        state.reference_value_at("recipe")
    with pytest.raises(ValueError, match="Ref boundary"):
        state.reference_value_at("recipe.exact")
    with pytest.raises(ValueError, match="invalid"):
        state.reference_value_at("missing")
    with pytest.raises(ValueError, match="Ref-only"):
        state.at(GraphPath((Parameter("referenced"),)))
