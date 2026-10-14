"""Focused recursive object-projection contract tests."""

import pytest

from dryml.core import (
    Definition,
    IdentitySet,
    ObjectId,
    ObjectRef,
    ObjectSelector,
    ObjectSelectorSet,
    Ref,
    SourceEvidence,
    StateRef,
    Template,
)
from dryml.core.cdef_graph import EdgeKind
from dryml.core.cdef_codec import CDefGraphCodecError, encode_cdef_graph
from dryml.core.links import DefLink
from dryml.core.object import Serializable
from dryml.core.symbol import ImportRef
from dryml.core.template import Par
from dryml.core.utils.graph.path import GraphPath, Parameter


class ProjectionLeaf(Serializable):
    """Stateful leaf used to retain exact reference identity in projection tests."""

    def __init__(self, value):
        self.value = value


class ProjectionRoot(Serializable):
    """Stateful container with materializing, reference, and recipe fields."""

    def __init__(self, materialized, referenced: Ref[StateRef], recipe: Template):
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
    """Projection creates recursive selectors without changing graph association data."""

    imported = _state_ref("child", "imported", _state_hash("a"))
    recipe = Definition(ProjectionLeaf, {"exact": imported, "width": Par("width")})
    root = Definition(
        ProjectionRoot,
        DefLink.finalized(EdgeKind.MATERIALIZE, imported),
        referenced=DefLink.finalized(EdgeKind.REF, imported),
        recipe=recipe,
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

    assert isinstance(projected, ObjectSelector)
    assert projected.objects == state.object.objects
    assert root.object_projection().graph_equal(projected.definition)
    assert referenced.kind is EdgeKind.REF
    assert materialized is referenced.target
    assert isinstance(materialized, ObjectSelector)
    assert opaque_recipe.target.value == recipe
    assert ObjectRef.from_data(projected.reference.to_data()) == projected.reference
    with pytest.raises(CDefGraphCodecError, match="query-only"):
        encode_cdef_graph(projected.definition)

    traversed = state.object_projection(traverse_refs=True)
    traversed_recipe = traversed.definition.parameters["recipe"].target.value
    assert traversed != projected
    projected_exact = traversed_recipe.args[0]["exact"]
    assert isinstance(projected_exact, ObjectSelector)


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


def test_object_selector_projection_deduplicates_states_and_preserves_exact_pins():
    """Generated references are selectors while manually embedded references stay exact."""

    child_object = _state_ref("child", "child", _state_hash("a")).object
    first_child = StateRef(child_object, {GraphPath(): _state_hash("a")})
    second_child = StateRef(child_object, {GraphPath(): _state_hash("b")})
    root_id = ObjectId(("run",))

    def checkpoint(child, root_state):
        definition = Definition(
            ProjectionCarrier,
            reference=DefLink.finalized(EdgeKind.REF, child),
        ).concretize()
        reference = ObjectRef(definition, {GraphPath(): root_id})
        return StateRef(reference, {GraphPath(): root_state})

    first = checkpoint(first_child, _state_hash("c"))
    second = checkpoint(second_child, _state_hash("d"))
    other_definition = Definition(
        ProjectionCarrier,
        reference=DefLink.finalized(EdgeKind.REF, second_child),
    ).concretize()
    other = StateRef(
        ObjectRef(other_definition, {GraphPath(): ObjectId(("other",))}),
        {GraphPath(): _state_hash("e")},
    )

    generated = first.object_projection()
    assert generated == second.object_projection()
    assert generated.reference == second.object_projection().reference
    assert generated != other.object_projection()
    assert generated.matches(first)
    assert generated.matches(second)

    pinned = ObjectSelector(first.object)
    assert pinned.matches(first)
    assert not pinned.matches(second)

    first_source = SourceEvidence.from_source("first")
    second_source = SourceEvidence.from_source("second")
    identities = IdentitySet((
        (first, first_source),
        (second, second_source),
        other,
        first.definition,
    ))
    assert identities.query().sel(generated).state_refs().count() == 2
    assert identities.query().sel(pinned).state_refs().one() == first
    assert IdentitySet((first.definition, second.definition)).query().sel(
        first.definition.object_projection()
    ).count() == 2

    projected = identities.object_projection()
    assert isinstance(projected, ObjectSelectorSet)
    assert projected.count() == 2
    assert projected.sources(generated) == frozenset((first_source, second_source))
    assert identities.query().object_projection().count() == 2


def test_reference_value_at_accepts_only_terminal_ref_data():
    """Exact Ref data is available only at the terminal declared graph boundary."""

    imported = _state_ref("child", "imported", _state_hash("a"))
    recipe = Definition(ProjectionLeaf, {"exact": imported})
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
        recipe=recipe,
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
