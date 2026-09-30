"""Focused substitution, remapping, and reference-projection tests."""

from __future__ import annotations

import pytest

from dryml.core import ConcreteDefinition, Definition, F, Mat, Object, ObjectId, ObjectRef, Ref, Serializable, StateRef
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_graph import EdgeKind
from dryml.core.domains import UniformFromSet
from dryml.core.errors import ParameterizationError, UnresolvedDefinitionError
from dryml.core.links import DefLink
from dryml.core.symbol import ImportRef
from dryml.core.template import Par, Template
from dryml.core.utils.graph.path import GraphPath, Parameter


class SubstitutionLeaf(Object):
    """Small Object that supplies retained graph evidence for projection tests."""

    def __init__(self, value):
        self.value = value


class SubstitutionRoot(Object):
    """Object with a retained child path for static live-object binding tests."""

    def __init__(self, child):
        self.child = child


class SubstitutionCheckpoint(Serializable):
    """Stateful binding root with exact model and reference-data fields."""

    def __init__(self, model, test_data: Ref[StateRef]):
        self.model = model
        self.test_data = test_data


class CheckpointLeaf(Serializable):
    """Stateful child used to distinguish checkpoint-local state selections."""

    def __init__(self, value):
        self.value = value


class SubstitutionTarget:
    """Inert construction target used to ensure factory values are not invoked."""

    calls = 0

    def __init__(self, value):
        type(self).calls += 1
        self.value = value


class CallShapeTarget:
    """Target whose declaration makes soft positional spelling observable."""

    def __init__(self, first, /, *rest, label):
        self.first = first
        self.rest = rest
        self.label = label


class ProbeDistribution:
    """Provider sentinel that proves substitution never samples static values."""

    sampled = 0

    def sample(self, rng):
        type(self).sampled += 1
        return 1

    def cardinality(self):
        return 1

    def value_at(self, index):
        return 1

    def contains(self, value):
        return value == 1

    def bounds(self):
        return 1, 1


def test_substitution_is_single_pass_and_namespace_mapping_is_exact():
    """Existing occurrences bind once while introduced template names await a later call."""
    inner = Template.from_value({"inner": Par("width")})
    outer = Template.from_value({"outer": Par("width"), "model": Par("model")})

    result = outer.sub(sub_dict={"model": inner}, width=64)

    assert result.root["outer"] == 64
    assert result.root["model"] == inner.root
    assert result.names == ("width",)
    assert result.sub(width=32).root["model"]["inner"] == 32


def test_substitution_rejects_duplicate_unknown_and_nested_distributions_before_rewrite():
    """Binding validation completes before a supplied provider can participate."""
    template = Template.from_value({"value": Par("width")})
    provider = UniformFromSet((1, 2))

    with pytest.raises(ParameterizationError, match="duplicate"):
        template.sub(sub_dict={"width": 1}, width=2)
    with pytest.raises(ParameterizationError, match="unknown"):
        template.sub(height=1)
    with pytest.raises(ParameterizationError, match="Distribution"):
        template.sub(width={"provider": provider})
    ProbeDistribution.sampled = 0
    with pytest.raises(ParameterizationError, match="Distribution"):
        template.sub(width=ProbeDistribution())
    assert ProbeDistribution.sampled == 0
    assert template.root["value"] == Par("width")


def test_control_named_roots_and_qualified_keyword_namespace_bindings():
    """Mapping keys address controls while keyword bindings receive the namespace."""
    template = Template.from_value(
        {"control": Par("traverse_refs"), "nested": Par("encoder/width")}
    )

    result = template.sub(sub_dict={"traverse_refs": 1}, namespace="encoder", width=64)

    assert result.root == {"control": 1, "nested": 64}


def test_remap_is_simultaneous_and_ref_templates_are_opaque_unless_requested():
    """Remapping changes the pre-existing closure and retains quoted recipes by default."""
    recipe = Template.from_value({"inside": Par("width")})
    template = Template.from_value(
        {"left": Par("width"), "right": Par("height"), "recipe": Ref(recipe)}
    )

    remapped = template.remap({"width": "size", "height": "size"}, prefix="encoder")
    assert remapped.names == ("encoder/size",)
    assert remapped.root["recipe"].target is recipe

    traversed = template.remap(prefix="encoder", traverse_refs=True)
    assert traversed.names == ("encoder/width", "encoder/height")
    assert traversed.root["recipe"].target.root["inside"] == Par("encoder/width")
    assert recipe.root["inside"] == Par("width")


def test_factory_values_are_rewritten_inertly_and_preserve_aliases():
    """One memoized rewrite retains shared factory arguments without resolving targets."""
    SubstitutionTarget.calls = 0
    shared = Par("width")
    factory = F(SubstitutionTarget, shared)
    template = Template.from_value({"first": factory, "alias": factory, "second": shared})

    result = template.sub(width=64)

    assert result.root["first"].args[0] == 64
    assert result.root["first"] is result.root["alias"]
    assert result.root["second"] == 64
    assert SubstitutionTarget.calls == 0


def test_repeated_root_bindings_reuse_one_detached_container_value():
    """Distinct occurrences of one root receive the same frozen replacement object."""
    template = Template.from_value({"left": Par("shared"), "right": Par("shared")})

    result = template.sub(shared=["value"])

    assert result.root["left"] is result.root["right"]


def test_live_projection_lowers_selected_objects_without_state_lookup_or_save(monkeypatch):
    """Live paths project before lowering while explicit StateRef values remain exact."""
    child = SubstitutionLeaf("child")
    root = SubstitutionRoot(child)
    state = StateRef(child.object_ref, {})
    template = Template.from_value({"child": Par("this.child"), "state": Par("state")})

    monkeypatch.setattr(
        SubstitutionLeaf,
        "last_state_ref",
        property(lambda self: pytest.fail("substitution must not select last state")),
    )
    result = template.sub(this=root, state=state)

    assert result.root["child"] == child.object_ref
    assert isinstance(result.root["child"], ObjectRef)
    assert result.root["state"] is state


def test_checkpoint_binding_keeps_model_and_ref_test_data_at_the_exact_state():
    """StateRef path binding crosses Mat edges but returns terminal Ref data unchanged."""

    leaf = Definition(CheckpointLeaf, "model").concretize()
    model_object = ObjectRef(leaf, {GraphPath(): ObjectId(("model",))})
    model_a = StateRef(model_object, {GraphPath(): "pkl-" + "a" * 64})
    model_b = StateRef(model_object, {GraphPath(): "pkl-" + "b" * 64})
    test_leaf = Definition(CheckpointLeaf, "test").concretize()
    test_object = ObjectRef(test_leaf, {GraphPath(): ObjectId(("test",))})
    test_data = StateRef(test_object, {GraphPath(): "pkl-" + "c" * 64})
    model_path = GraphPath((Parameter("model"),))

    def checkpoint(model, namespace):
        definition = Definition(
            SubstitutionCheckpoint,
            Mat(model),
            test_data=Ref(test_data),
        ).concretize()
        object_ref = ObjectRef(
            definition,
            {GraphPath(): ObjectId((namespace,)), model_path: model.object_id},
        )
        return StateRef(
            object_ref,
            {GraphPath(): "pkl-" + "d" * 64, model_path: model.states[GraphPath()]},
        )

    template = Template.from_value(
        {"model": Par("this.model"), "test_data": Par("this.test_data")}
    )

    bound_a = template.sub(this=checkpoint(model_a, "checkpointa"))
    bound_b = template.sub(this=checkpoint(model_b, "checkpointb"))

    assert bound_a.root["model"] == model_a
    assert bound_b.root["model"] == model_b
    assert bound_a.root["model"] != bound_b.root["model"]
    assert bound_a.root["test_data"] is test_data
    assert bound_b.root["test_data"] is test_data


def test_substitution_paths_fail_without_mutating_the_source_template():
    """Unsupported root paths are deterministic ParameterizationError failures."""
    template = Template.from_value({"value": Par("root.missing")})

    with pytest.raises(ParameterizationError, match="path"):
        template.sub(root={"present": 1})
    assert template.root["value"] == Par("root.missing")


def test_resolve_and_to_definition_preserve_import_free_root_contracts(monkeypatch):
    """Extraction keeps soft call spelling and never thaws an unavailable CDef."""
    soft = Definition(CallShapeTarget, 1, 2, 3, label="written")
    template = Template.from_value(soft)
    assert template.to_definition() is soft
    assert template.to_definition().args == (1, 2, 3)
    assert template.to_definition().kwargs == {"label": "written"}

    bound = Template(
        CallShapeTarget, Par("first"), Par("rest"), Par("tail"), label=Par("label")
    ).sub(first=1, rest=2, tail=3, label="bound").to_definition()
    assert bound.args == (1, 2, 3)
    assert bound.kwargs == {"label": "bound"}

    cdef = ConcreteDefinition._from_bound_record(
        ImportRef("missing.template", "Unavailable"), BoundArguments({"value": "written"})
    )
    cdef_template = Template.from_value(cdef)
    monkeypatch.setattr(ImportRef, "resolve", lambda self: pytest.fail("must not resolve"))
    assert cdef_template.resolve() is cdef
    with pytest.raises(ParameterizationError, match="Definition root"):
        cdef_template.to_definition()
    with pytest.raises(ParameterizationError, match="Definition root"):
        Template.from_value({"value": 1}).to_definition()
    with pytest.raises(UnresolvedDefinitionError):
        Template.from_value(Par("missing")).resolve()


def test_remap_strip_restores_roots_and_rejects_unmatched_prefixes():
    """Namespace prefixing and stripping retain paths while validating the closure."""
    template = Template.from_value({"value": Par("encoder/width.child")})

    assert template.remap(strip="encoder").root["value"] == Par("width.child")
    with pytest.raises(ParameterizationError, match="does not match"):
        template.remap(strip="decoder")


def test_ref_recipe_remains_single_pass_when_explicitly_traversed():
    """Traversal visits pre-existing quoted recipes but not replacements introduced by the call."""
    replacement = Template.from_value({"later": Par("width")})
    recipe = Template.from_value({"value": Par("recipe")})
    template = Template.from_value({"recipe": Ref(recipe)})

    result = template.sub(recipe=replacement, traverse_refs=True)

    assert result.root["recipe"].target.root["value"] == replacement.root
    assert result.names == ()
    assert result.root["recipe"].target.root["value"]["later"] == Par("width")
    assert recipe.root["value"] == Par("recipe")


def test_materialize_links_participate_in_substitution():
    """Owned construction links expose and evaluate their template parameters."""

    child = Definition(SubstitutionLeaf, Par("width"))
    template = Template.from_value({"child": Mat(child)})

    assert template.names == ("width",)
    result = template.sub(width=64)

    assert result.root["child"].kind is EdgeKind.MATERIALIZE
    assert result.root["child"].target.parameters["value"] == 64


def test_traversed_finalized_ref_preserves_portability_and_cdef_authority():
    """Explicit recipe traversal retains finalized links and exact CDef roots."""

    recipe = Template.from_value({"value": Par("width") * 2})
    link = DefLink.finalized(EdgeKind.REF, recipe)
    cdef = ConcreteDefinition._from_bound_record(
        SubstitutionRoot,
        BoundArguments({"child": link}.items()),
    )

    result = Template.from_value(cdef).sub(width=32, traverse_refs=True)

    assert isinstance(result.root, ConcreteDefinition)
    rewritten = result.root.parameters["child"]
    assert rewritten.is_finalized
    assert rewritten.target.root["value"] == 64
    assert Template.from_data(result.to_data()) == result
