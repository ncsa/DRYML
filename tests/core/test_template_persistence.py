"""Persistence coverage for Definition values carried by the Template role."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from dryml.core import Definition, Generator, GeneratorSelector, Object, Par, Repo, Template
from dryml.core.cdef_codec import decode_cdef_graph, encode_cdef_graph
from dryml.core.domains import UniformFromSet
from dryml.core.errors import ParameterizationError
from dryml.core.quoted import QuotedDef
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import canonical_key_bytes


class PortableLeaf(Object):
    """Inert target proving symbolic quotation never constructs a recipe."""

    constructed = 0

    def __init__(self, value):
        type(self).constructed += 1
        self.value = value


class TemplateOwner(Object):
    """Own a single symbolic Definition through the Template receiving role."""

    def __init__(self, recipe: Template):
        self.recipe = recipe


class MappingOwner(Object):
    """Own a flat canonical mapping of independently quoted Definition recipes."""

    def __init__(self, recipes: Mapping[str, Template] | None = None):
        self.recipes = recipes


def _recipe(name: str = "width") -> Definition:
    """Return an inert symbolic Definition without constructing its target."""

    return Definition(PortableLeaf, Par(name) * 2)


def test_template_marker_is_noninstantiable_and_retired_carriers_are_not_public():
    """Template remains annotation vocabulary while runtime carrier APIs are absent."""

    import dryml
    import dryml.core as core

    with pytest.raises(TypeError, match="annotation-only"):
        Template(PortableLeaf, 1)
    assert not hasattr(dryml, "TemplateBundle")
    assert not hasattr(core, "TemplateBundle")
    assert not hasattr(dryml, "TemplateGenerator")
    assert not hasattr(dryml, "TemplateSelector")


def test_template_role_persists_symbolic_definition_as_quoted_data_without_construction(tmp_path):
    """CDef and Store replay deliver the original Definition rather than a carrier."""

    PortableLeaf.constructed = 0
    recipe = _recipe()
    cdef = Definition(TemplateOwner, recipe).concretize()
    restored_cdef = decode_cdef_graph(encode_cdef_graph(cdef))
    link = restored_cdef.parameters["recipe"]
    assert isinstance(link.target, QuotedDef)
    assert link.target.value.names == ("width",)

    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(TemplateOwner(recipe, repo=repo))
    restored = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert isinstance(restored.recipe, Definition)
    assert restored.recipe.names == ("width",)
    assert PortableLeaf.constructed == 0


def test_template_mapping_quotes_each_recipe_in_canonical_key_order():
    """Flat mapping admission uses canonical key bytes, not authored insertion order."""

    first, second = _recipe("first"), _recipe("second")
    cdef = Definition(MappingOwner, {"z": first, "a!": second}).concretize()
    recipes = cdef.parameters["recipes"]

    assert tuple(recipes) == tuple(sorted(("z", "a!"), key=canonical_key_bytes))
    assert all(isinstance(link.target, QuotedDef) for link in recipes.values())
    direct = MappingOwner({"z": first, "a!": second})
    assert tuple(direct.recipes) == tuple(recipes)


@pytest.mark.parametrize("recipes", ({"": _recipe()}, {1: _recipe()}, {"nested": {"x": _recipe()}}))
def test_template_mapping_rejects_invalid_shape_before_owner_effects(recipes):
    """Malformed mappings cannot reach the owning constructor boundary."""

    PortableLeaf.constructed = 0
    with pytest.raises(Exception):
        MappingOwner(recipes)
    assert PortableLeaf.constructed == 0


def test_template_mapping_rejects_more_than_4096_entries_before_construction():
    """The operation-wide mapping limit is enforced before recipe targets run."""

    recipes = {f"metric-{index}": _recipe() for index in range(4_097)}
    with pytest.raises(Exception, match="4096"):
        MappingOwner(recipes)
    assert PortableLeaf.constructed == 0


def test_quoted_symbolic_definition_expression_codec_fails_closed():
    """Quoted Definition expression data round-trips and rejects malformed payloads."""

    from dryml.core.definition_expression_codec import from_data, to_data

    payload = to_data(_recipe())
    restored = from_data(payload)
    assert isinstance(restored, Definition)
    assert restored.names == ("width",)
    payload["version"] = 2
    with pytest.raises(ParameterizationError):
        from_data(payload)


def test_generator_selector_v1_payload_still_round_trips_definition_state():
    """Generator selector persistence retains the established portable payload shape."""

    selector = Generator(
        Definition(PortableLeaf, value=Par("width")),
        {"width": UniformFromSet((2, 4))},
    ).support_selector()
    restored = GeneratorSelector.from_data(selector.to_data())

    assert restored.matches(Definition(PortableLeaf, value=2))
    assert not restored.matches(Definition(PortableLeaf, value=3))
