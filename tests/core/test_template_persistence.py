"""Closed persistence and Ref-delivery contracts for definition templates."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from dryml.core import ConcreteDefinition, Definition, Object, ObjectRef, Ref, Repo, StateRef, Template
from dryml.core.cdef_codec import CDefGraphCodecError, decode_cdef_graph, encode_cdef_graph
from dryml.core.errors import TemplateError
from dryml.core.links import DefLink
from dryml.core.repo_definition import RepoDefinition
from dryml.core.store.dir import DirStore
from dryml.core.template import Par
from dryml.core.template_selector import TemplateGenerator, TemplateSelector
from dryml.core.domains import UniformFromSet


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "template_v1"


class PortableLeaf(Object):
    """Importable inert leaf used to prove template decoding stays lightweight."""

    constructed = 0

    def __init__(self, value):
        type(self).constructed += 1
        self.value = value


class TemplateOwner(Object):
    """Concrete owner that explicitly admits an unresolved carried recipe."""

    def __init__(self, recipe: Ref[Template]):
        self.recipe = recipe


class UndeclaredOwner(Object):
    """Owner without a role declaration used to reject quotation bypasses."""

    def __init__(self, recipe):
        self.recipe = recipe


class MaterializingOwner(Object):
    """Materializing owner that must reject unresolved recipe data."""

    def __init__(self, recipe: Ref[Definition]):
        self.recipe = recipe


def _recipe() -> Template:
    """Return a portable unresolved expression whose target need not exist."""

    return Template(PortableLeaf, Par("width") * 2)


def test_template_round_trip_preserves_expression_and_never_constructs_target():
    """Portable recipe decoding remains inert and preserves its unresolved root."""

    PortableLeaf.constructed = 0
    recipe = _recipe()

    restored = Template.from_data(recipe.to_data())

    assert restored.to_data() == recipe.to_data()
    assert restored.names == ("width",)
    assert PortableLeaf.constructed == 0


def test_synthetic_v1_fixture_decodes_without_resolving_its_target():
    """The checked-in codec fixture remains portable independent of Store data."""

    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="ascii"))
    payload = json.loads((FIXTURE_ROOT / "template.json").read_text(encoding="ascii"))

    decoded = Template.from_data(payload)

    assert manifest["codec_schema"] == payload["schema"]
    assert manifest["codec_version"] == payload["version"]
    assert decoded.names == ("width",)

    selector_payload = json.loads((FIXTURE_ROOT / "template_selector.json").read_text(encoding="ascii"))
    selector = TemplateSelector.from_data(selector_payload)
    assert selector._generator.template.names == ("width",)


def test_synthetic_v1_fixtures_preserve_ref_owner_and_reference_meaning():
    """Retained synthetic payloads decode inertly with their exact root meaning."""

    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="ascii"))
    expected = {
        "ref_owner_cdef.json": ConcreteDefinition,
        "object_ref.json": ObjectRef,
        "state_ref.json": StateRef,
    }
    for name, root_type in expected.items():
        payload = json.loads((FIXTURE_ROOT / name).read_text(encoding="ascii"))
        restored = Template.from_data(payload)
        assert restored.to_data() == payload
        assert isinstance(restored.root, root_type)
        assert manifest["fixtures"][name]["root_type"] == root_type.__name__

    owner = Template.from_data(
        json.loads((FIXTURE_ROOT / "ref_owner_cdef.json").read_text(encoding="ascii"))
    ).root
    assert isinstance(owner.parameters["recipe"], DefLink)
    assert owner.parameters["recipe"].target.names == ("width",)

    repo_data = json.loads((FIXTURE_ROOT / "repo_definition.json").read_text(encoding="ascii"))
    assert RepoDefinition.from_data(repo_data).to_data() == repo_data


def test_declared_ref_template_owner_persists_opaque_unresolved_recipe():
    """Only an explicit Ref[Template] owner retains unresolved recipe data."""

    PortableLeaf.constructed = 0
    owner = Definition(TemplateOwner, _recipe()).concretize()
    restored = decode_cdef_graph(encode_cdef_graph(owner))

    link = restored.parameters["recipe"]
    assert isinstance(link, DefLink)
    assert isinstance(link.target, Template)
    assert link.target.names == ("width",)
    assert PortableLeaf.constructed == 0


def test_declared_ref_template_owner_restores_through_a_store(tmp_path):
    """Structural Store restoration delivers a carried recipe without building it."""

    PortableLeaf.constructed = 0
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    owner = TemplateOwner(_recipe(), repo=repo)

    saved = repo.save_object(owner)
    restored = Repo(DirStore(store.base_dir)).load_object(saved.definition)

    assert isinstance(restored.recipe, Template)
    assert restored.recipe.names == ("width",)
    assert PortableLeaf.constructed == 0


@pytest.mark.parametrize("owner", (UndeclaredOwner, MaterializingOwner))
def test_unresolved_template_requires_exact_ref_template_admission(owner):
    """Raw and non-Template roles cannot bypass concrete-owner admission."""

    with pytest.raises((TypeError, TemplateError)):
        Definition(owner, _recipe()).concretize()


def test_finalized_template_link_is_rechecked_at_fresh_declared_boundary():
    """A manually finalized link is not a generic Ref-wrapper admission bypass."""

    link = DefLink.finalized(Ref.kind, _recipe())

    with pytest.raises(Exception):
        Definition(UndeclaredOwner, link).concretize()


def test_template_codec_rejects_unknown_tags_and_duplicate_labels():
    """Malformed closed payloads fail before exposing a recipe."""

    data = _recipe().to_data()
    unknown = deepcopy(data)
    unknown["version"] = 2
    duplicate = deepcopy(data)
    duplicate["nodes"].append(deepcopy(duplicate["nodes"][0]))

    with pytest.raises(TemplateError):
        Template.from_data(unknown)
    with pytest.raises(TemplateError):
        Template.from_data(duplicate)


def test_cdef_codec_rejects_template_payload_outside_ref_link():
    """The additive CDef leaf cannot turn a Template into ordinary atom data."""

    owner = Definition(TemplateOwner, _recipe()).concretize()
    encoded = encode_cdef_graph(owner)
    encoded["nodes"][0]["parameters"]["items"][0][1] = {
        "kind": "template", "value": _recipe().to_data(),
    }

    with pytest.raises(CDefGraphCodecError):
        decode_cdef_graph(encoded)


def test_exact_selector_round_trip_uses_only_builtin_domain_data():
    """Exact support selectors retain built-in choices without providers/code."""

    selector = TemplateGenerator(
        Template(PortableLeaf, value=Par("width")), sub_dict={"width": UniformFromSet((2, 4))}
    ).support_selector()

    restored = TemplateSelector.from_data(selector.to_data())

    assert restored.matches(Definition(PortableLeaf, value=2))
    assert not restored.matches(Definition(PortableLeaf, value=3))


def test_selector_codec_rejects_noncanonical_node_order():
    """Selector decoding applies the same canonical graph check as Template data."""

    selector = TemplateGenerator(
        Template(PortableLeaf, value=Par("width")), sub_dict={"width": UniformFromSet((2, 4))}
    ).support_selector()
    payload = selector.to_data()
    payload["nodes"].reverse()

    with pytest.raises(TemplateError, match="canonical"):
        TemplateSelector.from_data(payload)


def test_template_codec_rejects_actual_private_cycles_before_back_references():
    """A malformed private root cannot be encoded as a graph alias cycle."""

    root = []
    root.append(root)

    with pytest.raises(TemplateError, match="cycle"):
        Template._from_root(root).to_data()


def test_template_name_limits_and_set_member_boundary():
    """Public construction enforces portable root limits and unordered-value rules."""

    component = "a" * 64
    assert Par(component).name == component
    with pytest.raises(TemplateError):
        Par("a" * 65)
    assert Par("/".join(["a"] * 16)).name.count("/") == 15
    with pytest.raises(TemplateError):
        Par("/".join(["a"] * 17))
    assert Par("/".join(("a" * 64, "b" * 64, "c" * 64, "d" * 61))).name
    with pytest.raises(TemplateError):
        Par("/".join(("a" * 64, "b" * 64, "c" * 64, "d" * 62)))
    qualified = "/".join(("a",) * 15 + ("width",))
    assert Template(PortableLeaf, Par(qualified)).sub(namespace=("a",) * 15, width=1).is_resolved
    with pytest.raises(TemplateError):
        Template(PortableLeaf, Par(qualified)).sub(namespace=("a",) * 16, width=1)
    with pytest.raises(TemplateError, match="set members"):
        Template.from_value({Par("width")})


def test_template_codec_rejects_expression_set_members():
    """Parsed template data cannot place an expression in an unordered set."""

    payload = Template.from_value({"literal"}).to_data()
    record = next(record for record in payload["nodes"] if record["value"]["tag"] == "set")
    record["value"]["items"] = [{"tag": "ref", "label": "n-expression"}]
    payload["nodes"].append({
        "label": "n-expression",
        "value": {"tag": "par", "name": "width", "path": {"schema_version": 3, "segments": []}},
    })

    with pytest.raises(TemplateError, match="set members"):
        Template.from_data(payload)


def test_template_equality_and_hash_preserve_alias_topology():
    """Shared and independent equal recipes remain distinct across round-trips."""

    leaf = Definition(PortableLeaf, 1)
    shared = Template.from_value([leaf, leaf])
    independent = Template.from_value([Definition(PortableLeaf, 1), Definition(PortableLeaf, 1)])

    assert shared != independent
    assert shared.stable_hash() != independent.stable_hash()
    restored_shared = Template.from_data(shared.to_data())
    restored_independent = Template.from_data(independent.to_data())
    assert restored_shared.root[0] is restored_shared.root[1]
    assert restored_independent.root[0] is not restored_independent.root[1]
