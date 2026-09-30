"""Focused admission and persistence proofs for Definition receiving roles."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from dryml.core import Definition, Object, Par, Ref, Repo, Template
from dryml.core.cdef_codec import CDefGraphCodecError, decode_cdef_graph, encode_cdef_graph
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.definition_expression_codec import from_data as expression_from_data
from dryml.core.definition_expression_codec import to_data as expression_to_data
from dryml.core.links import DefLink
from dryml.core.quoted import QuotedDef
from dryml.core.repo_definition import _SelectorEncoder, _selector_from_data, _validate_selector
from dryml.core.selector import Selector
from dryml.core.signatures import Mat, SignatureError, compile_signature
from dryml.core.template import repeat
from dryml.core.utils.graph.path import canonical_key_bytes


class RoleLeaf(Object):
    """Leaf whose counter proves that symbolic role checks do not construct it."""

    constructions = 0

    def __init__(self, width):
        type(self).constructions += 1
        self.width = width


class TemplateHolder(Object):
    """Owner whose recipe is inert Definition quotation data."""

    constructions = 0

    def __init__(self, recipe: Template):
        type(self).constructions += 1
        self.recipe = recipe


class PlainHolder(Object):
    """Owner whose ordinary slot remains materializing structure."""

    constructions = 0

    def __init__(self, recipe):
        type(self).constructions += 1
        self.recipe = recipe


class MappingHolder(Object):
    """Owner exercising the one supported flat Template mapping grammar."""

    def __init__(self, recipes: Mapping[str, Template] | None = None):
        self.recipes = recipes


class NoEffectHolder(Object):
    """Owner used to prove rejection precedes cache seeding and construction."""

    constructions = 0

    def __init__(self, live, recipe):
        type(self).constructions += 1
        self.live = live
        self.recipe = recipe


def _plan(annotation):
    """Compile one single-value role boundary for concise matrix assertions."""

    def target(value):
        return value

    target.__annotations__["value"] = annotation
    return compile_signature(target)


def test_template_marker_compiles_as_fixed_definition_role_and_rejects_values():
    """Bare and subscribed Template spellings are the same non-instantiable role."""

    assert _plan(Template).slots["value"] == _plan(Template[Definition]).slots["value"]
    assert _plan(Template | None).slots["value"].nullable
    with pytest.raises(TypeError, match="cannot be instantiated"):
        Template(Definition(RoleLeaf, 1))
    with pytest.raises(TypeError, match="exact Definition"):
        Template[Object]
    with pytest.raises(SignatureError):
        _plan(Ref[Template])

    def variadic(*recipes: Template):
        return recipes

    symbolic = Definition(RoleLeaf, Par("width"))
    assert compile_signature(variadic).prepare_args(
        (symbolic, Ref(symbolic)), {}
    ).deliver_args()[0] == (symbolic, symbolic)


def test_ref_template_and_mat_roles_admit_symbolic_definitions_distinctly():
    """Ref rejects active expressions, Template quotes them, and Mat fails pre-effect."""

    symbolic = Definition(RoleLeaf, Par("width"))
    incomplete = Definition(RoleLeaf)
    RoleLeaf.constructions = 0

    assert _plan(Ref[Definition]).prepare_args((incomplete,), {}).deliver_args()[0][0] is incomplete
    with pytest.raises(SignatureError, match="symbolically resolved"):
        _plan(Ref[Definition]).prepare_args((symbolic,), {})
    delivered = _plan(Template).prepare_args((Ref(symbolic),), {}).deliver_args()[0][0]
    assert delivered is symbolic
    with pytest.raises(SignatureError, match="conflicts"):
        _plan(Template).prepare_args((Mat(symbolic),), {})
    with pytest.raises(Exception, match="active symbolic expression"):
        _plan(Mat[Definition]).prepare_args((symbolic,), {}, repo=Repo()).deliver_args()
    assert RoleLeaf.constructions == 0


def test_auto_ref_requires_explicit_intent_for_soft_definitions():
    """AutoRef keeps its bare-value contract while honoring Ref assertions."""

    from dryml.core import AutoRef

    resolved = Definition(RoleLeaf, 1)
    with pytest.raises(SignatureError, match="requested authority is unavailable"):
        _plan(Ref[AutoRef]).prepare_args((resolved,), {})

    delivered = _plan(Ref[AutoRef]).prepare_args((Ref(resolved),), {}).deliver_args()[0][0]
    assert delivered is resolved


def test_template_barrier_allows_direct_and_outer_materialization_without_recipe_effects():
    """Activated Template slots stop materialization admission but preserve authored symbols."""

    symbolic = Definition(RoleLeaf, Par("width"))
    holder = Definition(TemplateHolder, symbolic)
    TemplateHolder.constructions = PlainHolder.constructions = RoleLeaf.constructions = 0

    direct = TemplateHolder(symbolic)
    outer = Repo().materialize_boundary((holder,))[0]

    assert direct.recipe is symbolic
    assert outer.recipe == symbolic
    assert holder.names == ("width",)
    assert not holder.is_resolved
    assert TemplateHolder.constructions == 2
    assert RoleLeaf.constructions == 0
    with pytest.raises(Exception, match="active symbolic expression"):
        PlainHolder(symbolic)
    assert PlainHolder.constructions == 0
    with pytest.raises(SignatureError, match="symbolically resolved"):
        _plan(Ref[Definition]).prepare_args((holder,), {})


def test_template_values_persist_as_quoted_definitions_and_replay_without_signatures(monkeypatch):
    """Template delivery uses Ref-to-QuotedDef persistence and codec replay stays inert."""

    symbolic = Definition(RoleLeaf, Par("width") * 2)
    cdef = Definition(TemplateHolder, symbolic).concretize()
    marker = cdef.parameters["recipe"]
    assert isinstance(marker, DefLink)
    assert isinstance(marker.target, QuotedDef)
    assert ConcreteDefinitionGraph.from_root(cdef).edges() == ()

    encoded = encode_cdef_graph(cdef)
    restored = decode_cdef_graph(encoded)
    assert restored.stable_hash() == cdef.stable_hash()
    monkeypatch.setattr(
        "dryml.core.signatures.compile_signature",
        lambda *args, **kwargs: pytest.fail("persisted Template delivery recompiled a signature"),
    )
    rebuilt = Repo().load_or_build(restored)

    assert isinstance(rebuilt.recipe, Definition)
    assert rebuilt.recipe.names == ("width",)
    malformed = encode_cdef_graph(cdef)
    malformed["nodes"][0]["parameters"]["items"][0][1]["target"]["value"]["version"] = 2
    with pytest.raises(CDefGraphCodecError, match="quoted Definition"):
        decode_cdef_graph(malformed)


def test_definition_expression_codec_preserves_aliases_and_rejects_bad_budgets():
    """Quoted Definition expression data round-trips arithmetic, repetition, and aliases data-only."""

    shared = Definition(RoleLeaf, Par("width"))
    data = expression_to_data([shared, shared, Par("scale") * 2, repeat([shared], 2)])
    restored = expression_from_data(data)

    assert restored[0] is restored[1]
    assert restored[2].operation == "mul"
    assert restored[3].shared is False
    malformed = dict(data)
    malformed["version"] = 2
    with pytest.raises(Exception, match="schema or version"):
        expression_from_data(malformed)


def test_repo_definition_selector_data_round_trips_symbolic_quoted_definitions():
    """Repo-definition selectors retain symbolic QuotedDef payloads as data-only records."""

    quoted = Definition(RoleLeaf, Par("width") * 2).quote()
    selector = Selector(Definition(PlainHolder, recipe=quoted))
    data = _SelectorEncoder().selector(selector, "$.selector")

    _validate_selector(data, "$.selector")
    restored = _selector_from_data(data)
    restored_quote = restored.root.kwargs["recipe"]
    assert isinstance(restored_quote, QuotedDef)
    assert restored_quote.value.names == ("width",)


def test_flat_template_mapping_orders_and_admits_each_value_independently():
    """Only Mapping[str, Template] is admitted, in canonical key order, as quotation data."""

    first = Definition(RoleLeaf, Par("first"))
    second = Definition(RoleLeaf, Par("second"))
    cdef = Definition(MappingHolder, {"z": Ref(first), "a!": second}).concretize()

    assert tuple(cdef.parameters["recipes"]) == tuple(
        sorted(("z", "a!"), key=canonical_key_bytes)
    )
    assert all(isinstance(item.target, QuotedDef) for item in cdef.parameters["recipes"].values())
    assert MappingHolder({"z": first, "a!": second}).recipes.keys() == {"a!", "z"}
    assert MappingHolder().recipes is None
    assert _plan(Template | None).prepare_args((None,), {}).deliver_args()[0][0] is None

    with pytest.raises(SignatureError, match="mapping keys"):
        Definition(MappingHolder, {1: first}).concretize()
    with pytest.raises(SignatureError):
        _plan(list[Template])


def test_direct_failure_precedes_live_cache_seeding_and_owner_construction(monkeypatch):
    """A later symbolic materializing value cannot seed cache entries from earlier live inputs."""

    repo = Repo()
    live = RoleLeaf(1, repo=repo)
    cache_calls = []
    monkeypatch.setattr(repo, "cache_weak", lambda value: cache_calls.append(value))
    NoEffectHolder.constructions = 0

    with pytest.raises(Exception, match="active symbolic expression"):
        NoEffectHolder(live, Definition(RoleLeaf, Par("width")), repo=repo)

    assert cache_calls == []
    assert NoEffectHolder.constructions == 0
