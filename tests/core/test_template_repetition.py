"""Focused arithmetic evaluation and identity-aware template repetition tests."""

from __future__ import annotations

import pytest

from dryml.core import Definition, F, Mat, Object
from dryml.core.cdef_identity import cdef_node_key
from dryml.core.cdef_graph import EdgeKind
from dryml.core.errors import ParameterizationError, ParameterizationLimitError
from dryml.core.links import DefLink
from dryml.core.template import Par, Shared, Template, repeat


class RepeatLeaf(Object):
    """Minimal construction node used to verify copy topology."""

    def __init__(self, value):
        self.value = value


class RepeatParent(Object):
    """Minimal parent node retaining ordered repeated children."""

    def __init__(self, children):
        self.children = children


class RepeatFactoryTarget:
    """Inert factory target proving template expansion does not build values."""

    calls = 0

    def __init__(self, value):
        type(self).calls += 1
        self.value = value


def _leaf(value="value"):
    return Definition(RepeatLeaf, value).concretize()


def test_substitution_evaluates_arithmetic_post_order_without_coercion():
    """Partial operands persist until all bindings are exact finite built-in numbers."""

    template = Template.from_value(
        {
            "product": Par("width") * Par("scale"),
            "division": Par("width") / 2,
            "floor": -5 // Par("scale"),
        }
    )

    partial = template.sub(width=65)
    assert partial.root["product"].left == 65
    assert partial.root["product"].right == Par("scale")

    resolved = partial.sub(scale=2)
    assert resolved.root == {"product": 130, "division": 32.5, "floor": -3}
    assert type(resolved.root["division"]) is float

    with pytest.raises(ParameterizationError, match="division"):
        template.sub(width=1, scale=0)
    with pytest.raises(ParameterizationError, match="numbers"):
        template.sub(width=True, scale=1)
    assert template.root["product"] == Par("width") * Par("scale")


def test_arithmetic_enforces_the_integer_bit_length_limit():
    """Operands and results may use at most 1,024-bit exact integers."""

    at_limit = 1 << 1023
    assert Template.from_value(Par("value") * 1).sub(value=at_limit).root == at_limit

    with pytest.raises(ParameterizationLimitError, match="bit-length"):
        Template.from_value(Par("value") * 1).sub(value=1 << 1024)
    with pytest.raises(ParameterizationLimitError, match="bit-length"):
        Template.from_value(Par("value") * 2).sub(value=at_limit)


def test_independent_repetition_preserves_order_aliases_and_parameter_linkage():
    """Each copy receives fresh construction nodes but keeps its internal aliases."""

    block = Definition(RepeatLeaf, Par("width"))
    template = Template.from_value(
        {
            "list": repeat([block, block], Par("count")),
            "tuple": repeat((block,), Par("count")),
        }
    )

    result = template.sub(width=64, count=2)
    listed = result.root["list"]
    tupled = result.root["tuple"]

    assert isinstance(listed, tuple)
    assert type(listed).__name__ == "FrozenList"
    assert isinstance(tupled, tuple)
    assert type(tupled).__name__ == "FrozenTuple"
    assert [item.parameters["value"] for item in listed] == [64, 64, 64, 64]
    assert listed[0] is listed[1]
    assert listed[2] is listed[3]
    assert listed[0] is not listed[2]
    assert tupled[0] is not tupled[1]
    concrete = Definition(RepeatParent, listed).concretize().parameters["children"]
    assert cdef_node_key(concrete[0]) is cdef_node_key(concrete[1])
    assert cdef_node_key(concrete[0]) is not cdef_node_key(concrete[2])


def test_repeat_zero_one_shared_and_cross_boundary_identity_contracts():
    """Independent copies freshen even once while Shared retains all original nodes."""

    block = _leaf()
    independent = Template.from_value(
        {"extra": block, "layers": repeat([block], Par("count"))}
    )
    one = independent.sub(count=1).root
    two = independent.sub(count=2).root

    assert one["layers"] == (one["layers"][0],)
    assert cdef_node_key(one["extra"]) is cdef_node_key(block)
    assert cdef_node_key(one["layers"][0]) is not cdef_node_key(block)
    assert len({id(two["extra"]), id(two["layers"][0]), id(two["layers"][1])}) == 3
    assert independent.sub(count=0).root["layers"] == ()

    shared = Template.from_value(
        {"extra": block, "layers": repeat([block], Shared(2))}
    ).sub()
    assert shared.root["extra"] is block
    assert shared.root["layers"][0] is block
    assert shared.root["layers"][1] is block


def test_repetition_keeps_explicit_ref_edges_and_factories_inert():
    """External reference links stay atomic and factory recipes are never built."""

    external = _leaf()
    reference = DefLink.finalized(EdgeKind.REF, external)
    factory = F(RepeatFactoryTarget, Par("width"))
    template = Template.from_value(
        {"refs": repeat([reference], 2), "factories": repeat([factory], 2)}
    )

    RepeatFactoryTarget.calls = 0
    result = template.sub(width=64).root

    assert result["refs"][0] is reference
    assert result["refs"][1] is reference
    assert result["refs"][0].target is external
    assert result["factories"][0].args[0] == 64
    assert result["factories"][0] is not result["factories"][1]
    assert RepeatFactoryTarget.calls == 0


def test_repetition_freshens_materialized_construction_links():
    """Independent groups copy owned Mat targets while retaining linked values."""

    child = Definition(RepeatLeaf, Par("width"))
    template = Template.from_value(repeat([Mat(child)], 2))

    result = template.sub(width=64).root

    assert all(link.kind is EdgeKind.MATERIALIZE for link in result)
    assert all(link.target.parameters["value"] == 64 for link in result)
    assert result[0] is not result[1]
    assert result[0].target is not result[1].target


def test_repetition_limits_are_cumulative_and_leave_sources_unchanged():
    """Nested expansion uses one depth and output budget rather than resetting per copy."""

    exact = Template.from_value(repeat([repeat(["value"], 255)], 256)).sub()
    assert len(exact.root) == 256
    assert all(len(group) == 255 for group in exact.root)

    over = Template.from_value(repeat([repeat(["value"], 256)], 256))
    with pytest.raises(ParameterizationLimitError, match="expansion"):
        over.sub()
    assert isinstance(over.root, type(repeat(["value"], 1)))

    depth_exact = "value"
    for _ in range(128):
        depth_exact = repeat([depth_exact], 1)
    resolved_depth = Template.from_value(depth_exact).sub().root
    for _ in range(128):
        resolved_depth = resolved_depth[0]
    assert resolved_depth == "value"

    depth_over = depth_exact
    depth_over = repeat([depth_over], 1)
    with pytest.raises(ParameterizationLimitError, match="depth"):
        Template.from_value(depth_over).sub()

    with pytest.raises(ParameterizationLimitError, match="count"):
        Template.from_value(repeat(["value"], 1025)).sub()
    for count in (True, 1.0, -1):
        with pytest.raises(ParameterizationError):
            repeat(["value"], count)
