"""Portable GeneratorSelector persistence contracts."""

from __future__ import annotations

import json
import importlib
from pathlib import Path

import pytest

import dryml
from dryml.core import Definition, Generator, GeneratorSelector, Object
from dryml.core.errors import ParameterizationError
from dryml.core.template import Par


_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "template_v1" / "template_selector.json"


class PortableGeneratorLeaf(Object):
    """Inert Definition target used for portable exact-selector tests."""

    def __init__(self, value):
        self.value = value


class RuntimeDistribution:
    """Valid runtime provider deliberately excluded from portable selector data."""

    def sample(self, rng):
        return 1

    def cardinality(self):
        return 1

    def value_at(self, index):
        return 1

    def contains(self, value):
        return value == 1

    def bounds(self):
        return 1, 1


def _canonical_bytes(value: object) -> bytes:
    """Return the v1 canonical JSON bytes independently of fixture formatting."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def test_generator_selector_preserves_committed_v1_payload_bytes_and_tag():
    """V1 selector data decodes directly to Definition state and re-encodes exactly."""

    fixture = json.loads(_FIXTURE.read_text(encoding="ascii"))
    selector = GeneratorSelector.from_data(fixture)

    assert selector._generator.definition.names == ("width",)
    assert selector.to_data()["kind"] == "template-selector"
    assert _canonical_bytes(selector.to_data()) == _canonical_bytes(fixture)


def test_generator_selector_rejects_runtime_provider_persistence():
    """Only immutable built-in distributions are portable selector state."""

    selector = Generator(
        Definition(PortableGeneratorLeaf, Par("width")), {"width": RuntimeDistribution()}
    ).support_selector()

    with pytest.raises(ParameterizationError, match="nonportable"):
        selector.to_data()


def test_retired_template_generation_exports_and_module_are_absent():
    """The renamed API provides no beta-name or module compatibility aliases."""

    assert not hasattr(dryml, "TemplateGenerator")
    assert not hasattr(dryml, "TemplateSelector")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dryml.core.template_selector")


def test_node_generator_selector_round_trip_keeps_fixed_facets_and_support():
    """New selector persistence retains Node constraints with exact witness proof."""

    from dryml.core import ObjectId, UniformFromSet

    source = Definition(PortableGeneratorLeaf, Par("width")).with_identity(ObjectId(("selector",)))
    selector = Generator(source, {"width": UniformFromSet((2, 4))}).support_selector()
    data = selector.to_data()
    restored = GeneratorSelector.from_data(data)
    candidate = source.sub(width=2)
    assert restored._generator.definition.graph_equal(source)
    assert restored.matches(candidate)
    assert not restored.matches(candidate.without_identity())
    assert restored.to_data() == data


def test_closed_node_query_predicate_codec_preserves_boolean_support_and_facets():
    """Portable predicates use the Node family and reject runtime providers."""
    from dryml.core import ObjectId, QueryPredicate, UniformFromSet
    source = Definition(PortableGeneratorLeaf, Par("width")).with_identity(ObjectId(("query",)))
    predicate = QueryPredicate.support(Generator(source, {"width": UniformFromSet((2, 4))})) & ~QueryPredicate.absent("identity")
    data = json.loads(json.dumps(predicate.to_data()))
    assert data["schema"] == "dryml-node-query-predicate" and data["version"] == 1
    restored = QueryPredicate.from_data(data)
    assert restored.matches(source.sub(width=2))
    assert not restored.matches(source.sub(width=True))
    assert not restored.matches(source.sub(width=3))
    assert not restored.matches(source.sub(width=2).without_identity())
    assert restored.to_data() == data
    runtime = QueryPredicate.support(Generator(Definition(PortableGeneratorLeaf, Par("width")), {"width": RuntimeDistribution()}))
    with pytest.raises(ParameterizationError, match="nonportable"):
        runtime.to_data()
    with pytest.raises((ValueError, ParameterizationError)):
        QueryPredicate.from_data({**data, "version": True})
    child = Definition(PortableGeneratorLeaf, 2)
    quoted = QueryPredicate(Definition(child=child, recipe=child.quote()))
    decoded = QueryPredicate.from_data(json.loads(json.dumps(quoted.to_data())))
    assert decoded.value.kwargs["child"] is decoded.value.kwargs["recipe"].value
    assert decoded.value.kwargs["recipe"].value.args == (2,)
    assert decoded.matches(quoted.value.copy_graph())
