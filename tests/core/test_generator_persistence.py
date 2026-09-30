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
