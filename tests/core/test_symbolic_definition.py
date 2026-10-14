"""Focused Definition-native symbolic authoring coverage."""

from __future__ import annotations

import pytest

from dryml.core import Definition, Par, Ref
from dryml.core.cdef_graph import EdgeKind
from dryml.core.domains import UniformFromSet
from dryml.core.errors import (
    ParameterizationError,
    ParameterizationLimitError,
    UnresolvedDefinitionError,
    UnsupportedGeneratorVerificationError,
)
from dryml.core.symbol import ImportRef


class SymbolicLeaf:
    """Constructor probe that must remain uncalled by symbolic operations."""

    calls = 0

    def __init__(self, width, *, label="default"):
        type(self).calls += 1
        self.width = width
        self.label = label


class SymbolicEnvelope:
    """Structural target containing materializing and reference-like values."""

    def __init__(self, *, child=None, left=None, right=None, members=None, width=None, quoted=None):
        self.child = child
        self.left = left
        self.right = right
        self.members = members
        self.width = width
        self.quoted = quoted


def test_definition_substitution_is_immutable_partial_and_target_inert():
    """Definition owns ordered discovery and single-pass static substitution."""

    SymbolicLeaf.calls = 0
    definition = Definition(
        SymbolicLeaf,
        Par("width"),
        label=Par("label") * 2,
    )

    partial = definition.sub(width=32)
    resolved = partial.sub(label=3)

    assert definition.names == ("width", "label")
    assert not definition.is_resolved
    assert isinstance(partial, Definition)
    assert partial.names == ("label",)
    assert not partial.is_resolved
    assert isinstance(resolved, Definition)
    assert resolved.is_resolved
    assert resolved.parameters == {"width": 32, "label": 6}
    assert definition.parameters["width"] == Par("width")
    assert SymbolicLeaf.calls == 0


def test_definition_substitution_preserves_definition_values_inside_binding_containers():
    """Mapping-like definitions stay graph values rather than ordinary dictionaries."""

    soft = Definition(SymbolicLeaf, width=7)
    concrete = soft.concretize()
    SymbolicLeaf.calls = 0
    template = Definition(SymbolicEnvelope, child=Par("child"), members=Par("members"))

    for value in (soft, concrete):
        bound = template.sub(child=value, members={"nested": (value,)})
        assert bound.child is value
        assert bound.members["nested"][0] is value
        assert bound.concretize() == Definition(
            SymbolicEnvelope, child=value, members={"nested": (value,)},
        ).concretize()

    assert SymbolicLeaf.calls == 0


def test_shared_authoring_recognition_handles_expressions_authority_and_quote_barriers():
    """Authoring detection is effect-free and distinct from active-expression admission."""

    from dryml.core import F, Mat
    from dryml.core.template import _contains_template_value, _is_definition_value

    definition = Definition(SymbolicLeaf, width=7)
    concrete = definition.concretize()
    SymbolicLeaf.calls = 0

    assert _is_definition_value(Par("width") * 2)
    assert _is_definition_value(definition)
    assert not _is_definition_value(concrete)
    assert _is_definition_value(concrete, include_authority=True)
    assert _contains_template_value({"values": (F("builtins:tuple", Par("width") * 2),)})
    assert _contains_template_value(Mat(definition))
    assert not _contains_template_value(Ref(definition))
    assert not _contains_template_value(definition.quote())
    assert not _contains_template_value(concrete)
    cyclic = []
    cyclic.append(cyclic)
    assert not _contains_template_value(cyclic)
    cyclic.append(Par("width"))
    assert _contains_template_value(cyclic)
    assert SymbolicLeaf.calls == 0


def test_definition_rewrites_owned_structure_and_keeps_quotation_boundaries():
    """Shared nodes rewrite once while QuotedDef and Ref stay opaque by default."""

    shared = Definition(SymbolicLeaf, Par("width"))
    quoted = Definition(SymbolicLeaf, Par("quoted")).quote()
    referenced = Definition(SymbolicLeaf, Par("reference"))
    definition = Definition(
        SymbolicEnvelope,
        child=Ref(referenced),
        left=shared,
        right=shared,
        width=Par("outer"),
        quoted=quoted,
    )

    default = definition.sub(width=64, outer=1)
    traversed = definition.sub(reference=2, width=64, outer=1, traverse_refs=True)

    assert definition.names == ("width", "outer")
    assert default.parameters["child"].target is referenced
    assert default.parameters["quoted"] is quoted
    assert default.parameters["left"] is default.parameters["right"]
    assert default.parameters["left"].parameters["width"] == 64
    assert traversed.parameters["child"].kind is EdgeKind.REF
    assert not traversed.parameters["child"].is_finalized
    assert traversed.parameters["child"].target.parameters["width"] == 2
    assert referenced.parameters["width"] == Par("reference")


def test_definition_remap_projection_and_set_admission_are_structural_only():
    """Remapping and loose projection retain existing selector behavior without targets."""

    definition = Definition(SymbolicLeaf, Par("encoder/width"), label="fixed")

    remapped = definition.remap(prefix="model", strip="encoder")
    selector = definition.loose_selector()

    assert remapped.names == ("model/width",)
    assert selector.matches(Definition(SymbolicLeaf, 64, label="fixed"))
    assert not selector.matches(Definition(SymbolicLeaf, 64, label="other"))
    assert Definition(SymbolicEnvelope, members={1, 2}).parameters["members"] == {1, 2}
    with pytest.raises(ParameterizationError, match="sets"):
        Definition(SymbolicEnvelope, members={Par("width")})


def test_definition_rejects_distributions_at_all_public_value_boundaries():
    """Distribution policy cannot enter Definition-owned construction structure."""

    provider = UniformFromSet((32, 64))
    definition = Definition(SymbolicLeaf, Par("width"))

    with pytest.raises(ParameterizationError, match="Distribution"):
        Definition(SymbolicLeaf, {"provider": provider})
    with pytest.raises(ParameterizationError, match="Distribution"):
        definition.with_kwarg("label", {"provider": provider})
    with pytest.raises(ParameterizationError, match="Distribution"):
        definition.sub(width={"provider": provider})


def test_definition_substitution_validates_bindings_once_not_each_parent(monkeypatch):
    """Trusted symbolic rebuilding does not recursively rescan every ancestor."""

    import dryml.core.template as template_module

    definition = Definition(SymbolicLeaf, Par("width"))
    for _ in range(64):
        definition = Definition(SymbolicEnvelope, child=definition)
    calls = 0
    original = template_module._validate_distribution_free

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(template_module, "_validate_distribution_free", counted)

    assert definition.sub(width=1).is_resolved
    assert calls == 1


def test_definition_substitution_rejects_active_expressions_inside_sets():
    """One final validation rejects order-sensitive symbolic binding output."""

    definition = Definition(SymbolicLeaf, Par("width"))

    with pytest.raises(ParameterizationError, match="sets cannot contain"):
        definition.sub(width={Par("nested")})

    hidden = Definition(
        SymbolicEnvelope,
        child=Ref(Definition(SymbolicLeaf, Par("width"))),
    )
    with pytest.raises(ParameterizationError, match="sets cannot contain"):
        hidden.sub(
            width={Par("nested")},
            traverse_refs=True,
        )

    inverse = Definition(SymbolicLeaf, Par("width"))
    with pytest.raises(ParameterizationError, match="sets cannot contain"):
        inverse.sub(
            width={Ref(Definition(SymbolicLeaf, Par("nested")))},
            traverse_refs=True,
        )


def test_unresolved_positional_definition_hasattr_obeys_attribute_protocol():
    """Protocol probes do not leak semantic-name errors through ``hasattr``."""

    definition = Definition(ImportRef("builtins", "dict"), 1)

    with pytest.raises(TypeError, match="positional names are unknown"):
        _ = definition.parameters
    assert not hasattr(definition, "sample")


def test_parameterization_error_hierarchy_replaces_template_errors_without_aliases():
    """The public error surface uses only the Definition parameterization names."""

    from dryml.core import errors

    assert issubclass(UnresolvedDefinitionError, ParameterizationError)
    assert issubclass(ParameterizationLimitError, ParameterizationError)
    assert issubclass(UnsupportedGeneratorVerificationError, ParameterizationError)
    assert not hasattr(errors, "TemplateError")
