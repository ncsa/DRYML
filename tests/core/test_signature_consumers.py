"""Consumer integration coverage for the shared core signature authority."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest

from dryml.core import ConcreteDefinition, Definition, Object, Repo, Selector
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_codec import decode_cdef_graph, encode_cdef_graph
from dryml.core.cdef_graph import ConcreteDefinitionGraph, EdgeKind
from dryml.core.factory import FactorySpec
from dryml.core.freeze import FrozenList
from dryml.core.links import DefLink
from dryml.core.params import Present
from dryml.core.quoted import QuotedDef, SelectorSpec
from dryml.core.signatures import Mat, Ref, SignatureError, compile_signature


class ConsumerLeaf(Object):
    """Small construction fixture used as a Ref edge target."""

    def __init__(self, value=1):
        self.value = value


class PreparedConsumer(Object):
    """Count the constructor hook used by new-definition formation."""

    preparations = 0

    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        cls.preparations += 1
        return args, kwargs

    def __init__(self, child: Ref[ConcreteDefinition]):
        self.child = child


class RefPair(Object):
    """Retain two reference-valued constructor slots for sharing checks."""

    def __init__(self, left: Ref[ConcreteDefinition], right: Ref[ConcreteDefinition]):
        self.left = left
        self.right = right


class MatConsumer(Object):
    """Receive one exact CDef through a materializing constructor role."""

    def __init__(self, child: Mat[ConcreteDefinition]):
        self.child = child


class PlainConsumer(Object):
    """Exercise the default materializing constructor slot."""

    def __init__(self, value):
        self.value = value


class ExactDefinitionDataConsumer(Object):
    """Retain exact Definition data rather than its persisted quotation marker."""

    def __init__(self, value: Ref[Definition]):
        self.value = value


class ExactSelectorDataConsumer(Object):
    """Retain exact Selector data rather than its persisted quotation marker."""

    def __init__(self, value: Ref[Selector]):
        self.value = value


def test_constructor_binding_never_invokes_retired_hooks() -> None:
    """New, bound, and persisted paths never invoke retired constructor hooks."""

    child = Definition(ConsumerLeaf).concretize()
    PreparedConsumer.preparations = 0
    cdef = Definition(PreparedConsumer, child).concretize()

    assert PreparedConsumer.preparations == 0
    plan = compile_signature(PreparedConsumer, constructor=True)
    bound = plan.prepare_bound(BoundArguments((("child", child),)))
    assert bound.canonical["child"].kind is EdgeKind.REF
    assert PreparedConsumer.preparations == 0

    restored = pickle.loads(pickle.dumps(cdef))
    assert restored == cdef
    assert PreparedConsumer.preparations == 0


def test_constructor_ref_annotations_choose_finalized_edges_and_keep_sharing() -> None:
    """Live and supplied recipes pass through the one annotation-owned edge rule."""

    supplied = Definition(ConsumerLeaf, 2).concretize()
    live = ConsumerLeaf(3)
    supplied_pair = Definition(RefPair, supplied, supplied).concretize()
    live_pair = Definition(RefPair, live, live).concretize()

    for pair, expected in ((supplied_pair, supplied), (live_pair, live.definition)):
        assert pair.parameters["left"].kind is EdgeKind.REF
        assert pair.parameters["left"].target is expected
        assert pair.parameters["right"].target is expected
        assert pair.parameters["left"].target is pair.parameters["right"].target


def test_fresh_constructor_exact_roles_validate_finalized_link_authority() -> None:
    """Compatible canonical links normalize while opposing fresh links fail."""

    child = Definition(ConsumerLeaf, 3).concretize()
    ref_link = DefLink.finalized(EdgeKind.REF, child)
    mat_link = DefLink.finalized(EdgeKind.MATERIALIZE, child)

    reference_owner = Definition(PreparedConsumer, ref_link).concretize()
    material_owner = Definition(MatConsumer, mat_link).concretize()

    assert reference_owner.parameters["child"].kind is EdgeKind.REF
    assert reference_owner.parameters["child"].target is child
    assert material_owner.parameters["child"] is child
    assert Repo().load_or_build(reference_owner).child is child
    assert isinstance(Repo().load_or_build(material_owner).child, ConsumerLeaf)

    with pytest.raises(SignatureError, match="conflicts"):
        Definition(PreparedConsumer, mat_link).concretize()
    with pytest.raises(SignatureError, match="conflicts"):
        Definition(MatConsumer, ref_link).concretize()


def test_persisted_ref_edges_replay_without_signature_reinterpretation() -> None:
    """Codec and pickle reconstruction retain existing normalized Ref bytes."""

    child = Definition(ConsumerLeaf, 4).concretize()
    root = Definition(
        PlainConsumer, DefLink.finalized(EdgeKind.REF, child)
    ).concretize()
    encoded = encode_cdef_graph(root)

    restored = decode_cdef_graph(encoded)

    assert encode_cdef_graph(restored) == encoded
    assert restored.parameters["value"].kind is EdgeKind.REF
    built = Repo().load_or_build(restored)
    assert built.value is restored.parameters["value"].target
    assert not isinstance(built.value, DefLink)


def test_constructor_exact_data_roles_round_trip_and_decode_without_recompilation(
    monkeypatch,
) -> None:
    """Canonical quotation markers replay as exact Definition/Selector data."""

    definition = Definition(ConsumerLeaf)
    selector = Selector(Definition(ConsumerLeaf, value=Present()))
    definition_cdef = Definition(ExactDefinitionDataConsumer, definition).concretize()
    selector_cdef = Definition(ExactSelectorDataConsumer, selector).concretize()

    definition_marker = definition_cdef.parameters["value"]
    selector_marker = selector_cdef.parameters["value"]
    assert isinstance(definition_marker, DefLink)
    assert isinstance(definition_marker.target, QuotedDef)
    assert isinstance(selector_marker, DefLink)
    assert isinstance(selector_marker.target, SelectorSpec)
    assert ConcreteDefinitionGraph.from_root(definition_cdef).edges() == ()
    assert ConcreteDefinitionGraph.from_root(selector_cdef).edges() == ()
    assert encode_cdef_graph(definition_cdef) != encode_cdef_graph(selector_cdef)

    restored_definition = decode_cdef_graph(encode_cdef_graph(definition_cdef))
    restored_selector = decode_cdef_graph(encode_cdef_graph(selector_cdef))
    monkeypatch.setattr(
        "dryml.core.signatures.compile_signature",
        lambda *args, **kwargs: pytest.fail("persisted constructor replay compiled a signature"),
    )

    definition_owner = Repo().load_or_build(restored_definition)
    selector_owner = Repo().load_or_build(restored_selector)
    assert isinstance(definition_owner.value, Definition)
    assert definition_owner.value == definition
    assert isinstance(selector_owner.value, Selector)
    assert selector_owner.value == selector


def test_direct_quotation_roles_still_deliver_wrapper_types() -> None:
    """Ref[QuotedDef]/Ref[SelectorSpec] remain distinct from exact unwrapped data."""

    quoted = QuotedDef(Definition(ConsumerLeaf))
    spec = SelectorSpec(Selector(Definition(ConsumerLeaf)))

    def target(definition, selector):
        return definition, selector

    target.__annotations__ = {
        "definition": Ref[QuotedDef],
        "selector": Ref[SelectorSpec],
    }
    result = compile_signature(target).prepare_args((quoted, spec), {}).deliver_args()[0]
    assert result == (quoted, spec)


@pytest.mark.parametrize(
    ("consumer", "value", "query_value"),
    (
        (
            ExactDefinitionDataConsumer,
            Definition(ConsumerLeaf),
            QuotedDef(Definition(ConsumerLeaf)),
        ),
        (
            ExactSelectorDataConsumer,
            Selector(Definition(ConsumerLeaf, value=Present())),
            SelectorSpec(Selector(Definition(ConsumerLeaf, value=Present()))),
        ),
    ),
)
def test_constructor_exact_data_markers_remain_query_transparent(
    consumer, value, query_value,
) -> None:
    """Query indexing treats constructor markers as their quotation payloads."""

    cdef = Definition(consumer, value).concretize()
    repo = Repo()
    repo._query_catalog.register_stored(cdef, object())

    selector = Definition(consumer, query_value)
    assert repo.query(selector).stored(refresh=False).count() == 1


def test_partial_query_definitions_do_not_activate_constructor_signatures(monkeypatch) -> None:
    """Partial selectors keep matchers inert and never bind or prepare constructors."""

    target = Definition(PlainConsumer, 3).concretize()
    repo = Repo()
    repo._query_catalog.register_stored(target, object())
    monkeypatch.setattr(
        "dryml.core.signatures.compile_signature",
        lambda *args, **kwargs: pytest.fail("query selector activated a constructor signature"),
    )

    assert repo.query(Definition(PlainConsumer, value=Present())).stored(refresh=False).count() == 1


def test_cdef_codec_never_compiles_signatures_or_resolves_classes(monkeypatch) -> None:
    """CDef graph hydration is a normalized-record operation with no live effects."""

    root = Definition(PlainConsumer, 5).concretize()
    encoded = encode_cdef_graph(root)
    monkeypatch.setattr(
        "dryml.core.signatures.compile_signature",
        lambda *args, **kwargs: pytest.fail("codec compiled a signature"),
    )
    monkeypatch.setattr(
        "dryml.core.symbol.resolve_symbol",
        lambda *args, **kwargs: pytest.fail("codec resolved a class"),
    )

    assert decode_cdef_graph(encoded).graph_equal(root)


def test_recursive_values_and_factory_specs_keep_existing_canonical_rules() -> None:
    """Recursive containers still reject while FactorySpec remains an identity leaf."""

    cdef = Definition(PlainConsumer, FactorySpec("Factory", 1)).concretize()
    assert cdef.parameters["value"] == FactorySpec("Factory", 1)

    cycle = []
    cycle.append(cycle)
    with pytest.raises(Exception, match="Cycle"):
        Definition(PlainConsumer, cycle).concretize()
    def nested(value: Ref[list[ConcreteDefinition]]):
        return value

    with pytest.raises(SignatureError):
        compile_signature(nested)


def test_all_constructor_entry_points_delegate_to_signature_normalization(monkeypatch) -> None:
    """Definition, exact construction, and direct Objects share one normalizer."""

    import dryml.core.signatures as signatures

    original = signatures._normalize_value
    calls = []

    def observe(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(signatures, "_normalize_value", observe)

    Definition(PlainConsumer, 1).concretize()
    ConcreteDefinition(PlainConsumer, (2,), {})
    PlainConsumer(3)

    assert calls == ["value", "value", "value"]


def test_public_signature_surface_has_one_owner_and_no_role_facades() -> None:
    """Core and root re-export only the supported signature vocabulary."""

    import dryml
    import dryml.core as core
    import dryml.core.signatures as signatures

    public = (
        "Ref", "Mat", "AutoRef", "normalize_args", "normalize_return",
        "signature_context", "function", "SignatureError",
    )
    retired = (
        "ArgRole", "MaterializeArg", "RefCDef", "RefCDefArg", "SelectorArg",
        "ValueArg", "apply_arg_roles", "apply_bound_arg_roles",
        "apply_definition_arg_roles", "resolve_arg_roles", "normalize_role",
        "role_from_annotation",
    )

    assert all(getattr(dryml, name) is getattr(core, name) is getattr(signatures, name) for name in public)
    import dryml.core.links as links

    assert not any(hasattr(links, name) for name in ("Ref", "Mat"))
    assert not any(hasattr(module, name) for module in (dryml, core) for name in retired)
    assert importlib.util.find_spec("dryml.core.arg_roles") is None
    assert not any(hasattr(core, name) for name in ("SignaturePlan", "BoundaryPlan", "compile_signature"))
    assert all(hasattr(signatures, name) for name in ("SignaturePlan", "BoundaryPlan", "compile_signature"))


def test_public_wrapper_delegates_selection_and_materialization_to_the_core_owner(monkeypatch) -> None:
    """A backend-free public flow binds, selects, and realizes without private policy."""

    import dryml
    import dryml.core.signatures as signatures

    cdef = Definition(ConsumerLeaf, 9).concretize()
    calls = []
    original = signatures._normalize_value

    def observe(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(signatures, "_normalize_value", observe)

    def build(value):
        return value.definition

    build.__annotations__ = {
        "value": dryml.Mat[ConcreteDefinition],
        "return": dryml.Ref[ConcreteDefinition],
    }
    build = dryml.function(build)

    with dryml.signature_context(repo=Repo()):
        result = build(cdef)

    assert result == cdef
    assert calls == ["value", "return"]


def test_signature_documentation_centralizes_contract_and_migration() -> None:
    """The public page owns conversion rules while consumer pages link to it."""

    docs = Path(__file__).resolve().parents[2] / "docs"
    signatures = (docs / "signatures.md").read_text(encoding="utf-8")
    required = (
        "AutoRef", "Same-role unions", "Nullable forms", "QuotedDef",
        "already fully bound", "available", "Mat[StateRef]", "reuse_live",
        "normalize_args", "signature_context", "Managed", "no aliases",
        "RefCDef", "RefCDefArg", "SelectorArg", "MaterializeArg", "ValueArg",
        "ArgRole", "__dryml_arg_roles__",
    )

    assert all(item.lower() in signatures.lower() for item in required)
    for name in (
        "ref_selector_values.md", "objects_and_defs.md", "immutable_definition_graph.md",
        "methods.md", "managed_operations.md",
    ):
        assert "signatures.md" in (docs / name).read_text(encoding="utf-8")
