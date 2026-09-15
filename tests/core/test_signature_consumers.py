"""Consumer integration coverage for the shared core signature authority."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest

from dryml.core import ConcreteDefinition, Definition, Object, Repo
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_codec import decode_cdef_graph, encode_cdef_graph
from dryml.core.cdef_graph import EdgeKind
from dryml.core.factory import FactorySpec
from dryml.core.freeze import FrozenList
from dryml.core.links import DefLink
from dryml.core.params import Present
from dryml.core.signatures import Ref, SignatureError, compile_signature


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


class PlainConsumer(Object):
    """Exercise the default materializing constructor slot."""

    def __init__(self, value):
        self.value = value


def test_constructor_prepares_once_and_persisted_or_bound_paths_do_not_replay() -> None:
    """New calls use one hook while bound and persisted records remain inert."""

    child = Definition(ConsumerLeaf).concretize()
    PreparedConsumer.preparations = 0
    cdef = Definition(PreparedConsumer, child).concretize()

    assert PreparedConsumer.preparations == 1
    plan = compile_signature(PreparedConsumer, constructor=True)
    bound = plan.prepare_bound(BoundArguments((("child", child),)))
    assert bound.canonical["child"].kind is EdgeKind.REF
    assert PreparedConsumer.preparations == 1

    restored = pickle.loads(pickle.dumps(cdef))
    assert restored == cdef
    assert PreparedConsumer.preparations == 1


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
