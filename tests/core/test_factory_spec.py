import pytest

from dryml import F as RootF
from dryml import FactorySpec as RootFactorySpec
from dryml.core import F, FactorySpec, Object
from dryml.core.canonical import NodeKind, node_kind


class FactoryTarget:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Namespace:
    FactoryTarget = FactoryTarget


class FactoryObject(Object):
    pass


class UnsupportedDefaultTarget:
    def __init__(self, unsupported=object()):
        self.unsupported = unsupported


class FailingTarget:
    def __init__(self):
        raise RuntimeError("constructor failed")


def test_public_factory_aliases_are_one_class():
    assert F is FactorySpec is RootF is RootFactorySpec


def test_factory_spec_builds_from_namespace_short_name():
    spec = FactorySpec("FactoryTarget", 1, label="x")

    obj = spec.build(namespace=Namespace, instance_type=FactoryTarget)

    assert obj.args == (1,)
    assert obj.kwargs == {"label": "x"}


def test_factory_spec_keeps_omitted_defaults_and_call_spellings_opaque():
    omitted_default = F(UnsupportedDefaultTarget)
    positional = F("FactoryTarget", 1)
    keyword = F("FactoryTarget", value=1)

    assert omitted_default.args == ()
    assert omitted_default.kwargs == {}
    assert positional != keyword
    assert positional.__stable_leaf_bytes__() != keyword.__stable_leaf_bytes__()


def test_factory_spec_prefers_namespace_and_propagates_build_failures():
    namespaced = F("builtins.dict").build(namespace={"builtins.dict": FactoryTarget})

    assert isinstance(namespaced, FactoryTarget)
    with pytest.raises(ValueError, match="without a namespace"):
        F("UnresolvedTarget").build()
    with pytest.raises(RuntimeError, match="constructor failed"):
        F(FailingTarget).build()
    with pytest.raises(TypeError, match="expected FactoryTarget"):
        F(object).build(instance_type=FactoryTarget)


def test_factory_spec_coerces_tuple_shorthand():
    assert FactorySpec.coerce(("FactoryTarget",)).args == ()
    assert FactorySpec.coerce(("FactoryTarget", {"label": "x"})).kwargs["label"] == "x"
    assert FactorySpec.coerce(("FactoryTarget", 1, {"label": "x"})).args == (1,)
    assert FactorySpec.coerce(("FactoryTarget", (1, 2))).args == ((1, 2),)
    assert FactorySpec.coerce(("FactoryTarget", (1, 2), {})).args == (1, 2)


def test_factory_spec_coerce_many_allows_passthrough_unless_strict():
    sentinel = object()

    prepared = FactorySpec.coerce_many([("FactoryTarget",), sentinel])

    assert isinstance(prepared[0], FactorySpec)
    assert prepared[1] is sentinel

    with pytest.raises(TypeError):
        FactorySpec.coerce_many([sentinel], strict=True)


def test_factory_spec_is_canonical_leaf():
    assert node_kind(FactorySpec("FactoryTarget")) is NodeKind.IDENTITY_VALUE


def test_factory_spec_rejects_hidden_dryml_graph_nodes():
    obj = FactoryObject()

    with pytest.raises(TypeError, match="DRYML graph nodes"):
        FactorySpec("FactoryTarget", obj)
