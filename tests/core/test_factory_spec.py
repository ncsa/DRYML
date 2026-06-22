import pytest

from dryml.core2 import FactorySpec, Object
from dryml.core2.canonical import NodeKind, node_kind


class FactoryTarget:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Namespace:
    FactoryTarget = FactoryTarget


class FactoryObject(Object):
    pass


def test_factory_spec_builds_from_namespace_short_name():
    spec = FactorySpec("FactoryTarget", 1, label="x")

    obj = spec.build(namespace=Namespace, instance_type=FactoryTarget)

    assert obj.args == (1,)
    assert obj.kwargs == {"label": "x"}


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
