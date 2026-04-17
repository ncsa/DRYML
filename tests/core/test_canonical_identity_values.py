from __future__ import annotations

from dryml.core2.canonical import (
    NodeKind,
    from_canonical,
    is_canonical_value,
    is_runtime_leaf,
    node_kind,
    thaw_value,
    to_canonical,
)
from dryml.core2.dtype import DType
from dryml.core2.repo import Repo
from dryml.core2.tensor_spec import Dynamic, Layout, TensorSpec

from dryml.core2.definition import ConcreteDefinition, Definition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.object import Object
from dryml.core2.repo import default_repo


class _DummyObject(Object):
    def __init__(self, *args, **kwargs):
        super().__init__()


def test_dtype_is_identity_value():
    dt = DType("float", 32)

    assert node_kind(dt) is NodeKind.IDENTITY_VALUE
    assert is_canonical_value(dt)
    assert is_runtime_leaf(dt)


def test_tensorspec_is_identity_value():
    spec = TensorSpec(
        dtype=DType("float", 32),
        shape=(28, 28, 1),
        batch=Dynamic,
        layout=Layout.DENSE,
    )

    assert node_kind(spec) is NodeKind.IDENTITY_VALUE
    assert is_canonical_value(spec)
    assert is_runtime_leaf(spec)


def test_enum_values_are_identity_values():
    assert node_kind(Dynamic) is NodeKind.IDENTITY_VALUE
    assert node_kind(Layout.DENSE) is NodeKind.IDENTITY_VALUE

    assert is_canonical_value(Dynamic)
    assert is_canonical_value(Layout.DENSE)

    assert is_runtime_leaf(Dynamic)
    assert is_runtime_leaf(Layout.DENSE)


def test_to_canonical_passes_dtype_through_unchanged():
    dt = DType("int", 64)
    out = to_canonical(dt)

    assert out is dt


def test_to_canonical_passes_tensorspec_through_unchanged():
    spec = TensorSpec(
        dtype=DType("float", 32),
        shape=(16,),
        batch=None,
    )
    out = to_canonical(spec)

    assert out is spec


def test_thaw_value_passes_dtype_through_unchanged():
    dt = DType("bool", None)
    out = thaw_value(dt)

    assert out is dt


def test_thaw_value_passes_tensorspec_through_unchanged():
    spec = TensorSpec(
        dtype=DType("bfloat", 16),
        shape=(4, 4),
        batch=8,
    )
    out = thaw_value(spec)

    assert out is spec


def test_from_canonical_passes_dtype_through_unchanged():
    repo = Repo()
    dt = DType("float", 64)

    out = from_canonical(dt, repo=repo)

    assert out is dt


def test_from_canonical_passes_tensorspec_through_unchanged():
    repo = Repo()
    spec = TensorSpec(
        dtype=DType("float", 32),
        shape=(3, 5),
        batch=None,
    )

    out = from_canonical(spec, repo=repo)

    assert out is spec


def test_to_canonical_inside_dict_preserves_identity_value_objects():
    spec = TensorSpec(
        dtype=DType("float", 32),
        shape=(8, 8),
    )

    out = to_canonical({"spec": spec})

    # dict itself becomes canonicalized, but the identity-value child
    # should be preserved as-is
    assert out["spec"] is spec


def test_to_canonical_inside_list_preserves_identity_value_objects():
    dt = DType("int", 32)

    out = to_canonical([dt])

    assert out[0] is dt


def test_thaw_value_inside_frozen_dict_preserves_identity_value_objects():
    spec = TensorSpec(
        dtype=DType("float", 32),
        shape=(2, 2),
    )

    frozen = to_canonical({"spec": spec})
    thawed = thaw_value(frozen)

    assert thawed["spec"] is spec


def test_from_canonical_inside_frozen_dict_preserves_identity_value_objects():
    repo = Repo()
    spec = TensorSpec(
        dtype=DType("float", 32),
        shape=(10,),
    )

    frozen = to_canonical({"spec": spec})
    realized = from_canonical(frozen, repo=repo)

    assert realized["spec"] is spec


def test_concretedefinition_preserves_identity_values_in_args_and_kwargs():
    spec = TensorSpec(
        dtype=DType("float", 32),
        shape=(28, 28, 1),
        batch=Dynamic,
    )
    dt = DType("int", 64)

    cdef = Definition(_DummyObject, spec, dtype=dt).concretize()

    assert isinstance(cdef, ConcreteDefinition)
    assert isinstance(cdef.args, FrozenTuple)
    assert isinstance(cdef.kwargs, FrozenDict)

    # identity-value objects should be preserved as-is inside the canonical structure
    assert cdef.args[0] is spec
    assert cdef.kwargs["dtype"] is dt


def test_thaw_value_preserves_identity_values_inside_concretedefinition():
    spec = TensorSpec(
        dtype=DType("float", 32),
        shape=(16, 16),
    )
    dt = DType("bool", None)

    cdef = ConcreteDefinition(
        _DummyObject,
        FrozenTuple((spec,)),
        FrozenDict({"dtype": dt}),
    )

    defn = thaw_value(cdef)

    assert isinstance(defn, Definition)
    assert defn.args[0] is spec
    assert defn.kwargs["dtype"] is dt


def test_from_canonical_preserves_identity_values_inside_concretedefinition_args_kwargs():
    repo = Repo()
    with default_repo(repo):

        spec = TensorSpec(
            dtype=DType("float", 32),
            shape=(8, 8),
        )
        dt = DType("int", 32)

        cdef = ConcreteDefinition(
            _DummyObject,
            FrozenTuple((spec,)),
            FrozenDict({"dtype": dt}),
        )

        # realize only the arg/kwarg payloads, not the object itself
        rt_args = from_canonical(cdef.args, repo=repo)
        rt_kwargs = from_canonical(cdef.kwargs, repo=repo)

        assert rt_args[0] is spec
        assert rt_kwargs["dtype"] is dt


def test_build_preserves_identity_values_inside_concretedefinition_args_kwargs():
    repo = Repo()
    with default_repo(repo):

        spec = TensorSpec(
            dtype=DType("float", 32),
            shape=(8, 8),
        )
        dt = DType("int", 32)

        cdef = ConcreteDefinition(
            _DummyObject,
            FrozenTuple((spec,)),
            FrozenDict({"dtype": dt}),
        )

        o = cdef.build(repo=repo)

        assert o.__cdef__.args[0] is spec
        assert o.__cdef__.kwargs["dtype"] is dt
