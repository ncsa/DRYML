import pytest
import numpy as np

from dryml.core2 import Definition
from dryml.core2.object import Object
from dryml.core2.freeze import deep_freeze, deep_thaw, FreezeError, FrozenList, FrozenDict


class ReceivesTypes(Object):
    def __init__(self, payload, **kwargs):
        super().__init__(**kwargs)
        self.payload = payload
        # store types for assertions
        self.t_payload = type(payload)
        self.t_a = type(payload["a"])
        self.t_b = type(payload["b"])
        self.t_c = type(payload["b"][1]["c"])


class Inner(Object):
    def __init__(self, xs, **kwargs):
        super().__init__(**kwargs)
        self.xs = xs
        self.t_xs = type(xs)


class Outer(Object):
    def __init__(self, inner, second, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.second = second
        self.t_second = type(second)


def test_deep_freeze_thaw_roundtrip_types_and_aliasing():
    shared = [1, 2, 3]
    x = {
        "p": shared,
        "q": shared,
        "t": (1, 2),
        "s": {4, 5},
        "arr": np.arange(6, dtype=np.int32).reshape(2, 3),
        "nest": [{"k": [9]}],
    }

    fx = deep_freeze(x)
    y = deep_thaw(fx)

    assert isinstance(y["p"], list)
    assert isinstance(y["q"], list)
    assert y["p"] is y["q"]  # alias preserved
    assert isinstance(y["t"], tuple)
    assert isinstance(y["s"], set)
    assert isinstance(y["arr"], np.ndarray)
    assert isinstance(y["nest"], list)
    assert isinstance(y["nest"][0], dict)
    assert isinstance(y["nest"][0]["k"], list)


def test_concretedef_freezes_and_detaches_payload():
    orig = [[1, 2, 3], [4, 5, 6]]
    d = Definition(ReceivesTypes, {"a": [1], "b": (0, {"c": orig})}).concretize()

    # Payload in cdef is frozen
    assert isinstance(d.kwargs, FrozenDict)
    frozen_orig = d.args[0]["b"][1]["c"]
    assert isinstance(frozen_orig, FrozenList)

    # Mutating original input does NOT affect cdef (detach)
    orig[0].append(999)
    assert list(frozen_orig[0]) == [1, 2, 3]  # unchanged

    # Attempt to mutate via cdef should fail
    with pytest.raises(TypeError):
        d.kwargs["newkey"] = 123  # FrozenDict immutable


def test_definition_unhashable_concretedef_hashable():
    d = Definition(ReceivesTypes, {"a": [1], "b": (0, {"c": [2]})})
    with pytest.raises(TypeError):
        hash(d)

    cd = d.concretize()
    assert isinstance(hash(cd), int)


def test_runtime_init_receives_thawed_containers_top_level():
    payload = {"a": [1, 2], "b": (0, {"c": [3, 4]})}

    obj = Definition(ReceivesTypes, payload).build()
    assert obj.t_payload is dict
    assert obj.t_a is list
    assert obj.t_b is tuple
    assert obj.t_c is list


def test_runtime_init_receives_thawed_containers_nested_object():
    inner_def = Definition(Inner, [1, 2, 3])
    outer_def = Definition(Outer, inner_def, second=[4, 5, 6])

    obj = outer_def.build()
    assert isinstance(obj.inner, Inner)
    assert obj.inner.t_xs is list          # Inner saw list (thawed in Repo._load_single_object)
    assert obj.t_second is list            # Outer saw list (thawed in Dryml.__call__)


def test_deep_freeze_rejects_unsupported_types_with_path():
    class Nope: pass
    bad = {"ok": [1], "bad": Nope()}
    with pytest.raises(FreezeError) as e:
        deep_freeze(bad, path=("kwargs",))
    assert "kwargs/bad" in str(e.value) or "bad" in str(e.value)
