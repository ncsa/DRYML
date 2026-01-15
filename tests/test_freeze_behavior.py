import pytest
import numpy as np

from dryml.core2 import Definition
from dryml.core2.object import Object
from dryml.core2.freeze import deep_freeze, deep_thaw, FreezeError, CycleError, FrozenList, FrozenDict


class Receives(Object):
    def __init__(self, payload, **kwargs):
        super().__init__(**kwargs)
        self.payload = payload
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


def test_definition_unhashable_concretedef_hashable():
    d = Definition(Receives, {"a": [1], "b": (0, {"c": [2]})})
    with pytest.raises(TypeError):
        hash(d)

    cd = d.concretize()
    assert isinstance(hash(cd), int)


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
    cd = Definition(Receives, {"a": [1], "b": (0, {"c": orig})}).concretize()

    frozen_payload = cd.args[0]
    assert isinstance(frozen_payload, FrozenDict)

    frozen_c = frozen_payload["b"][1]["c"]
    assert isinstance(frozen_c, FrozenList)

    # Mutating original input does NOT affect cdef payload
    orig[0].append(999)
    assert list(frozen_c[0]) == [1, 2, 3]

    # Cannot mutate frozen mapping
    with pytest.raises(TypeError):
        frozen_payload["newkey"] = 123  # Mapping is immutable / no __setitem__


def test_concretedef_itself_is_immutable():
    cd = Definition(Receives, {"a": [1], "b": (0, {"c": [2]})}).concretize()
    with pytest.raises(TypeError):
        cd["kwargs"] = {}  # ConcreteDefinition dict payload should be immutable


def test_runtime_init_receives_thawed_containers():
    payload = {"a": [1, 2], "b": (0, {"c": [3, 4]})}
    obj = Definition(Receives, payload).build()

    assert obj.t_payload is dict
    assert obj.t_a is list
    assert obj.t_b is tuple
    assert obj.t_c is list


def test_runtime_init_thaw_nested_objects_and_kwargs():
    inner_def = Definition(Inner, [1, 2, 3])
    outer_def = Definition(Outer, inner_def, second=[4, 5, 6])

    obj = outer_def.build()
    assert isinstance(obj.inner, Inner)
    assert obj.inner.t_xs is list
    assert obj.t_second is list


def test_concretize_rejects_selector_only_values():
    # Callables are fine in Definition matching, but should not be concretizable
    d = Definition(Receives, {"a": lambda x: True, "b": (0, {"c": [1]})})
    with pytest.raises((FreezeError, TypeError)):
        d.concretize()


def test_concretize_rejects_cycles():
    cyc = []
    cyc.append(cyc)
    d = Definition(Inner, cyc)
    with pytest.raises(CycleError):
        d.concretize()


def test_freeze_rejects_unknown_leaf_with_path():
    class Nope: pass
    bad = {"ok": [1], "bad": Nope()}
    with pytest.raises(FreezeError) as e:
        deep_freeze(bad, path=("kwargs",))
    assert "kwargs/bad" in str(e.value)
