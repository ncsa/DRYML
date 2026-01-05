

def test_cdef_payload_frozen_lists():
    import core2_objects as objs
    d = Definition(objs.TestClass1, layer_dims=[[1,2,3],[4,5,6]]).concretize()
    assert d.args[0] == ((1,2,3),(4,5,6))
    # try to mutate nested
    with pytest.raises(TypeError):
        d.args[0][0] += (999,)  # tuple immutability


def test_cdef_immutable_mapping():
    import core2_objects as objs
    d = Definition(objs.TestClass1, 10, test={"a": [1,2]}).concretize()
    with pytest.raises(TypeError):
        d["args"] = (20,)
    with pytest.raises(TypeError):
        d.kwargs["test"]["a"] = (1,2,3)  # FrozenDict blocks


def test_definition_unhashable():
    import core2_objects as objs
    d = Definition(objs.TestClass1, 10)
    with pytest.raises(TypeError):
        hash(d)


def test_hash_invariant_freeze_thaw_containers():
    x = {"a": [1, 2, (3, 4)], "b": {"c": 5}}
    hx = stable_hash_function(x)
    fx = deep_freeze(x)
    hfx = stable_hash_function(fx)
    assert hx == hfx


def test_hash_set_order_independent():
    s1 = {1, 2, 3, 4}
    s2 = {4, 3, 2, 1}
    assert stable_hash_function(s1) == stable_hash_function(s2)


def test_hash_list_order_dependent():
    assert stable_hash_function([1,2,3]) != stable_hash_function([3,2,1])
