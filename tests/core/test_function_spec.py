from __future__ import annotations

import pytest

from dryml.core2.function_spec import FunctionSpec
from dryml.core2.definition import Definition, ConcreteDefinition
from dryml.core2.object import Object
from dryml.core2.freeze import FrozenTuple


def top_level_add1(x):
    return x + 1


class FunctionHolder(Object):
    def __init__(self, fn):
        self.fn = fn


class FunctionKwargHolder(Object):
    def __init__(self, *, fn):
        self.fn = fn


def make_local_add2():
    def local_add2(x):
        return x + 2
    return local_add2


def make_multiplier(scale):
    def mul(x):
        return x * scale
    return mul


def test_function_spec_from_function_import():
    spec = FunctionSpec.from_function(top_level_add1)
    assert spec.kind == "import"
    assert spec.resolve()(10) == 11


def test_function_spec_from_function_named_source():
    def local_add2(x):
        return x + 2

    spec = FunctionSpec.from_function(local_add2)
    assert spec.kind == "source"
    assert spec.name == "local_add2"
    assert spec.resolve()(10) == 12


def test_function_spec_from_function_lambda_source():
    f = lambda x: x + 3

    spec = FunctionSpec.from_function(f)
    assert spec.kind == "source"
    assert spec.name is None
    assert spec.resolve()(10) == 13


def test_function_spec_rejects_closure():
    def outer(scale):
        def inner(x):
            return x * scale
        return inner

    inner = outer(5)

    with pytest.raises(ValueError, match="closure"):
        FunctionSpec.from_function(inner)


def test_concrete_definition_import_function_arg():
    d = Definition(FunctionHolder, top_level_add1)
    cdef = d.concretize()

    fn_spec = cdef["args"][0]
    assert isinstance(fn_spec, FunctionSpec)
    assert fn_spec.kind == "import"
    assert fn_spec.module == __name__
    assert fn_spec.qualname == "top_level_add1"

    fn = fn_spec.resolve()
    assert fn is top_level_add1
    assert fn(10) == 11


def test_concrete_definition_import_function_kwarg():
    d = Definition(FunctionKwargHolder, fn=top_level_add1)
    cdef = d.concretize()

    fn_spec = cdef["kwargs"]["fn"]
    assert isinstance(fn_spec, FunctionSpec)
    assert fn_spec.kind == "import"
    assert fn_spec.resolve()(10) == 11


def test_concrete_definition_local_function_falls_back_to_source():
    local_add2 = make_local_add2()

    d = Definition(FunctionHolder, local_add2)
    cdef = d.concretize()

    fn_spec = cdef["args"][0]
    assert isinstance(fn_spec, FunctionSpec)
    assert fn_spec.kind == "source"
    assert fn_spec.name == "local_add2"

    fn = fn_spec.resolve()
    assert fn(10) == 12


def test_concrete_definition_equal_for_same_import_function():
    cdef1 = Definition(FunctionHolder, top_level_add1).concretize()
    cdef2 = Definition(FunctionHolder, top_level_add1).concretize()

    assert cdef1 == cdef2
    assert hash(cdef1) == hash(cdef2)


def test_concrete_definition_equal_for_same_local_function_source():
    f1 = make_local_add2()
    f2 = make_local_add2()

    cdef1 = Definition(FunctionHolder, f1).concretize()
    cdef2 = Definition(FunctionHolder, f2).concretize()

    assert cdef1["args"][0].kind == "source"
    assert cdef2["args"][0].kind == "source"

    assert cdef1 == cdef2
    assert hash(cdef1) == hash(cdef2)


def test_concrete_definition_rejects_closure_function():
    fn = make_multiplier(5)

    with pytest.raises(ValueError, match="closure"):
        Definition(FunctionHolder, fn).concretize()


def test_concrete_definition_function_spec_passthrough():
    spec = FunctionSpec.from_function(top_level_add1)
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    assert cdef["args"][0] is spec


def test_concrete_definition_source_lambda_string():
    spec = FunctionSpec.from_source("lambda x: x + 3")
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    fn = cdef["args"][0].resolve()
    assert fn(10) == 13


def test_concrete_definition_thaw_source_lambda_string():
    spec = FunctionSpec.from_source("lambda x: x + 3")
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    d = cdef.thaw()

    assert d.args[0](10) == 13


def test_concrete_definition_thaw_function_spec():
    spec = FunctionSpec.from_function(top_level_add1)
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    d = cdef.thaw()

    assert d.args[0](3) == 4


def test_concrete_definition_build_source_lambda_string():
    spec = FunctionSpec.from_source("lambda x: x + 3")
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    o = cdef.build()

    assert o.fn(10) == 13


def test_concrete_definition_build_function_spec():
    spec = FunctionSpec.from_function(top_level_add1)
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    o = cdef.build()

    assert o.fn(3) == 4
