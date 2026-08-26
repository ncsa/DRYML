from __future__ import annotations

import pytest

from dryml.core2.symbol import ImportRef, SourceSpec, symbol_ref
from dryml.core2.definition import Definition, ConcreteDefinition
from dryml.core2.bound_args import BoundArguments
from dryml.core2.object import Object
from dryml.core2.freeze import FrozenTuple
from dryml.core2.utils.general import pickler, unpickler
import numpy as np


def top_level_add1(x):
    return x + 1


class FunctionHolder(Object):
    def __init__(self, fn):
        self.fn = fn


class FunctionKwargHolder(Object):
    def __init__(self, *, fn):
        self.fn = fn


class ValueHolder(Object):
    def __init__(self, value):
        self.value = value


def make_local_add2():
    def local_add2(x):
        return x + 2
    return local_add2


def make_multiplier(scale):
    def mul(x):
        return x * scale
    return mul


def test_symbol_ref_from_function_import():
    spec = symbol_ref(top_level_add1)
    assert isinstance(spec, ImportRef)
    assert spec.module == __name__
    assert spec.qualname == "top_level_add1"
    assert spec.resolve()(10) == 11


def test_symbol_ref_from_function_named_source():
    def local_add2(x):
        return x + 2

    spec = symbol_ref(local_add2)
    assert isinstance(spec, SourceSpec)
    assert spec.kind == "function"
    assert spec.name == "local_add2"
    assert spec.resolve()(10) == 12


def test_symbol_ref_from_function_lambda_source():
    f = lambda x: x + 3

    spec = symbol_ref(f)
    assert isinstance(spec, SourceSpec)
    assert spec.kind == "function"
    assert spec.name is None
    assert spec.resolve()(10) == 13


def test_symbol_ref_rejects_closure():
    def outer(scale):
        def inner(x):
            return x * scale
        return inner

    inner = outer(5)

    with pytest.raises(ValueError, match="closure"):
        symbol_ref(inner)


def test_concrete_definition_import_function_arg():
    d = Definition(FunctionHolder, top_level_add1)
    cdef = d.concretize()

    fn_spec = cdef["parameters"]["fn"]
    assert isinstance(fn_spec, ImportRef)
    assert fn_spec.module == __name__
    assert fn_spec.qualname == "top_level_add1"

    fn = fn_spec.resolve()
    assert fn is top_level_add1
    assert fn(10) == 11


def test_concrete_definition_import_function_kwarg():
    d = Definition(FunctionKwargHolder, fn=top_level_add1)
    cdef = d.concretize()

    fn_spec = cdef["parameters"]["fn"]
    assert isinstance(fn_spec, ImportRef)
    assert fn_spec.resolve()(10) == 11


def test_concrete_definition_local_function_falls_back_to_source():
    local_add2 = make_local_add2()

    d = Definition(FunctionHolder, local_add2)
    cdef = d.concretize()

    fn_spec = cdef["parameters"]["fn"]
    assert isinstance(fn_spec, SourceSpec)
    assert fn_spec.kind == "function"
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

    assert cdef1.parameters["fn"].kind == "function"
    assert cdef2.parameters["fn"].kind == "function"

    assert cdef1 == cdef2
    assert hash(cdef1) == hash(cdef2)


def test_concrete_definition_rejects_closure_function():
    fn = make_multiplier(5)

    with pytest.raises(TypeError, match="Anonymous function"):
        Definition(FunctionHolder, fn)


def test_concrete_definition_symbol_ref_passthrough():
    spec = symbol_ref(top_level_add1)
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    assert cdef.parameters["fn"] is spec


def test_concrete_definition_source_lambda_string():
    spec = SourceSpec.from_source("lambda x: x + 3")
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    fn = cdef.parameters["fn"].resolve()
    assert fn(10) == 13


def test_concrete_definition_thaw_source_lambda_string():
    spec = SourceSpec.from_source("lambda x: x + 3")
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    d = cdef.thaw()

    assert d.parameters["fn"].resolve()(10) == 13


def test_concrete_definition_thaw_symbol_ref():
    spec = symbol_ref(top_level_add1)
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    d = cdef.thaw()

    assert d.parameters["fn"].resolve()(3) == 4


def test_concrete_definition_build_source_lambda_string():
    spec = SourceSpec.from_source("lambda x: x + 3")
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    o = cdef.build()

    assert o.fn(10) == 13


def test_concrete_definition_build_symbol_ref():
    spec = symbol_ref(top_level_add1)
    cdef = ConcreteDefinition(FunctionHolder, FrozenTuple((spec,)))

    o = cdef.build()

    assert o.fn(3) == 4


def test_source_spec_captures_module_alias():
    def f(x):
        return np.sin(x)

    spec = symbol_ref(f)
    assert isinstance(spec, SourceSpec)
    assert spec.kind == "function"
    assert {name: ref.import_path() for name, ref in spec.imports.items()} == {"np": "numpy"}

    g = spec.resolve()
    assert g(0.0) == 0.0


def test_source_spec_captures_module_alias_nested():
    def build_func():
        def f(x):
            return np.sin(x)
        return f

    spec = symbol_ref(build_func)
    assert isinstance(spec, SourceSpec)
    assert spec.kind == "function"
    assert {name: ref.import_path() for name, ref in spec.imports.items()} == {"np": "numpy"}

    g = spec.resolve()
    assert g()(0.0) == 0.0


def test_source_spec_reuses_live_main_function(monkeypatch):
    import __main__

    def f(x):
        return x + 1

    __main__.f = f
    try:
        spec = symbol_ref(f)
        assert isinstance(spec, SourceSpec)
        assert spec.kind == "function"
        assert spec.resolve() is f
    finally:
        del __main__.f


def test_source_spec_captures_module_alias_via_outer_local():
    def outer():
        lib = np
        def f(x):
            return lib.sin(x)
        return f

    spec = symbol_ref(outer)
    assert isinstance(spec, SourceSpec)
    assert spec.kind == "function"
    assert {name: ref.import_path() for name, ref in spec.imports.items()} == {"np": "numpy"}

    g = spec.resolve()
    assert g()(0.0) == 0.0


def test_source_spec_nested_uses_outer_parameter():
    def outer(scale):
        def f(x):
            return x * scale
        return f

    spec = symbol_ref(outer)
    assert isinstance(spec, SourceSpec)
    assert spec.kind == "function"
    assert spec.imports == {}

    g = spec.resolve()
    assert g(3)(10) == 30


def test_source_spec_nested_uses_outer_parameter_args():
    from dryml.core2.dtype import dtype
    import numpy as np

    def outer(scale, dtype=dtype("float32")):
        def f(x):
            return dtype.np()(x * scale)
        return f

    spec = symbol_ref(outer)
    assert isinstance(spec, SourceSpec)
    assert spec.kind == "function"
    assert {name: ref.import_path() for name, ref in spec.imports.items()} == {
        "dtype": "dryml.core2.dtype:normalize_dtype"
    }

    g = spec.resolve()
    assert g(3)(10) == np.float32(30)


def test_source_spec_captures_kwdefault_factory():
    from dryml.core2.dtype import dtype

    def outer(*, dt=dtype("float32")):
        def f(x):
            return dt.np()(x)
        return f

    spec = symbol_ref(outer)
    assert {name: ref.import_path() for name, ref in spec.imports.items()} == {
        "dtype": "dryml.core2.dtype:normalize_dtype"
    }


def test_concrete_definition_import_class_arg():
    cdef = Definition(ValueHolder, FunctionHolder).concretize()

    assert isinstance(cdef.cls, ImportRef)
    assert cdef.cls.resolve() is ValueHolder
    assert isinstance(cdef.parameters["value"], ImportRef)
    assert cdef.parameters["value"].resolve() is FunctionHolder

    obj = cdef.build()
    assert obj.value is FunctionHolder


def test_concrete_definition_import_refs_survive_serialization():
    cdef = Definition(ValueHolder, FunctionHolder).concretize()

    loaded = unpickler(pickler(cdef))

    assert isinstance(loaded.cls, ImportRef)
    assert isinstance(loaded.parameters["value"], ImportRef)
    assert loaded.cls == cdef.cls
    assert loaded.parameters["value"] == cdef.parameters["value"]


def test_core_classes_remain_naked_in_concrete_definition():
    cdef = Definition(ValueHolder, Object).concretize()

    assert isinstance(cdef.cls, ImportRef)
    assert cdef.parameters["value"] is Object


def test_concrete_definition_local_class_arg_uses_source_spec():
    class LocalValue:
        def __init__(self, x=1):
            self.x = x

    cdef = Definition(ValueHolder, LocalValue).concretize()

    assert isinstance(cdef.parameters["value"], SourceSpec)
    assert cdef.parameters["value"].kind == "class"
    assert cdef.parameters["value"].name == "LocalValue"
    assert cdef.parameters["value"].resolve().__name__ == "LocalValue"


def test_concrete_definition_local_object_class_can_build_from_source_spec():
    class LocalHolder(Object):
        def __init__(self, value):
            self.value = value

    cdef = Definition(LocalHolder, 7).concretize()

    assert isinstance(cdef.cls, SourceSpec)
    assert cdef.cls.kind == "class"

    obj = cdef.build()
    assert type(obj).__name__ == "LocalHolder"
    assert obj.value == 7


def test_v2_source_spec_semantic_access_does_not_resolve_the_source(monkeypatch):
    spec = SourceSpec.from_source(
        "raise AssertionError('semantic inspection executed source')",
        kind="class",
        name="Danger",
    )
    cdef = ConcreteDefinition._from_bound_record(spec, BoundArguments((("value", 7),)))
    monkeypatch.setattr(SourceSpec, "resolve", lambda self: pytest.fail("must not resolve source"))

    assert cdef.value == 7
    assert cdef.parameters["value"] == 7
    with pytest.raises(AttributeError):
        cdef.missing
