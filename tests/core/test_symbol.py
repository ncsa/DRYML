from __future__ import annotations

import pytest
import types

from dryml.core.symbol import ImportRef, SourceSpec, symbol_ref
from dryml.core.definition import Definition, ConcreteDefinition
from dryml.core.bound_args import BoundArguments
from dryml.core.object import Object
from dryml.core.repo import Repo
from dryml.core.store.dir import DirStore
from dryml.core.freeze import FrozenTuple
from dryml.core.utils.general import pickler, unpickler
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


def caller_anchor_template(value):
    return caller_anchor_dependency.sin(value)


def resolution_precedence_template(value):
    return resolution_precedence_dependency.sin(value)


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
    from dryml.core.dtype import dtype
    import numpy as np

    def outer(scale, dtype=dtype("float32")):
        def f(x):
            return dtype.np()(x * scale)
        return f

    spec = symbol_ref(outer)
    assert isinstance(spec, SourceSpec)
    assert spec.kind == "function"
    assert {name: ref.import_path() for name, ref in spec.imports.items()} == {
        "dtype": "dryml.core.dtype:normalize_dtype"
    }

    g = spec.resolve()
    assert g(3)(10) == np.float32(30)


def test_source_spec_captures_kwdefault_factory():
    from dryml.core.dtype import dtype

    def outer(*, dt=dtype("float32")):
        def f(x):
            return dt.np()(x)
        return f

    spec = symbol_ref(outer)
    assert {name: ref.import_path() for name, ref in spec.imports.items()} == {
        "dtype": "dryml.core.dtype:normalize_dtype"
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


def test_source_spec_characterization_preserves_canonical_bytes_and_identity():
    """Freeze alias, nested-local, default, and class-header source capture behavior."""

    def outer(scale, dtype=np.dtype("float32")):
        def nested(value):
            return dtype.type(value * scale)
        return nested

    class LocalArray(np.ndarray):
        pass

    outer_spec = symbol_ref(outer)
    class_spec = symbol_ref(LocalArray)

    assert isinstance(outer_spec, SourceSpec)
    assert isinstance(class_spec, SourceSpec)
    assert {name: ref.import_path() for name, ref in outer_spec.imports.items()} == {"np": "numpy"}
    assert {name: ref.import_path() for name, ref in class_spec.imports.items()} == {"np": "numpy"}
    assert outer_spec.__stable_leaf_bytes__() == SourceSpec(
        "function", outer_spec.source, outer_spec.name, outer_spec.imports
    ).__stable_leaf_bytes__()

    first = Definition(FunctionHolder, outer).concretize()
    second = Definition(FunctionHolder, outer).concretize()
    assert first == second
    assert hash(first) == hash(second)


def test_source_spec_missing_dependency_fails_closed():
    """A source-backed function with an unresolved free name never gets a spec."""

    def missing_dependency(value):
        return missing_source_dependency + value

    with pytest.raises(ValueError, match="Missing/unimportable globals"):
        symbol_ref(missing_dependency)


@pytest.mark.parametrize("replacement", [None, object()])
def test_source_spec_rejects_missing_or_wrong_typed_lexical_output(monkeypatch, replacement):
    """Malformed generic lexical results cannot publish a source specification."""

    import dryml.code.algorithms.lexical_dependencies as lexical

    def needs_numpy(value):
        return np.sin(value)

    monkeypatch.setattr(lexical, "collect_lexical_dependencies", lambda target: replacement)
    with pytest.raises(ValueError, match="Could not capture stable import paths"):
        symbol_ref(needs_numpy)


def test_source_spec_rejects_failed_or_partial_lexical_output(monkeypatch):
    """Generic execution failures and malformed duplicate evidence fail closed."""

    import dryml.code.algorithms.lexical_dependencies as lexical

    def needs_numpy(value):
        return np.sin(value)

    monkeypatch.setattr(lexical, "collect_lexical_dependencies", lambda target: (_ for _ in ()).throw(RuntimeError("private generic detail")))
    with pytest.raises(ValueError, match="Could not capture stable import paths") as error:
        symbol_ref(needs_numpy)
    assert "private generic detail" not in str(error.value)

    duplicate = lexical.LexicalDependencies((
        lexical.LexicalDependency("np", None),
        lexical.LexicalDependency("np", None),
    ))
    monkeypatch.setattr(lexical, "collect_lexical_dependencies", lambda target: duplicate)
    with pytest.raises(ValueError, match="Could not capture stable import paths"):
        symbol_ref(needs_numpy)


def test_source_spec_captures_caller_anchor_before_lexical_analysis(monkeypatch):
    """A generic helper local cannot outrank the caller-local resolution anchor."""

    import dryml.code.algorithms.lexical_dependencies as lexical

    detached = types.FunctionType(
        caller_anchor_template.__code__, {}, name=caller_anchor_template.__name__
    )
    detached.__module__ = __name__
    caller_anchor_dependency = np

    def helper(source):
        caller_anchor_dependency = __import__("math")
        return lexical.LexicalDependencies((lexical.LexicalDependency("caller_anchor_dependency", None),))

    monkeypatch.setattr(lexical, "collect_lexical_dependencies", helper)
    spec = SourceSpec.from_function(detached)
    assert spec.imports["caller_anchor_dependency"].import_path() == "numpy"


def test_source_spec_preserves_globals_module_and_caller_precedence(monkeypatch):
    """Live dependency projection keeps the documented globals/module/caller order."""

    import math
    import sys

    globals_first = types.FunctionType(
        resolution_precedence_template.__code__,
        {"resolution_precedence_dependency": np},
        name=resolution_precedence_template.__name__,
    )
    globals_first.__module__ = __name__
    assert SourceSpec.from_function(globals_first).imports[
        "resolution_precedence_dependency"
    ].import_path() == "numpy"

    monkeypatch.setattr(sys.modules[__name__], "resolution_precedence_dependency", math, raising=False)
    module_second = types.FunctionType(
        resolution_precedence_template.__code__, {}, name=resolution_precedence_template.__name__
    )
    module_second.__module__ = __name__
    assert SourceSpec.from_function(module_second).imports[
        "resolution_precedence_dependency"
    ].import_path() == "math"

    caller_third = types.FunctionType(
        resolution_precedence_template.__code__, {}, name=resolution_precedence_template.__name__
    )
    caller_third.__module__ = "missing_source_spec_module"
    resolution_precedence_dependency = np
    assert SourceSpec.from_function(caller_third).imports[
        "resolution_precedence_dependency"
    ].import_path() == "numpy"


def test_source_spec_identity_round_trips_through_store_lookup(tmp_path):
    """Source-backed CDef identity remains usable for persisted structural lookup."""

    def local(value):
        return value + 1

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    saved = FunctionHolder(local, repo=repo)
    repo.save_object(saved)

    loaded = Repo(DirStore(tmp_path / "store")).load(Definition(FunctionHolder, local))
    assert loaded.definition == saved.definition
    assert hash(loaded.definition) == hash(saved.definition)
    assert loaded.fn(3) == 4
