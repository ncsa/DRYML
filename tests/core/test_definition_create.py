import numpy as np
import pytest
from tests.core import core_objects as objects
import dryml
from dryml.core.definition import Definition, ConcreteDefinition, SKIP_ARGS
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.object import Object
from dryml.core.symbol import ImportRef

### Tests for methods of creating definitions. We verify they have the intended properties.


# Directly creating Definition objects - Check they have expected properties

def test_create_definition_1():
    # Empty definition creation
    Definition()


def test_skip_args_is_available_from_top_level_dryml():
    assert dryml.SKIP_ARGS is SKIP_ARGS


def test_create_definition_2():
    # Plain single level definition
    definition = Definition(objects.TestClass1, 10, test='a')
    assert definition.cls == objects.TestClass1
    assert definition['cls'] == objects.TestClass1
    assert len(definition.args) == 1
    assert definition.args[0] == 10
    assert len(definition.kwargs) == 1
    assert definition.kwargs["test"] == "a"


def test_create_definition_3():
    # nesting definitions shouldn't change the nested Definition objects
    def_1 = Definition(objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1,
        def_1,
        test='b')
    assert def_2.args[0] is def_1


# Creating definitions through `def` classmethod - They have the correct values
def test_create_definition_4():
    """
    Definitions can be created through the 'def' classmethod and they have the correct values
    """
    obj_def = objects.TestClassA3.d(10)

    assert type(obj_def) is Definition

    assert obj_def.args[0] == 10


# def test_create_definition_5():
#     """
#     Definitions can be created, they have the correct values and they
#     weren't copied yet.
#     """
#     val = objects.DeepcopyAware(10)
#     assert val.counter == 0

#     obj_def = objects.TestClassA3.d(val)

#     assert obj_def.args[0].counter == 0


# def test_create_definition_6():
#     """
#     Definitions can be created, they have the correct values.
#     Test deepcopy and the DeepcopyAware object as well
#     """
#     val = objects.DeepcopyAware(10)
#     assert val.counter == 0

#     obj_def = objects.TestClassA3.d(val)

#     assert obj_def.args[0].counter == 0

#     new_obj_def = deepcopy(obj_def)
#     assert new_obj_def.args[0].counter == 1


# def test_create_definition_7():
#     """
#     Definitions can be created, they have the correct values.
#     Test deepcopy and the DeepcopyAware object as well
#     """
#     val = objects.DeepcopyAware(10)
#     assert val.counter == 0

#     obj_def = objects.TestClassA3.d(val)

#     assert obj_def.args[0].counter == 0

#     new_obj_def = deepcopy(obj_def)
#     assert new_obj_def.args[0].counter == 1


# def test_create_definition_8():
#     """
#     When we concretize a definition, we have the same objects.
#     """

#     val = objects.DeepcopyAware(10)
#     assert val.counter == 0

#     obj_def = objects.TestClassA3.d(val)

#     obj = obj_def.build()

#     assert obj.A.val == val.val

#     assert obj_def.args[0].counter == 0
#     # The definition should've been deep copied only a single time.
#     assert obj.definition.args[0].counter == 1
#     assert id(val) == id(obj.A)


# def test_def_5():
#     """
#     Nested definition build
#     """

#     val1 = objects.DeepcopyAware(10)
#     val2 = objects.DeepcopyAware(20)

#     obj_def = objects.TestClassA3.d((objects.TestClassA3.d(val2), val1))

#     obj = obj_def.build()

#     assert isinstance(obj, objects.TestClassA3)
#     assert isinstance(obj.A[0], objects.TestClassA3)

#     assert id(val1) == id(obj.A[1])
#     assert id(val2) == id(obj.A[0].A)
#     assert obj.A[0].A.counter == 0
#     assert obj.A[1].counter == 0
#     assert obj.definition.args[0][1].counter == 1
#     assert obj.definition.args[0][0].args[0].counter == 1 # Each object with the argument in it's graph will get a copy


# Directly creating Definition objects - check they have expected properties


def test_build_definition_1():
    # Definitions from objects
    obj = objects.TestClass1(10, test='a')
    definition = obj.definition
    assert isinstance(definition.cls, ImportRef)
    assert definition.cls.resolve() is objects.TestClass1
    assert definition['cls'] == definition.cls
    assert definition.args == ()
    assert definition.kwargs == {"x": 10, "test": "a"}
    assert definition.parameters["x"] == 10
    assert definition.parameters["test"] == "a"


def test_build_definition_2():
    # 1 nest object
    obj = objects.TestClass1(objects.TestClass1(10, test='b'), test='a')
    definition = obj.definition
    assert isinstance(definition.cls, ImportRef)
    assert definition.cls.resolve() is objects.TestClass1
    assert definition.args == ()
    assert definition.kwargs["test"] == "a"
    sub_def = definition.parameters["x"]
    assert type(sub_def) == ConcreteDefinition
    assert isinstance(sub_def.cls, ImportRef)
    assert sub_def.cls.resolve() is objects.TestClass1
    assert sub_def.args == ()
    assert sub_def.kwargs["x"] == 10
    assert sub_def.kwargs['test'] == 'b'


def test_build_definition_3():
    # with numpy array argument. algorithm should avoid it
    arr = np.random.random((2,2)).astype(np.float32)
    obj = objects.TestClass1(arr, test='a')
    definition = obj.definition
    assert isinstance(definition.cls, ImportRef)
    assert definition.cls.resolve() is objects.TestClass1
    assert definition.args == ()
    assert np.all(definition.parameters["x"] == arr)
    assert definition.kwargs['test'] == 'a'


def test_build_definition_4():
    # with numpy array argument. algorithm should avoid it
    # This time nested
    arr1 = np.random.random((2,2)).astype(np.float32)
    arr2 = np.random.random((2,2)).astype(np.float32)
    obj = objects.TestClass1(objects.TestClass1(arr2, test='b'), test=arr1)
    definition = obj.definition
    assert isinstance(definition.cls, ImportRef)
    assert definition.cls.resolve() is objects.TestClass1
    assert definition.args == ()
    assert np.all(definition.kwargs['test'] == arr1)
    sub_def = definition.parameters["x"]
    assert isinstance(sub_def.cls, ImportRef)
    assert sub_def.cls.resolve() is objects.TestClass1
    assert sub_def.args == ()
    assert np.all(sub_def.parameters["x"] == arr2)
    assert sub_def.kwargs['test'] == 'b'


def test_build_definition_5():
    # Test that definitions are properly instance cached
    obj1 = objects.TestClass1(10, test='a')
    obj2 = objects.TestClass1(obj1, test=obj1)
    assert obj2.x is obj1
    assert obj2.test is obj1
    def_2 = obj2.definition
    assert def_2.parameters["x"] is def_2.parameters["test"]


def test_build_definition_6():
    # Another instance caching test with deeper nesting
    obj1 = objects.TestClass1(10, test='a')
    obj2 = objects.TestClass1(20, test='b')
    obj3 = objects.TestClass1(obj1, test=obj2)
    obj4 = objects.TestClass1(obj3, test=obj2)
    assert obj3.x is obj1
    assert obj3.test is obj2
    assert obj4.test is obj2
    assert obj4.x is obj3
    def_4 = obj4.definition
    def_3 = def_4.parameters["x"]
    def_2 = def_4.parameters["test"]
    def_1 = def_3.parameters["x"]
    assert def_3.parameters["test"] == def_2
    assert def_3.parameters["x"] == def_1


class SemanticFixture(Object):
    def __init__(self, required, optional=2, *tail, keyword=3, **options):
        self.required = required
        self.optional = optional
        self.tail = tail
        self.keyword = keyword
        self.options = options


class SemanticCollisionFixture(Object):
    def __init__(self, cls, args, kwargs, build, stable_hash):
        self.values = (cls, args, kwargs, build, stable_hash)


def test_v2_concrete_definition_exposes_immutable_semantic_parameters():
    child = Definition(SemanticFixture, "child").concretize()
    cdef = Definition(SemanticFixture, child, 4, "tail", keyword=5, feature=True).concretize()

    assert cdef.parameters == FrozenDict({
        "required": child,
        "optional": 4,
        "tail": FrozenTuple(("tail",)),
        "keyword": 5,
        "options": FrozenDict({"feature": True}),
    })
    assert cdef.required is child
    assert cdef.required.required == "child"
    assert cdef.optional == 4
    assert cdef.tail == FrozenTuple(("tail",))
    assert cdef.options == FrozenDict({"feature": True})
    with pytest.raises(AttributeError):
        cdef.missing
    with pytest.raises(AttributeError):
        cdef.parameters.update({"required": "changed"})


def test_semantic_parameters_preserve_framework_member_collisions():
    cdef = Definition(
        SemanticCollisionFixture,
        "constructor-cls",
        "constructor-args",
        "constructor-kwargs",
        "constructor-build",
        "constructor-stable-hash",
    ).concretize()

    assert cdef.cls != "constructor-cls"
    assert cdef.args == FrozenTuple()
    assert cdef.kwargs == FrozenDict({
        "cls": "constructor-cls",
        "args": "constructor-args",
        "kwargs": "constructor-kwargs",
        "build": "constructor-build",
        "stable_hash": "constructor-stable-hash",
    })
    assert callable(cdef.build)
    assert callable(cdef.stable_hash)
    assert cdef.parameters == FrozenDict({
        "cls": "constructor-cls",
        "args": "constructor-args",
        "kwargs": "constructor-kwargs",
        "build": "constructor-build",
        "stable_hash": "constructor-stable-hash",
    })


def test_partial_definition_parameters_bind_only_supplied_fields(monkeypatch):
    partial = Definition(SemanticFixture, 1, keyword=4)
    skipped = Definition(SemanticFixture, SKIP_ARGS, keyword=4)
    unresolved = Definition(ImportRef("missing.semantic.fixture", "Fixture"), SKIP_ARGS, explicit=2)
    unresolved_positional = Definition(ImportRef("missing.semantic.fixture", "Fixture"), 1, explicit=2)

    assert partial.parameters == FrozenDict({"required": 1, "keyword": 4})
    assert partial.required == 1
    assert partial.keyword == 4
    assert not hasattr(partial, "optional")
    assert skipped.parameters == FrozenDict({"keyword": 4})
    assert unresolved.parameters == FrozenDict({"explicit": 2})
    assert unresolved.explicit == 2
    with pytest.raises(TypeError, match="positional"):
        unresolved_positional.parameters
    monkeypatch.setattr(ImportRef, "resolve", lambda self: pytest.fail("must not resolve symbols"))
    assert not hasattr(unresolved, "required")


def test_private_v1_concrete_definition_retains_raw_call_without_semantic_parameters():
    cdef = ConcreteDefinition._from_persisted_record(SemanticFixture, (1,), {"keyword": 4})

    assert cdef.args == FrozenTuple((1,))
    assert cdef.kwargs == FrozenDict({"keyword": 4})
    assert not hasattr(cdef, "parameters")
