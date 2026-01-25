import numpy as np
from copy import deepcopy
import core2_objects as objects
from dryml.core2.definition import Definition, ConcreteDefinition

### Tests for methods of creating definitions. We verify they have the intended properties.


# Directly creating Definition objects - Check they have expected properties

def test_create_definition_1():
    # Empty definition creation
    Definition()


def test_create_definition_2():
    # Plain single level definition
    definition = Definition(objects.TestClass1, 10, test='a')
    assert definition.cls == objects.TestClass1
    assert definition['cls'] == objects.TestClass1
    assert len(definition.args) == 1
    assert definition.args[0] == 10
    assert len(definition.kwargs.keys()) == 1
    assert definition.kwargs['test'] == 'a'


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
    assert definition.cls == objects.TestClass1
    assert definition['cls'] == objects.TestClass1
    assert len(definition.args) == 1
    assert definition.args[0] == 10
    assert len(definition.kwargs.keys()) == 1
    assert definition.kwargs['test'] == 'a'


def test_build_definition_2():
    # 1 nest object
    obj = objects.TestClass1(objects.TestClass1(10, test='b'), test='a')
    definition = obj.definition
    assert definition.cls == objects.TestClass1
    assert len(definition.args) == 1
    assert len(definition.kwargs.keys()) == 1
    assert definition.kwargs['test'] == 'a'
    sub_def = definition.args[0]
    assert type(sub_def) == ConcreteDefinition
    assert sub_def.cls == objects.TestClass1
    assert len(sub_def.args) == 1
    assert sub_def.args[0] == 10
    assert len(sub_def.kwargs.keys()) == 1
    assert sub_def.kwargs['test'] == 'b'


def test_build_definition_3():
    # with numpy array argument. algorithm should avoid it
    arr = np.random.random((2,2)).astype(np.float32)
    obj = objects.TestClass1(arr, test='a')
    definition = obj.definition
    assert definition.cls == objects.TestClass1
    assert len(definition.args) == 1
    assert len(definition.kwargs.keys()) == 1
    assert np.all(definition.args[0] == arr)
    assert definition.kwargs['test'] == 'a'


def test_build_definition_4():
    # with numpy array argument. algorithm should avoid it
    # This time nested
    arr1 = np.random.random((2,2)).astype(np.float32)
    arr2 = np.random.random((2,2)).astype(np.float32)
    obj = objects.TestClass1(objects.TestClass1(arr2, test='b'), test=arr1)
    definition = obj.definition
    assert definition.cls == objects.TestClass1
    assert len(definition.args) == 1
    assert len(definition.kwargs.keys()) == 1
    assert np.all(definition.kwargs['test'] == arr1)
    sub_def = definition.args[0]
    assert sub_def.cls == objects.TestClass1
    assert len(sub_def.args) == 1
    assert len(sub_def.kwargs.keys()) == 1
    assert np.all(sub_def.args[0] == arr2)
    assert sub_def.kwargs['test'] == 'b'


def test_build_definition_5():
    # Test that definitions are properly instance cached
    obj1 = objects.TestClass1(10, test='a')
    obj2 = objects.TestClass1(obj1, test=obj1)
    assert obj2.x is obj1
    assert obj2.test is obj1
    def_2 = obj2.definition
    assert def_2.args[0] is def_2.kwargs['test']


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
    def_3 = def_4.args[0]
    def_2 = def_4.kwargs['test']
    def_1 = def_3.args[0]
    assert def_3.kwargs['test'] == def_2
    assert def_3.args[0] == def_1
