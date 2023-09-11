import pytest
import numpy as np
import core2_objects as objects
from dryml.core2 import Definition, build_definition, build_from_definition


def test_create_definition_1():
    Definition()


def test_create_definition_2():
    Definition({'cls': objects.TestClass1, 'args': [10], 'kwargs': {'test': 'a'}})


def test_create_definition_3():
    # Plain single level definition
    definition = Definition(objects.TestClass1, 10, test='a')
    assert definition.cls == objects.TestClass1
    assert definition['cls'] == objects.TestClass1
    assert len(definition.args) == 1
    assert definition.args[0] == 10
    assert len(definition.kwargs.keys()) == 1
    assert definition.kwargs['test'] == 'a'


def test_build_definition_1():
    # Plain single level object
    obj = objects.TestClass1(10, test='a')
    definition = build_definition(obj)
    assert definition.cls == objects.TestClass1
    assert definition['cls'] == objects.TestClass1
    assert len(definition.args) == 1
    assert definition.args[0] == 10
    assert len(definition.kwargs.keys()) == 1
    assert definition.kwargs['test'] == 'a'


def test_build_definition_2():
    # 1 nest object
    obj = objects.TestClass1(objects.TestClass1(10, test='b'), test='a')
    definition = build_definition(obj)
    assert definition.cls == objects.TestClass1
    assert len(definition.args) == 1
    assert len(definition.kwargs.keys()) == 1
    assert definition.kwargs['test'] == 'a'
    sub_def = definition.args[0]
    assert type(sub_def) == Definition
    assert sub_def.cls == objects.TestClass1
    assert len(sub_def.args) == 1
    assert sub_def.args[0] == 10
    assert len(sub_def.kwargs.keys()) == 1
    assert sub_def.kwargs['test'] == 'b'


def test_build_definition_3():
    # with numpy array argument. algorithm should avoid it
    arr = np.random.random((2,2)).astype(np.float32)
    obj = objects.TestClass1(arr, test='a')
    definition = build_definition(obj)
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
    definition = build_definition(obj)
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


def test_build_from_definition_1():
    # 1 nest object
    definition = Definition(objects.TestClass1, 10, test='a')
    obj = build_from_definition(definition)
    assert type(obj) == objects.TestClass1
    assert obj.test == 'a'
    assert obj.x == 10


def test_build_from_definition_2():
    definition = Definition(
        objects.TestClass1,
        Definition(
            objects.TestClass1,
            10,
            test='b'),
        test='a')

    obj = build_from_definition(definition)
    assert type(obj) == objects.TestClass1
    assert obj.test == 'a'
    assert type(obj.x) == objects.TestClass1
    assert obj.x.test == 'b'
    assert obj.x.x == 10


def test_build_from_definition_3():
    # with numpy array
    arr = np.random.random((2,2)).astype(np.float32)
    definition = Definition(
        objects.TestClass1,
        arr, test='a')

    obj = build_from_definition(definition)
    assert type(obj) == objects.TestClass1
    assert np.all(obj.x == arr)
    assert obj.test == 'a'

def test_build_from_definition_4():
    arr1 = np.random.random((2,2)).astype(np.float32)
    arr2 = np.random.random((2,2)).astype(np.float32)
    definition = Definition(
        objects.TestClass1,
        Definition(
            objects.TestClass1,
            arr2,
            test='b'),
        test=arr1)
    obj = build_from_definition(definition)
    assert type(obj) == objects.TestClass1
    assert np.all(obj.test == arr1)
    assert type(obj.x) == objects.TestClass1
    assert np.all(obj.x.x == arr2)
    assert obj.x.test == 'b'
