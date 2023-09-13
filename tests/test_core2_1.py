import pytest
import numpy as np
import core2_objects as objects
from dryml.core2 import Definition, build_definition, build_from_definition, hash_function, selector_match


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


def test_definition_hash_1():
    definition1 = Definition(
        objects.TestClass1,
        10, test='a')

    def_hash1 = hash_function(definition1)

    definition2 = Definition(
        objects.TestClass1,
        10, test='a')

    def_hash_2 = hash_function(definition2)

    assert def_hash1 == def_hash_2


def test_definition_hash_2():
    definition1 = Definition(
        objects.TestClass2,
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = hash_function(definition2)

    assert def_hash1 == def_hash2


def test_definition_hash_4():
    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='A'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = hash_function(definition2)

    assert def_hash1 == def_hash2


def test_definition_hash_5():
    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='B'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = hash_function(definition2)

    assert def_hash1 != def_hash2


def test_definition_hash_6():
    arr = np.random.random((10,10)).astype(np.float32)
    arr2 = np.copy(arr)

    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr2,
            test='A'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = hash_function(definition2)

    assert def_hash1 == def_hash2


def test_definition_hash_7():
    arr = np.random.random((10,10)).astype(np.float32)
    arr2 = np.copy(arr)
    arr2[0,0] = 5.

    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr2,
            test='A'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = hash_function(definition2)

    assert def_hash1 != def_hash2


def test_selector_1():
    selector = [objects.TestClass1]
    definition = [objects.TestClass1]
    assert selector_match(selector, definition)


def test_selector_2():
    selector = Definition(
        objects.TestClass1,
        10,
        test='a')
    definition = selector

    assert selector_match(selector, definition)


def test_selector_3():
    selector = Definition(
        objects.TestClass1,
        10)

    definition = Definition(
        objects.TestClass1,
        10,
        test='a')

    assert selector_match(selector, definition)


def test_selector_4():
    selector = Definition(
        lambda x: x == objects.TestClass1,
        10)

    definition = Definition(
        objects.TestClass1,
        10,
        test='a')

    assert selector_match(selector, definition)


def test_selector_5():
    # Definitions with different numbers of arguments should be considered different.
    selector = Definition(
        objects.TestClass3,
        1, 2)

    definition = Definition(
        objects.TestClass3,
        1, 2, 3)

    assert not selector_match(selector, definition)


def test_definition_1():
    # Test changing args directly on a Definition
    # Shouldn't affect the original object

    obj = objects.TestClass1(10, test='a')

    definition = build_definition(obj)

    # We shouldn't be allowed to change a definition
    definition.kwargs['test'] = 'b'
    assert obj.__kwargs__['test'] == 'a'


def test_definition_2():
    # Test changing kwargs directly on a Definition
    # Shouldn't affect the original object
    # This time we'll edit a collection argument

    obj = objects.TestClass1([10], test=['a'])

    definition = build_definition(obj)

    # We shouldn't be allowed to change a definition
    definition.args[0][0] = 20
    definition.kwargs['test'][0] = 'b'
    assert obj.__args__[0][0] == 10
    assert obj.__kwargs__['test'][0] == 'a'
