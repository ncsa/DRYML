import numpy as np

import core_objects as objects
from dryml.core.definition import Definition, stable_hash_function


def test_definition_hash_1():
    definition1 = Definition(
        objects.TestClass1,
        10, test='a')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass1,
        10, test='a')

    def_hash_2 = stable_hash_function(definition2)

    assert def_hash1 == def_hash_2


def test_definition_hash_2():
    definition1 = Definition(
        objects.TestClass2,
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

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

    def_hash1 = stable_hash_function(definition1)

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

    def_hash2 = stable_hash_function(definition2)

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

    def_hash1 = stable_hash_function(definition1)

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

    def_hash2 = stable_hash_function(definition2)

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

    def_hash1 = stable_hash_function(definition1)

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

    def_hash2 = stable_hash_function(definition2)

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

    def_hash1 = stable_hash_function(definition1)

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

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 != def_hash2
