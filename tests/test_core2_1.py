import pytest
import core2_objects as objects 
from dryml.core2 import Definition, build_definition


def test_create_definition_1():
    Definition()


def test_create_definition_2():
    Definition({'cls': objects.TestClass1, 'args': [10], 'kwargs': {'test': 'a'}})


def test_create_definition_3():
    definition = Definition(objects.TestClass1, 10, test='a')
    assert definition.cls == objects.TestClass1
    assert definition['cls'] == objects.TestClass1
    assert len(definition.args) == 1
    assert definition.args[0] == 10
    assert len(definition.kwargs.keys()) == 1
    assert definition.kwargs['test'] == 'a'


def test_build_definition_1():
    obj = objects.TestClass1(10, test='a')
    definition = build_definition(obj)
    assert definition.cls == objects.TestClass1
    assert definition['cls'] == objects.TestClass1
    assert len(definition.args) == 1
    assert definition.args[0] == 10
    print(definition.kwargs)
    assert len(definition.kwargs.keys()) == 1
    assert definition.kwargs['test'] == 'a'
