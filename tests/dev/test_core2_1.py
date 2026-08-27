import pytest
import numpy as np
import core_objects as objects
from dryml.core.definition import Definition, \
    ConcreteDefinition, stable_hash_function, selector_match, \
    SKIP_ARGS
from dryml.core.repo import Repo, save_object, load_object
import os
import glob
from io import StringIO


# def test_definition_1():
#     # Test changing args directly on a Definition
#     # Shouldn't affect the original object

#     obj = objects.TestClass1(10, test='a')

#     definition = obj.definition

#     # We shouldn't be allowed to change a definition
#     definition.kwargs['test'] = 'b'
#     assert obj.definition.kwargs['test'] == 'a'


# def test_definition_2():
#     # Test changing kwargs directly on a Definition
#     # Shouldn't affect the original object
#     # This time we'll edit a collection argument

#     obj = objects.TestClass1([10], test=['a'])

#     definition = obj.definition

#     # We shouldn't be allowed to change a definition
#     definition.args[0][0] = 20
#     definition.kwargs['test'][0] = 'b'
#     assert obj.definition.args[0][0] == 10
#     assert obj.definition.kwargs['test'][0] == 'a'


# def test_definition_3():
#     # Test that we can build a definition from a Remember object
#     # and that it caches properly
#     obj = objects.TestClass1(10, test='a')

#     definition = obj.definition
#     assert id(definition) == id(obj.definition)

