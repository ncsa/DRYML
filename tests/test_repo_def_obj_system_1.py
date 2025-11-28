import pytest
import dryml
import os
import tempfile

from copy import deepcopy

import core2_objects_2 as objs
from dryml.core2.definition import Definition


def test_def_1():
    """
    Definitions can be created and they have the correct values
    """
    obj_def = objs.TestClassA.d(10)

    assert type(obj_def) is Definition

    assert obj_def.args[0] == 10


def test_def_2():
    """
    Definitions can be created, they have the correct values and they
    weren't copied yet.
    """
    val = objs.DeepcopyAware(10)
    assert val.counter == 0

    obj_def = objs.TestClassA.d(val)

    assert obj_def.args[0].counter == 0


def test_def_3():
    """
    Definitions can be created, they have the correct values.
    Test deepcopy and the DeepcopyAware object as well
    """
    val = objs.DeepcopyAware(10)
    assert val.counter == 0

    obj_def = objs.TestClassA.d(val)

    assert obj_def.args[0].counter == 0

    new_obj_def = deepcopy(obj_def)
    assert new_obj_def.args[0].counter == 1


def test_def_4():
    """
    When we concretize a definition, we have the same objects.
    """

    val = objs.DeepcopyAware(10)
    assert val.counter == 0

    obj_def = objs.TestClassA.d(val)

    obj = obj_def.build()

    assert obj.A.val == val.val

    assert obj_def.args[0].counter == 0
    # The definition should've been deep copied only a single time.
    assert obj.definition.args[0].counter == 1
    assert id(val) == id(obj.A)


def test_def_5():
    """
    Nested definition build
    """

    val1 = objs.DeepcopyAware(10)
    val2 = objs.DeepcopyAware(20)

    obj_def = objs.TestClassA.d((objs.TestClassA.d(val2), val1))

    obj = obj_def.build()

    assert isinstance(obj, objs.TestClassA)
    assert isinstance(obj.A[0], objs.TestClassA)

    assert id(val1) == id(obj.A[1])
    assert id(val2) == id(obj.A[0].A)
    assert obj.A[0].A.counter == 0
    assert obj.A[1].counter == 0
    assert obj.definition.args[0][1].counter == 1
    assert obj.definition.args[0][0].args[0].counter == 1 # Each object with the argument in it's graph will get a copy
