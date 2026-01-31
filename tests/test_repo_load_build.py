import pytest

import core2_objects as objects
from dryml.core2.definition import Definition
from dryml.core2.repo import Repo
from dryml.core2 import save_object, load_object


def test_build_from_definition_repo_1():
    # Test we can build one object
    definition = Definition(
        objects.TestClass1,
        10, test='a')

    repo = Repo() 
    obj = definition.build(repo=repo)
    assert repo._num_constructions == 1
    assert definition(obj.definition)
    assert definition(obj)


def test_build_from_definition_repo_2():
    # Test we can build a nested object
    definition = Definition(
        objects.TestClass1,
        Definition(
            objects.TestClass1,
            10,
            test='b'),
        test='a')

    repo = Repo() 
    obj = definition.build(repo=repo)
    assert repo._num_constructions == 2
    assert definition(obj.definition, verbose=True)
    assert definition(obj, verbose=True)


def test_build_from_definition_repo_3():
    # Test instance caching
    def_1 = Definition(
        objects.TestClass1,
        10,
        test='a')

    def_2 = Definition(
        objects.TestClass1,
        def_1,
        test=def_1)

    repo = Repo()

    obj = def_2.build(repo=repo)
    assert obj.x is obj.test
    assert repo._num_constructions == 2


def test_object_args_passing_1():
    import core2_objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    assert obj.definition.args == (1,)


@pytest.mark.usefixtures("create_name")
def test_object_args_passing_2(create_name):
    import core2_objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    save_object(obj, repo=create_name)

    obj_loaded = load_object(repo=create_name)

    assert obj_loaded.definition.args == (1,)


def test_object_args_passing_3():
    """
    Test passing nested dryobjects as arguments
    """
    import core2_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest(obj1)

    obj3 = objs.TestNest(obj2)

    obj1_def = obj1.definition

    obj1_cpy = obj1_def.build(instance="new", cache="none")

    assert obj1.definition == obj1_cpy.definition
    assert obj1 is not obj1_cpy

    obj2_def = obj2.definition

    obj2_cpy = obj2_def.build(instance="new", cache="none")

    assert obj2.definition == obj2_cpy.definition
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A) is type(obj1)
    assert type(obj2_cpy.A) is type(obj1)

    obj3_cpy = obj3.definition.build(instance="new", cache="none")

    assert obj3.definition == obj3_cpy.definition
    assert obj3 is not obj3_cpy
    assert obj3.A is not obj3_cpy.A
    assert type(obj3.A) is type(obj2)
    assert type(obj3_cpy.A) is type(obj2)
    assert obj3.A.A is not obj3_cpy.A.A
    assert type(obj3.A.A) is type(obj1)
    assert type(obj3_cpy.A.A) is type(obj1)


def test_object_args_passing_4():
    """
    Test passing nested dryobjects as arguments, within a list
    """
    import core2_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest([obj1])

    obj1_cpy = obj1.definition.build(instance="new", cache="none")

    assert obj1.definition == obj1_cpy.definition
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build(instance="new", cache="none")

    assert obj2.definition == obj2_cpy.definition
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A[0]) is type(obj1)
    assert type(obj2_cpy.A[0]) is type(obj1)


def test_object_args_passing_5():
    """
    Test passing nested dryobjects as arguments, within a nested list
    """
    import core2_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest([[obj1]])

    obj1_cpy = obj1.definition.build(instance="new", cache="none")

    assert obj1.definition == obj1_cpy.definition
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build(instance="new", cache="none")

    assert obj2.definition == obj2_cpy.definition
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A[0][0]) is type(obj1)
    assert type(obj2_cpy.A[0][0]) is type(obj1)


def test_object_args_passing_6():
    """
    Test passing nested dryobjects as arguments, within a dict
    """
    import core2_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest({'A': obj1})

    obj1_cpy = obj1.definition.build(instance="new", cache="none")

    assert obj1.definition == obj1_cpy.definition
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build(instance="new", cache="none")

    assert obj2.definition == obj2_cpy.definition
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A['A']) is type(obj1)
    assert type(obj2_cpy.A['A']) is type(obj1)


def test_object_args_passing_7():
    """
    Test passing nested dryobjects as arguments, within a dict with a list
    """
    import core2_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest({'A': [[obj1]]})

    obj1_cpy = obj1.definition.build(instance="new", cache="none")

    assert obj1.definition == obj1_cpy.definition
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build(instance="new", cache="none")

    assert obj2.definition == obj2_cpy.definition
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A['A'][0][0]) is type(obj1)
    assert type(obj2_cpy.A['A'][0][0]) is type(obj1)


