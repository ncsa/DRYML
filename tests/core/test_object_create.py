import numpy as np
import core2_objects as objects
from dryml.core2.definition import Definition
from dryml.core2.cdef_identity import V2_IDENTITY_VERSION
from dryml.core2.object import definition_mode
from dryml.core2.repo import Repo

### Tests for methods of creating objects. We verify they have the intended properties.


def test_build_from_definition_1():
    # 1 nest object
    definition = Definition(objects.TestClass1, 10, test='a')
    obj = definition.build()
    assert type(obj) == objects.TestClass1
    assert obj.test == 'a'
    assert obj.x == 10
    assert obj.definition.identity_version == V2_IDENTITY_VERSION


def test_all_public_exact_object_routes_emit_v2():
    repo = Repo()

    direct = objects.TestClass1(10, test="direct", repo=repo)
    built = Definition(objects.TestClass1, 11, test="build").build(repo=repo)
    loaded = repo.load_or_build(Definition(objects.TestClass1, 12, test="load-or-build"))
    with definition_mode(concrete=True):
        concrete = objects.TestClass1(13, test="concrete", repo=repo)

    assert direct.definition.identity_version == V2_IDENTITY_VERSION
    assert built.definition.identity_version == V2_IDENTITY_VERSION
    assert loaded.definition.identity_version == V2_IDENTITY_VERSION
    assert concrete.identity_version == V2_IDENTITY_VERSION


def test_build_from_definition_2():
    definition = Definition(
        objects.TestClass1,
        Definition(
            objects.TestClass1,
            10,
            test='b'),
        test='a')

    obj = definition.build()
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

    obj = definition.build()
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
    obj = definition.build()
    assert type(obj) == objects.TestClass1
    assert np.all(obj.test == arr1)
    assert type(obj.x) == objects.TestClass1
    assert np.all(obj.x.x == arr2)
    assert obj.x.test == 'b'


# def test_build_from_definition_5():
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


# def test_build_from_definition_6():
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


def test_build_from_definition_7():
    # Test instance caching with deeper nesting
    def_1 = Definition(
        objects.TestClass1,
        10,
        test='a')
    def_2 = Definition(
        objects.TestClass1,
        20,
        test='b')
    def_3 = Definition(
        objects.TestClass1,
        def_1,
        test=def_2)
    def_4 = Definition(
        objects.TestClass1,
        def_3,
        test=def_2)
    assert def_3 is def_4.args[0]
    assert def_2 is def_4.kwargs['test']
    assert def_1 is def_3.args[0]
    assert def_2 is def_3.kwargs['test']
    obj4 = def_4.build()
    assert def_4(obj4)
    obj3 = obj4.x
    obj2 = obj3.test
    assert obj4.test is obj2


def test_build_from_definition_8():
    """
    Test nested definitions build appropriately.
    """

    data_def1 = Definition(objects.TestNest2, A=1)
    data_def2 = Definition(objects.TestNest2, A=2)

    data_def = Definition(objects.TestClassC, data_def1, B=data_def1)
    obj = data_def.build()
    assert obj.A.A == 1
    assert obj.B.A == 1
    assert obj.A is obj.B

    data_def = Definition(objects.TestClassC, data_def2, B=data_def2)
    obj = data_def.build()
    assert obj.A.A == 2
    assert obj.B.A == 2
    assert obj.A is obj.B

    data_def = Definition(objects.TestClassC, data_def1, B=data_def2)
    obj = data_def.build()
    assert obj.A.A == 1
    assert obj.B.A == 2
    assert obj.A is not obj.B
