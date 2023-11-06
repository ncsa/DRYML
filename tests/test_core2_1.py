import pytest
import numpy as np
import core2_objects as objects
from dryml.core2.definition import Definition, \
    ConcreteDefinition, hash_function, selector_match, \
    SKIP_ARGS
from dryml.core2.repo import Repo, save_object, load_object
import os
from io import StringIO


def test_create_definition_1():
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


def test_build_definition_1():
    # Plain single level object
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
    assert def_3.kwargs['test'] is def_2
    assert def_3.args[0] is def_1


def test_build_from_definition_1():
    # 1 nest object
    definition = Definition(objects.TestClass1, 10, test='a')
    obj = definition.build()
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


def test_build_from_definition_5():
    # Test we can build one object
    definition = Definition(
        objects.TestClass1,
        10, test='a')

    repo = Repo() 
    obj = definition.build(repo=repo)
    assert repo._num_constructions == 1
    assert definition(obj.definition)
    assert definition(obj)


def test_build_from_definition_6():
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
    assert definition(obj.definition)
    assert definition(obj)


def test_build_from_definition_7():
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


def test_build_from_definition_8():
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
    obj = definition.build()

    assert selector(definition)
    assert selector(obj)


def test_selector_3():
    selector = Definition(
        objects.TestClass1,
        10)

    definition = Definition(
        objects.TestClass1,
        10,
        test='a')

    assert selector(definition)
    assert selector(definition)


def test_selector_4():
    selector = Definition(
        lambda x: x == objects.TestClass1,
        10)

    definition = Definition(
        objects.TestClass1,
        10,
        test='a')

    assert selector(definition)


def test_selector_5():
    # Definitions with different numbers of arguments should be considered different.
    selector = Definition(
        objects.TestClass3,
        1, 2)

    definition = Definition(
        objects.TestClass3,
        1, 2, 3)

    assert not selector(definition)


def test_selector_6():
    sel_1 = Definition(
        objects.TestClass1, lambda x: x == 10, test='a')
    def_2 = Definition(
        objects.TestClass1, 10, test='a')
    assert sel_1(def_2)


def test_selector_7():
    sel_1 = Definition(
        objects.TestClass1, 10, test=lambda x: x == 'a')
    def_2 = Definition(
        objects.TestClass1, 10, test='a')
    assert sel_1(def_2)


def test_selector_8():
    def_1 = Definition(
        objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1, 20, test='a')
    sel_1 = Definition(
        objects.TestClass1, SKIP_ARGS, test='a')

    assert sel_1(def_1)
    assert sel_1(def_2)


def test_selector_9():
    obj_1 = objects.TestClass1(10, test='a')
    def_1 = Definition(
        objects.TestClass1, 20, test='b')
    temp_stream = StringIO()
    assert not def_1(obj_1, verbose=True, output_stream=temp_stream)
    temp_stream.seek(0)
    stream_text = temp_stream.read()
    assert "[root/args/0]: Values differ" in stream_text
    assert "[root/kwargs/test]: Values differ" in stream_text


def test_selector_10():
    def_2 = Definition(objects.TestClass1, 10, test='a')
    def_1 = Definition(
        objects.TestClass1, 20, test='b')
    temp_stream = StringIO()
    assert not def_1(def_2, verbose=True, output_stream=temp_stream)
    temp_stream.seek(0)
    stream_text = temp_stream.read()
    assert "[root/args/0]: Values differ" in stream_text
    assert "[root/kwargs/test]: Values differ" in stream_text


def test_selector_11():
    obj_1 = objects.TestClass1(10, test='a')
    def_1 = Definition(
        objects.TestClass2,
        test='a')

    temp_stream = StringIO()
    assert not def_1(obj_1, verbose=True, output_stream=temp_stream)
    temp_stream.seek(0)
    stream_text = temp_stream.read()
    assert "[root/cls]: core2_objects.TestClass1 is not a subclass of core2_objects.TestClass2" in stream_text


def test_selector_12():
    obj_1 = objects.TestClass1(
        10,
        test=objects.TestClass1(
            20,
            test=objects.TestClass1(30, test='c')))
    def_1 = Definition(
        objects.TestClass2,
        10,
        test=Definition(
            objects.TestClass1,
            30,
            test=Definition(
                objects.TestClass2,
                30,
                test=lambda x: x != 'c')))
    temp_stream = StringIO()
    assert not def_1(obj_1, verbose=True, output_stream=temp_stream)
    temp_stream.seek(0)
    stream_text = temp_stream.read()
    assert "[root/cls]: core2_objects.TestClass1 is not a subclass of core2_objects.TestClass2" in stream_text
    assert "[root/kwargs/test/args/0]: Values differ" in stream_text
    assert "[root/kwargs/test/kwargs/test/cls]: core2_objects.TestClass1 is not a subclass of core2_objects.TestClass2" in stream_text
    assert "[root/kwargs/test/kwargs/test/kwargs/test]: Callable test failed" in stream_text


def test_definition_1():
    # Test changing args directly on a Definition
    # Shouldn't affect the original object

    obj = objects.TestClass1(10, test='a')

    definition = obj.definition

    # We shouldn't be allowed to change a definition
    definition.kwargs['test'] = 'b'
    assert obj.__kwargs__['test'] == 'a'


def test_definition_2():
    # Test changing kwargs directly on a Definition
    # Shouldn't affect the original object
    # This time we'll edit a collection argument

    obj = objects.TestClass1([10], test=['a'])

    definition = obj.definition

    # We shouldn't be allowed to change a definition
    definition.args[0][0] = 20
    definition.kwargs['test'][0] = 'b'
    assert obj.__args__[0][0] == 10
    assert obj.__kwargs__['test'][0] == 'a'


def test_definition_3():
    # Test that we can build a definition from a Remember object
    # and that it caches properly
    obj = objects.TestClass1(10, test='a')

    definition = obj.definition
    assert id(definition) == id(obj.definition)


def test_definition_eq_1():
    def_1 = Definition(
        objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1, 10, test='a')
    assert def_1 == def_2


def test_definition_eq_2():
    def_1 = Definition(
        objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1, 20, test='a')
    assert def_1 != def_2


def test_definition_eq_3():
    sel_1 = Definition(
        objects.TestClass1, lambda x: x == 10, test='a')
    def_2 = Definition(
        objects.TestClass1, 10, test='a')
    assert sel_1 != def_2
    assert def_2 != sel_1


def test_definition_eq_4():
    sel_1 = Definition(
        objects.TestClass1, 10, test=lambda x: x == 'a')
    def_2 = Definition(
        objects.TestClass1, 10, test='a')
    assert sel_1 != def_2
    assert def_2 != sel_1


def test_definition_concrete_1():
    definition = Definition(
        objects.TestClass1,
        10,
        test='a')

    new_def = definition.concretize()
    assert definition != new_def
    assert type(new_def) is ConcreteDefinition


def test_definition_concrete_2():
    definition = Definition(
        objects.TestClass4,
        10,
        test='a')

    new_def = definition.concretize()
    assert selector_match(definition, new_def)
    assert definition != new_def
    assert type(new_def) is ConcreteDefinition
    assert 'uid' in new_def.kwargs
    assert 'metadata' in new_def.kwargs


def test_definition_concrete_3():
    # Test that concrete definition is hashable
    definition = Definition(objects.TestClass1, 10, test='a').concretize()
    hash(definition)


def test_definition_concrete_4():
    # Test that the same Definition objects product identical ConcreteDefinition objects after concretization
    def_1 = Definition(objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1,
        def_1,
        test=def_1)

    concrete_def = def_2.concretize()

    conc_def_1 = concrete_def.kwargs['test']

    assert concrete_def.args[0] is conc_def_1


def test_definition_concrete_5():
    # Test that the same Definition objects produce identical ConcreteDefinition objects after concretization even after the original Definition has been deepcopied
    def_1 = Definition(objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1,
        def_1,
        test=def_1)

    def_3 = def_2.copy()

    conc_def = def_3.concretize()

    conc_def_1 = conc_def.kwargs['test']

    assert conc_def.args[0] is conc_def_1


def test_definition_concrete_6():
    def_1 = Definition(objects.TestClass1, SKIP_ARGS, test='a')

    flag = True
    try:
        def_1.concretize()
        flag = False
    except ValueError:
        pass

    assert flag


@pytest.mark.usefixtures("create_temp_dir")
def test_save_load_1(create_temp_dir):
    # Test save/load to/from a directory
    obj1 = objects.TestClass5(10, test='a')

    save_object(obj1, dest=create_temp_dir)

    assert len(os.listdir(create_temp_dir)) == 2
    assert len(os.listdir(os.path.join(create_temp_dir, 'objects'))) == 1

    obj1_2 = load_object(obj1.definition, dest=create_temp_dir)
    assert obj1_2.x == 10
    assert obj1_2.test == 'a'


@pytest.mark.usefixtures("create_temp_dir")
def test_save_load_2(create_temp_dir):
    # Test save/load to/from a directory
    obj1 = objects.TestClass5(10, test='a')
    obj2 = objects.TestClass5(20, test='b')
    obj3 = objects.TestClass5(obj1, test=obj2)
    obj4 = objects.TestClass5(obj3, test=obj2)
    assert obj3.x is obj1
    assert obj3.test is obj2
    assert obj4.test is obj2
    assert obj4.x is obj3

    save_object(obj4, dest=create_temp_dir)

    assert len(os.listdir(create_temp_dir)) == 2
    assert len(os.listdir(os.path.join(create_temp_dir, 'objects'))) == 4

    obj4_2 = load_object(obj4.definition, dest=create_temp_dir)
    assert obj4_2 is not obj4
    obj3_2 = obj4_2.x
    obj2_2 = obj4_2.test
    assert obj3_2.test is obj2_2
    assert obj4


@pytest.mark.usefixtures("create_temp_dir")
def test_save_load_3(create_temp_dir):
    # Test save/load to/from a directory another nested object
    obj1 = objects.TestClass5(10, test='a')
    obj2 = objects.TestClass5(20, test='b')
    obj3 = objects.TestClass5(30, test='c')
    obj4 = objects.TestClass5(40, test='d')

    obj5 = objects.TestClass5(obj1, test=obj2)
    obj6 = objects.TestClass5(obj2, test=obj3)
    obj7 = objects.TestClass5(obj3, test=obj4)

    obj8 = objects.TestClass5(obj5, test=obj6)
    obj9 = objects.TestClass5(obj6, test=obj7)

    obj10 = objects.TestClass5(obj8, test=obj9)

    obj11 = objects.TestClass5(obj10, test=obj10)

    save_object(obj11, dest=create_temp_dir)

    assert len(os.listdir(create_temp_dir)) == 2
    assert len(os.listdir(os.path.join(create_temp_dir, 'objects'))) == 11

    obj11_2 = load_object(obj11.definition, dest=create_temp_dir)
    obj10_2 = obj11_2.x
    assert obj11_2.test is obj10_2
    obj6_2 = obj10_2.x.test
    assert obj6_2 is obj10_2.test.x
    obj2_2 = obj6_2.x
    assert obj2_2 is obj10_2.x.x.test
    obj3_2 = obj6_2.test
    assert obj3_2 is obj10_2.test.test.x


def test_defer_1():
    # Test that mock initialization properly fails case 1
    try:
        objects.TestDefer1(test=10)
        assert False
    except TypeError:
        pass

    # Test that mock initialization properly fails case 1
    try:
        objects.TestDefer1(5, test=10)
        assert False
    except TypeError:
        pass

    defer_obj = objects.TestDefer1(5)

    assert len(defer_obj.__dict__) == 5

    # Trigger deferred initialization
    assert defer_obj.x == 5

    # Check for new attribute
    assert len(defer_obj.__dict__) == 6
