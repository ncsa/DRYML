import pytest
import dryml
import io
import os
import sys
import time
import importlib

test_objs_text = """import dryml


class SimpleObject(dryml.DryObject):
    def __init__(self, i, **kwargs):
        self.i = i

    def version(self):
        return {version}

    def __eq__(self, rhs):
        return self.i == rhs.i
"""


def test_save_object_1():
    """
    Test Saving objects through an io buffer
    """
    with open('./tests/test_objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import test_objs
    importlib.reload(test_objs)

    # Define simple class
    temp_buffer = io.BytesIO()
    obj = test_objs.SimpleObject(10)

    # Test that save to buffer works
    assert obj.save_self(temp_buffer)

    obj2 = dryml.load_object(temp_buffer)

    # Test that restore from buffer creates identical object in this context.
    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1


@pytest.mark.usefixtures("create_name")
def test_save_object_2(create_name):
    """
    Test Saving objects to a file which doesn't yet exist
    """
    with open('./tests/test_objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import test_objs
    importlib.reload(test_objs)

    obj = test_objs.SimpleObject(10)

    assert obj.save_self(create_name)

    obj2 = dryml.load_object(create_name)

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1


@pytest.mark.usefixtures("create_temp_named_file")
def test_save_object_3(create_temp_named_file):
    """
    Test Saving objects to a file using file which was already created
    """
    with open('./tests/test_objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import test_objs
    importlib.reload(test_objs)

    obj = test_objs.SimpleObject(10)

    assert obj.save_self(create_temp_named_file)

    obj2 = dryml.load_object(create_temp_named_file)

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1


@pytest.mark.usefixtures("create_temp_file")
def test_save_object_4(create_temp_file):
    """
    Test Saving objects to a file using bytes-like file object
    """
    with open('./tests/test_objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import test_objs
    importlib.reload(test_objs)

    obj = test_objs.SimpleObject(10)

    assert obj.save_self(create_temp_file)

    create_temp_file.flush()
    create_temp_file.seek(0)
    obj2 = dryml.load_object(create_temp_file)

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1


@pytest.mark.xfail
@pytest.mark.usefixtures("create_temp_file")
def test_save_object_5(create_temp_file):
    """
    Test Saving objects to a file, then loading in an environment
    without class definition
    """
    # This is currently not possible, or annoyingly difficult:
    # https://github.com/uqfoundation/dill/issues/128
    # Write test objects module, and load it.
    with open('./tests/test_objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import test_objs
    importlib.reload(test_objs)

    # Create object and save
    obj = test_objs.SimpleObject(10)

    assert obj.save_self(create_temp_file)

    # Delete test_objs source and module from sys
    if os.path.exists('./tests/test_objs.py'):
        os.remove('./tests/test_objs.py')

    del test_objs
    if 'test_objs' in sys.modules:
        del sys.modules['test_objs']

    # Rewind file
    create_temp_file.flush()
    create_temp_file.seek(0)

    obj2 = dryml.load_object(create_temp_file)

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1


def test_basic_object_def_update_1():
    def build_and_save_obj_1():
        time.sleep(1.1)
        with open('tests/test_objs.py', 'w') as f:
            f.write(test_objs_text.format(version=1))

        import test_objs
        importlib.reload(test_objs)

        obj = test_objs.SimpleObject(10)

        buffer = io.BytesIO()

        assert obj.save_self(buffer)

        return obj, buffer

    obj1, buffer = build_and_save_obj_1()

    def build_obj_2(buffer):
        time.sleep(1.1)
        with open('tests/test_objs.py', 'w') as f:
            f.write(test_objs_text.format(version=2))
        # Sleep to invalidate the cache.

        obj2 = dryml.load_object(buffer, update=True, reload=True)

        return obj2

    obj2 = build_obj_2(buffer)

    assert obj1 == obj2

    assert obj1.version() == 1
    assert obj2.version() == 2


@pytest.mark.usefixtures("create_name")
def test_basic_object_def_update_2(create_name):
    def build_and_save_obj_1():
        time.sleep(1.1)
        with open('tests/test_objs.py', 'w') as f:
            f.write(test_objs_text.format(version=1))
        import test_objs
        importlib.reload(test_objs)

        obj = test_objs.SimpleObject(10)

        assert obj.save_self(create_name)

        return obj

    obj1 = build_and_save_obj_1()

    def build_obj_2():
        # Sleep to invalidate the cache.
        time.sleep(1.1)
        with open('tests/test_objs.py', 'w') as f:
            f.write(test_objs_text.format(version=2))

        obj2 = dryml.load_object(create_name, update=True, reload=True)

        return obj2

    obj2 = build_obj_2()

    assert obj1 == obj2

    assert obj1.version() == 1
    assert obj2.version() == 2


def test_object_build_from_def_1():
    """
    Test that an object definition with no id results in an object with an id.
    """

    import objects
    obj = dryml.DryObjectDef.from_dict({
        'cls': objects.TestClassB,
        'dry_args': [1],
        'dry_kwargs': {'base_msg': 'Test'}}).build()

    assert 'dry_id' in obj.dry_kwargs


def test_object_args_passing_1():
    import objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    assert obj.dry_args == [1]


@pytest.mark.usefixtures("create_name")
def test_object_args_passing_2(create_name):
    import objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    dryml.save_object(obj, create_name)

    obj_loaded = dryml.load_object(create_name)

    assert obj_loaded.dry_args == [1]


def test_object_args_passing_3():
    """
    Test passing nested dryobjects as arguments
    """
    import objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest(obj1)

    obj3 = objs.TestNest(obj2)

    obj1_cpy = obj1.definition().build()

    assert obj1.definition() == obj1_cpy.definition()
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition().build()

    assert obj2.definition() == obj2_cpy.definition()
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A) is type(obj1)
    assert type(obj2_cpy.A) is type(obj1)

    obj3_cpy = obj3.definition().build()

    assert obj3.definition() == obj3_cpy.definition()
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
    import objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest([obj1])

    obj1_cpy = obj1.definition().build()

    assert obj1.definition() == obj1_cpy.definition()
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition().build()

    assert obj2.definition() == obj2_cpy.definition()
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A[0]) is type(obj1)
    assert type(obj2_cpy.A[0]) is type(obj1)


def test_object_args_passing_5():
    """
    Test passing nested dryobjects as arguments, within a nested list
    """
    import objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest([[obj1]])

    obj1_cpy = obj1.definition().build()

    assert obj1.definition() == obj1_cpy.definition()
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition().build()

    assert obj2.definition() == obj2_cpy.definition()
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A[0][0]) is type(obj1)
    assert type(obj2_cpy.A[0][0]) is type(obj1)


def test_object_args_passing_6():
    """
    Test passing nested dryobjects as arguments, within a dict
    """
    import objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest({'A': obj1})

    obj1_cpy = obj1.definition().build()

    assert obj1.definition() == obj1_cpy.definition()
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition().build()

    assert obj2.definition() == obj2_cpy.definition()
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A['A']) is type(obj1)
    assert type(obj2_cpy.A['A']) is type(obj1)


def test_object_args_passing_7():
    """
    Test passing nested dryobjects as arguments, within a dict with a list
    """
    import objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest({'A': [[obj1]]})

    obj1_cpy = obj1.definition().build()

    assert obj1.definition() == obj1_cpy.definition()
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition().build()

    assert obj2.definition() == obj2_cpy.definition()
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A['A'][0][0]) is type(obj1)
    assert type(obj2_cpy.A['A'][0][0]) is type(obj1)


def test_object_config_1():
    import objects as objs

    obj = objs.HelloStr(msg="Test")
    msg = obj.get_message()
    assert msg == "Hello! Test"

    obj = objs.HelloInt(msg=10)
    msg = obj.get_message()
    assert msg == "Hello! 10"


def test_object_hash_1():
    "Test that object hashes are unique within classes"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    obj2 = objs.HelloStr(msg="Test")
    assert obj1.definition().get_individual_id() != \
        obj2.definition().get_individual_id()


def test_object_hash_2():
    "Test that object hashes are are same for two elements of the same class"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    obj2 = objs.HelloStr(msg="Test")
    assert obj1.definition().get_category_id() == \
        obj2.definition().get_category_id()


@pytest.mark.usefixtures("create_name")
def test_object_hash_3(create_name):
    "Test that object hashes are the same after saving and restoring"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    obj2 = dryml.load_object(create_name)
    assert obj1.definition().get_category_id() == \
        obj2.definition().get_category_id()


@pytest.mark.usefixtures("create_name")
def test_object_hash_4(create_name):
    "Test that loaded objects are identical hash wise"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    obj2 = dryml.load_object(create_name)
    assert obj1.definition().get_individual_id() == \
        obj2.definition().get_individual_id()


@pytest.mark.usefixtures("create_name")
def test_object_file_hash_1(create_name):
    "Test that object hashes are the same after saving and restoring"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    with dryml.DryObjectFile(create_name) as dry_file:
        assert obj1.definition().get_category_id() == \
            dry_file.definition().get_category_id()


@pytest.mark.usefixtures("create_name")
def test_object_file_hash_2(create_name):
    "Test that loaded objects are identical hash wise"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    with dryml.DryObjectFile(create_name) as dry_file:
        assert obj1.definition().get_individual_id() == \
            dry_file.definition().get_individual_id()


@pytest.mark.usefixtures("create_name")
def test_object_file_hash_3(create_name):
    "Test that object and object factory hashes are the same"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    f = dryml.DryObjectFactory(dryml.DryObjectDef(
        objs.HelloStr, msg="Test"))

    assert obj1.definition().get_category_id() == \
        f.obj_def.get_category_id()


def test_change_obj_cls_1():
    "Test that we can change an object's class"
    import objects as objs
    obj1 = objs.TestClassA(item=[5])
    obj2 = dryml.change_object_cls(obj1, objs.TestClassA2)

    assert type(obj2) is objs.TestClassA2
    assert obj1.dry_kwargs['item'] == obj2.dry_kwargs['item']


def test_object_def_1():
    import objects
    obj_def = dryml.DryObjectDef(objects.HelloInt, msg=10)
    other_def = dryml.DryObjectDef.from_dict({
        'cls': 'objects.HelloInt',
        'dry_kwargs': {'msg': 10}
    })

    assert obj_def['cls'] is other_def['cls']
    assert obj_def['dry_args'] == obj_def['dry_args']
    assert obj_def['dry_kwargs'] == obj_def['dry_kwargs']


def test_object_def_2():
    import objects
    obj_def = dryml.DryObjectDef(objects.HelloInt, msg=10)

    new_obj = obj_def.build()

    assert isinstance(new_obj, objects.HelloInt)
    assert new_obj.dry_kwargs['msg'] == 10


def test_object_fac_1():
    import objects

    obj_fac = dryml.DryObjectFactory(
        dryml.DryObjectDef(objects.HelloInt, msg=10))

    obj = obj_fac()

    assert isinstance(obj, objects.HelloInt)
    assert obj.dry_kwargs['msg'] == 10


@pytest.mark.usefixtures("create_temp_named_file")
def test_object_save_restore_1(create_temp_named_file):
    """
    We test save and restore of nested objects through arguments
    """
    import objects

    # Create the data containing objects
    data_obj1 = objects.TestClassC2(10)
    data_obj1.set_val(20)

    data_obj2 = objects.TestClassC2(20)
    data_obj2.set_val(40)

    # Enclose them in another object
    obj = objects.TestClassC(data_obj1, B=data_obj2)

    assert obj.save_self(create_temp_named_file)

    # Load the object from the file
    obj2 = dryml.load_object(create_temp_named_file)

    assert obj.definition() == obj2.definition()
    assert obj.A.data == obj2.A.data
    assert obj.B.data == obj2.B.data


@pytest.mark.usefixtures("create_temp_named_file")
def test_object_save_restore_2(create_temp_named_file):
    """
    We test save and restore of nested objects through arguments
    This time, we make sure identical objects are loaded as
    the same object.
    """
    import objects

    # Create the data containing objects
    data_obj1 = objects.TestClassC2(10)
    data_obj1.set_val(20)

    # Enclose them in another object
    obj = objects.TestClassC(data_obj1, B=data_obj1)

    assert obj.save_self(create_temp_named_file)

    # Load the object from the file
    obj2 = dryml.load_object(create_temp_named_file)

    assert obj.definition() == obj2.definition()
    assert obj.A is obj.B


@pytest.mark.usefixtures("create_temp_named_file")
def test_object_save_restore_3(create_temp_named_file):
    """
    We test save and restore of nested objects through arguments
    Deeper nesting
    """
    import objects

    # Create the data containing objects
    data_obj1 = objects.TestClassC2(10)
    data_obj1.set_val(20)

    data_obj2 = objects.TestClassC2(20)
    data_obj2.set_val(40)

    data_obj3 = objects.TestClassC2('test')
    data_obj3.set_val('test')

    data_obj4 = objects.TestClassC2(0.5)
    data_obj4.set_val(30.5)

    obj1 = objects.TestClassC(data_obj1, B=data_obj2)
    obj2 = objects.TestClassC(data_obj3, B=data_obj4)

    # Enclose them in another object
    obj = objects.TestClassC(obj1, B=obj2)

    assert obj.save_self(create_temp_named_file)

    # Load the object from the file
    obj2 = dryml.load_object(create_temp_named_file)

    assert obj.definition() == obj2.definition()
    assert obj.A.A.data == obj2.A.A.data
    assert obj.A.B.data == obj2.A.B.data
    assert obj.B.A.data == obj2.B.A.data
    assert obj.B.B.data == obj2.B.B.data
