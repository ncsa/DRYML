import pytest
import dryml
from dryml.core2.definition import Definition
from dryml.core2.repo import manage_repo
import io
import os
import sys
import time
import importlib

test_objs_text = """from dryml.core2 import Metadata, UniqueID


class SimpleObject(Metadata, UniqueID):
    def __init__(self, i, **kwargs):
        super().__init__(**kwargs)
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
    with open('./tests/objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import objs
    importlib.reload(objs)

    # Define simple class
    temp_buffer = io.BytesIO()
    obj = objs.SimpleObject(10)

    # Test that save to buffer works
    obj.save(repo=temp_buffer)

    temp_buffer.seek(0)

    obj2 = dryml.core2.load_object(repo=temp_buffer)

    # Test that restore from buffer creates identical object in this context.
    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1


@pytest.mark.usefixtures("create_name")
def test_save_object_2(create_name):
    """
    Test Saving objects to a file which doesn't yet exist
    """
    with open('./tests/objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import objs
    importlib.reload(objs)

    obj = objs.SimpleObject(10)

    file_name = ".".join([create_name, "dry"])
    obj.save(file_name)

    obj2 = dryml.core2.load_object(repo=file_name)

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1


@pytest.mark.usefixtures("create_temp_named_file")
def test_save_object_3(create_temp_named_file):
    """
    Test Saving objects to a file using file which was already created
    """
    with open('./tests/objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import objs
    importlib.reload(objs)

    obj = objs.SimpleObject(10)

    obj.save(create_temp_named_file)

    obj2 = dryml.core2.load_object(repo=create_temp_named_file)

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1


@pytest.mark.usefixtures("create_temp_file")
def test_save_object_4(create_temp_file):
    """
    Test Saving objects to a file using bytes-like file object
    """
    with open('./tests/objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import objs
    importlib.reload(objs)

    obj = objs.SimpleObject(10)

    obj.save(create_temp_file)

    create_temp_file.flush()
    create_temp_file.seek(0)
    obj2 = dryml.core2.load_object(repo=create_temp_file)

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
    with open('./tests/objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import objs
    importlib.reload(objs)

    # Create object and save
    obj = objs.SimpleObject(10)

    obj.save(create_temp_file)

    # Delete test_objs source and module from sys
    if os.path.exists('./tests/objs.py'):
        os.remove('./tests/objs.py')

    del objs
    if 'objs' in sys.modules:
        del sys.modules['objs']

    # Rewind file
    create_temp_file.flush()
    create_temp_file.seek(0)

    obj2 = dryml.core2.load_object(repo=create_temp_file)

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1


@pytest.mark.usefixtures("create_temp_named_file")
def test_save_object_6(create_temp_named_file):
    """
    Test object default metadata saving
    """
    with open('./tests/objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import objs
    importlib.reload(objs)

    desc_str = 'Test Description'
    obj = objs.SimpleObject(10, metadata={'description': desc_str})
    orig_creation_time = obj.definition.kwargs['metadata']['creation_time']

    obj.save(create_temp_named_file)

    obj2 = dryml.core2.load_object(repo=create_temp_named_file)

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1

    assert orig_creation_time == obj2.definition.kwargs['metadata']['creation_time']
    assert desc_str == obj2.definition.kwargs['metadata']['description']


def test_basic_object_def_update_1():
    def build_and_save_obj_1():
        time.sleep(1.1)
        with open('tests/objs.py', 'w') as f:
            f.write(test_objs_text.format(version=1))

        import objs
        importlib.reload(objs)

        obj = objs.SimpleObject(10)

        buffer = io.BytesIO()

        obj.save(buffer)

        return obj, buffer

    obj1, buffer = build_and_save_obj_1()

    buffer.seek(0)

    def build_obj_2(buffer):
        time.sleep(1.1)
        with open('tests/objs.py', 'w') as f:
            f.write(test_objs_text.format(version=2))
        # Sleep to invalidate the cache.
        import objs
        importlib.reload(objs)

        obj2 = dryml.core2.load_object(repo=buffer)

        return obj2

    obj2 = build_obj_2(buffer)

    assert obj1 == obj2

    assert obj1.version() == 1
    assert obj2.version() == 2


@pytest.mark.usefixtures("create_name")
def test_basic_object_def_update_2(create_name):
    def build_and_save_obj_1():
        time.sleep(1.1)
        with open('tests/objs.py', 'w') as f:
            f.write(test_objs_text.format(version=1))
        import objs
        importlib.reload(objs)

        obj = objs.SimpleObject(10)

        obj.save(create_name)

        return obj

    obj1 = build_and_save_obj_1()

    def build_obj_2():
        # Sleep to invalidate the cache.
        time.sleep(1.1)
        with open('tests/objs.py', 'w') as f:
            f.write(test_objs_text.format(version=2))

        import objs
        importlib.reload(objs)

        obj2 = dryml.core2.load_object(repo=create_name)

        return obj2

    obj2 = build_obj_2()

    assert obj1 == obj2

    assert obj1.version() == 1
    assert obj2.version() == 2





def test_object_config_1():
    import core2_objects as objs

    obj = objs.HelloStr(msg="Test")
    msg = obj.get_message()
    assert msg == "Hello! Test"

    obj = objs.HelloInt(msg=10)
    msg = obj.get_message()
    assert msg == "Hello! 10"


def test_object_hash_1():
    "Test that object hashes are unique within classes"
    import core2_objects as objs
    obj1 = objs.HelloStr(msg="Test")
    obj2 = objs.HelloStr(msg="Test")
    assert obj1.definition != \
        obj2.definition


def test_object_hash_2():
    "Test that object hashes are are same for two elements of the same class"
    import core2_objects as objs
    obj1 = objs.HelloStr(msg="Test")
    obj2 = objs.HelloStr(msg="Test")
    assert obj1.definition.categorical() == \
        obj2.definition.categorical()


@pytest.mark.usefixtures("create_name")
def test_object_hash_3(create_name):
    "Test that object hashes are the same after saving and restoring"
    import core2_objects as objs
    obj1 = objs.HelloStr(msg="Test")
    obj1.save(repo=create_name)

    obj2 = dryml.core2.load_object(repo=create_name)
    assert obj1.definition.categorical() == \
        obj2.definition.categorical()


@pytest.mark.usefixtures("create_name")
def test_object_hash_4(create_name):
    "Test that loaded objects are identical hash wise"
    import core2_objects as objs
    obj1 = objs.HelloStr(msg="Test")
    obj1.save(repo=create_name)

    obj2 = dryml.core2.load_object(repo=create_name)
    assert obj1.definition == \
        obj2.definition


# def test_change_obj_cls_1():
#     "Test that we can change an object's class"
#     import objects as objs
#     obj1 = objs.TestClassA(item=[5])
#     obj2 = dryml.change_object_cls(obj1, objs.TestClassA2)

#     assert type(obj2) is objs.TestClassA2
#     assert obj1.dry_kwargs['item'] == obj2.dry_kwargs['item']


# TODO: possibly redundant test
def test_object_def_1():
    import core2_objects as objs
    obj_def = Definition(objs.HelloInt, msg=10)
    other_def = Definition(
        objs.HelloInt,
        msg=10)

    assert obj_def.cls is other_def.cls
    assert obj_def.args == obj_def.args
    assert obj_def.kwargs == obj_def.kwargs


# TODO: possibly redundant test
def test_object_def_2():
    import core2_objects as objs
    obj_def = Definition(objs.HelloInt, msg=10)

    new_obj = obj_def.build()

    assert isinstance(new_obj, objs.HelloInt)
    assert new_obj.definition.kwargs['msg'] == 10





