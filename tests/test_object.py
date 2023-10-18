import pytest
import dryml
from dryml.core2 import Definition
import zipfile
import io
import os
import sys
import time
import importlib

test_objs_text = """from dryml.core2 import Serializable, \
    Metadata, UniqueID


class SimpleObject(Serializable, Metadata, UniqueID):
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
    assert obj.save(temp_buffer)

    obj2 = dryml.core2.load_object(dest=temp_buffer)

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
    assert obj.save(file_name)

    obj2 = dryml.core2.load_object(dest=file_name)

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

    assert obj.save(create_temp_named_file)

    obj2 = dryml.core2.load_object(dest=create_temp_named_file)

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

    assert obj.save(create_temp_file)

    create_temp_file.flush()
    create_temp_file.seek(0)
    obj2 = dryml.core2.load_object(dest=create_temp_file)

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

    assert obj.save(create_temp_file)

    # Delete test_objs source and module from sys
    if os.path.exists('./tests/objs.py'):
        os.remove('./tests/objs.py')

    del objs
    if 'objs' in sys.modules:
        del sys.modules['objs']

    # Rewind file
    create_temp_file.flush()
    create_temp_file.seek(0)

    obj2 = dryml.core2.load_object(dest=create_temp_file)

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
    orig_creation_time = obj.__kwargs__['metadata']['creation_time']

    assert obj.save(create_temp_named_file)

    obj2 = dryml.core2.load_object(dest=create_temp_named_file)

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1

    assert orig_creation_time == obj2.__kwargs__['metadata']['creation_time']
    assert desc_str == obj2.__kwargs__['metadata']['description']


def test_basic_object_def_update_1():
    def build_and_save_obj_1():
        time.sleep(1.1)
        with open('tests/objs.py', 'w') as f:
            f.write(test_objs_text.format(version=1))

        import objs
        importlib.reload(objs)

        obj = objs.SimpleObject(10)

        from dryml.core2.definition import hash_value

        buffer = io.BytesIO()

        assert obj.save(buffer)

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

        from dryml.core2.definition import hash_value

        obj2 = dryml.core2.load_object(dest=buffer)

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

        assert obj.save(create_name)

        return obj

    obj1 = build_and_save_obj_1()

    def build_obj_2():
        # Sleep to invalidate the cache.
        time.sleep(1.1)
        with open('tests/objs.py', 'w') as f:
            f.write(test_objs_text.format(version=2))

        import objs
        importlib.reload(objs)

        obj2 = dryml.core2.load_object(dest=create_name)

        return obj2

    obj2 = build_obj_2()

    assert obj1 == obj2

    assert obj1.version() == 1
    assert obj2.version() == 2


def test_object_build_from_def_1():
    """
    Test that an object definition with no id results in an object with an id.
    """

    import core2_objects as objects
    obj = Definition(
        objects.TestClassB,
        1,
        base_msg='Test').build()

    assert 'uid' in obj.__kwargs__


def test_object_args_passing_1():
    import core2_objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    assert obj.__args__ == (1,)


@pytest.mark.usefixtures("create_name")
def test_object_args_passing_2(create_name):
    import core2_objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    dryml.core2.save_object(obj, dest=create_name)

    obj_loaded = dryml.core2.load_object(dest=create_name)

    assert obj_loaded.__args__ == (1,)


def test_object_args_passing_3():
    """
    Test passing nested dryobjects as arguments
    """
    import core2_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest(obj1)

    obj3 = objs.TestNest(obj2)

    obj1_def = obj1.definition

    obj1_cpy = obj1_def.build()

    assert obj1.definition == obj1_cpy.definition
    assert obj1 is not obj1_cpy

    obj2_def = obj2.definition

    obj2_cpy = obj2_def.build()

    assert obj2.definition == obj2_cpy.definition
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A) is type(obj1)
    assert type(obj2_cpy.A) is type(obj1)

    obj3_cpy = obj3.definition.build()

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

    obj1_cpy = obj1.definition.build()

    assert obj1.definition == obj1_cpy.definition
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build()

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

    obj1_cpy = obj1.definition.build()

    assert obj1.definition == obj1_cpy.definition
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build()

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

    obj1_cpy = obj1.definition.build()

    assert obj1.definition == obj1_cpy.definition
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build()

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

    obj1_cpy = obj1.definition.build()

    assert obj1.definition == obj1_cpy.definition
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build()

    assert obj2.definition == obj2_cpy.definition
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A['A'][0][0]) is type(obj1)
    assert type(obj2_cpy.A['A'][0][0]) is type(obj1)


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
    assert obj1.save(dest=create_name)

    obj2 = dryml.core2.load_object(dest=create_name)
    assert obj1.definition.categorical() == \
        obj2.definition.categorical()


@pytest.mark.usefixtures("create_name")
def test_object_hash_4(create_name):
    "Test that loaded objects are identical hash wise"
    import core2_objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save(dest=create_name)

    obj2 = dryml.core2.load_object(dest=create_name)
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
    assert new_obj.__kwargs__['msg'] == 10


# def test_object_fac_1():
#     import objects

#     obj_fac = dryml.ObjectFactory(
#         dryml.ObjectDef(objects.HelloInt, msg=10))

#     obj = obj_fac()

#     assert isinstance(obj, objects.HelloInt)
#     assert obj.dry_kwargs['msg'] == 10


# @pytest.mark.usefixtures("create_temp_named_file")
# def test_object_save_restore_1(create_temp_named_file):
#     """
#     We test save and restore of nested objects through arguments
#     """
#     import objects

#     # Create the data containing objects
#     data_obj1 = objects.TestClassC2(10)
#     data_obj1.set_val(20)

#     data_obj2 = objects.TestClassC2(20)
#     data_obj2.set_val(40)

#     # Enclose them in another object
#     obj = objects.TestClassC(data_obj1, B=data_obj2)

#     assert obj.save_self(create_temp_named_file)

#     obj2 = dryml.load_object(create_temp_named_file)

#     assert obj.definition() == obj2.definition()
#     assert obj.A.data == obj2.A.data
#     assert obj.B.data == obj2.B.data


# @pytest.mark.usefixtures("create_temp_named_file")
# def test_object_save_restore_2(create_temp_named_file):
#     """
#     We test save and restore of nested objects through arguments
#     This time, we make sure identical objects are loaded as
#     the same object.
#     """
#     import objects

#     # Create the data containing objects
#     data_obj1 = objects.TestClassC2(10)
#     data_obj1.set_val(20)

#     # Enclose them in another object
#     obj = objects.TestClassC(data_obj1, B=data_obj1)

#     assert obj.save_self(create_temp_named_file)

#     # Load the object from the file
#     obj2 = dryml.load_object(create_temp_named_file)

#     assert obj.definition() == obj2.definition()
#     assert obj.A is obj.B


# @pytest.mark.usefixtures("create_temp_named_file")
# def test_object_save_restore_3(create_temp_named_file):
#     """
#     We test save and restore of nested objects through arguments
#     Deeper nesting
#     """
#     import objects

#     # Create the data containing objects
#     data_obj1 = objects.TestClassC2(10)
#     data_obj1.set_val(20)

#     data_obj2 = objects.TestClassC2(20)
#     data_obj2.set_val(40)

#     data_obj3 = objects.TestClassC2('test')
#     data_obj3.set_val('test')

#     data_obj4 = objects.TestClassC2(0.5)
#     data_obj4.set_val(30.5)

#     obj1 = objects.TestClassC(data_obj1, B=data_obj2)
#     obj2 = objects.TestClassC(data_obj3, B=data_obj4)

#     # Enclose them in another object
#     obj = objects.TestClassC(obj1, B=obj2)

#     assert obj.save_self(create_temp_named_file)

#     # Load the object from the file
#     obj2 = dryml.load_object(create_temp_named_file)

#     assert obj.definition() == obj2.definition()
#     assert obj.A.A.data == obj2.A.A.data
#     assert obj.A.B.data == obj2.A.B.data
#     assert obj.B.A.data == obj2.B.A.data
#     assert obj.B.B.data == obj2.B.B.data


# def test_object_save_restore_4():
#     """
#     Test saving/restoring arguments/kwargs
#     """

#     import objects

#     # Create the data containing objects
#     data_obj1 = objects.TestClassC2(10)
#     data_obj1.set_val(20)

#     data_obj2 = objects.TestClassC2(20)
#     data_obj2.set_val(40)

#     data_obj3 = objects.TestClassC2('test')
#     data_obj3.set_val('test')

#     data_obj4 = objects.TestClassC2(0.5)
#     data_obj4.set_val(30.5)

#     obj1 = objects.TestClassC(data_obj1, B=data_obj2)
#     obj2 = objects.TestClassC(data_obj3, B=data_obj4)

#     from dryml.core.object import DryObjectPlaceholder, \
#         prep_args_kwargs, reconstruct_args_kwargs

#     args = (obj1, obj2)

#     (args, kwargs), ph = prep_args_kwargs(args, {})

#     assert type(args[0]) is DryObjectPlaceholder
#     assert type(args[1]) is DryObjectPlaceholder

#     reconstruct_args_kwargs(args, kwargs, ph)

#     assert type(args[0]) is objects.TestClassC
#     assert type(args[1]) is objects.TestClassC

#     assert obj1.A.data == args[0].A.data
#     assert obj1.B.data == args[0].B.data
#     assert obj2.A.data == args[1].A.data
#     assert obj2.B.data == args[1].B.data


# def test_object_save_restore_5():
#     """
#     Test saving/restoring arguments/kwargs
#     """

#     import objects

#     # Create the data containing objects
#     model_obj = objects.TestNest(10)
#     opt_obj = objects.TestNest3(20, model=model_obj)
#     loss_obj = objects.TestNest2(A='func')
#     train_fn_obj = objects.TestNest3(
#         optimizer=opt_obj,
#         loss=loss_obj,
#         epochs=10)

#     trainable_obj = objects.TestNest3(
#         model=model_obj,
#         train_fn=train_fn_obj
#     )

#     from dryml.core.object import DryObjectPlaceholder, \
#         prep_args_kwargs, reconstruct_args_kwargs

#     args = (trainable_obj,)

#     (args, kwargs), ph = prep_args_kwargs(args, {})

#     assert type(args[0]) is DryObjectPlaceholder

#     reconstruct_args_kwargs(args, kwargs, ph)

#     recon_trainable_obj = args[0]
#     assert type(recon_trainable_obj) is objects.TestNest3

#     assert recon_trainable_obj['model'] is \
#         recon_trainable_obj['train_fn']['optimizer']['model']
#     assert recon_trainable_obj['train_fn']['epochs'] == 10
#     assert recon_trainable_obj['model'].A == 10
#     assert recon_trainable_obj['train_fn']['optimizer'][0] == 20


# def test_nested_def_build_1():
#     """
#     Test nested definitions build appropriately.
#     """

#     import objects

#     data_def1 = dryml.ObjectDef(objects.TestNest2, A=1)
#     data_def2 = dryml.ObjectDef(objects.TestNest2, A=2)

#     data_def = dryml.ObjectDef(objects.TestClassC, data_def1, B=data_def1)
#     obj = data_def.build()
#     assert obj.A.A == 1
#     assert obj.B.A == 1
#     assert obj.A is obj.B

#     data_def = dryml.ObjectDef(objects.TestClassC, data_def2, B=data_def2)
#     obj = data_def.build()
#     assert obj.A.A == 2
#     assert obj.B.A == 2
#     assert obj.A is obj.B

#     data_def = dryml.ObjectDef(objects.TestClassC, data_def1, B=data_def2)
#     obj = data_def.build()
#     assert obj.A.A == 1
#     assert obj.B.A == 2
#     assert obj.A is not obj.B


# def test_build_crash_1():
#     import objects

#     # First wrong definition
#     test_def_1 = dryml.ObjectDef(objects.TestClassG1, 1, tmp='5')

#     try:
#         test_def_1.build()
#     except TypeError:
#         # This definition will throw a TypeError.
#         pass

#     print(dryml.core.config.build_cache)

#     # Second corrected definition
#     test_def_2 = dryml.ObjectDef(objects.TestClassG1, 1)

#     obj1 = test_def_2.build()
#     obj2 = test_def_2.build()

#     assert obj1.dry_id != obj2.dry_id
