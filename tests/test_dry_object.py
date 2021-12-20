import pytest
import dryml
import io
import os
import uuid
import time
import importlib
import tempfile

test_objs_text = """import dryml


class SimpleObject(dryml.DryObject):
    def __init__(self, i=0, **kwargs):
        self.i = i
        dry_kwargs = {{
            'i': i
        }}
        super().__init__(
            dry_kwargs=dry_kwargs,
            **kwargs
        )

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True

    def version(self):
        return {version}

    def __eq__(self, rhs):
        return self.i == rhs.i
"""


def test_basic_object_1():
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


@pytest.fixture
def create_name():
    tempf = str(uuid.uuid4())
    yield tempf
    fullpath = f"{tempf}.dry"
    if os.path.exists(fullpath):
        os.remove(fullpath)


@pytest.fixture
def create_temp_named_file():
    with tempfile.NamedTemporaryFile(mode='wb') as f:
        yield f.name


@pytest.fixture
def create_temp_file():
    # We need to open with 'w+b' permission so that we can both
    # Read and write
    with tempfile.TemporaryFile(mode='w+b') as f:
        yield f


def test_basic_object_2(create_name):
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


def test_basic_object_3(create_temp_named_file):
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


def test_basic_object_4(create_temp_file):
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


def test_object_args_passing_1():
    import objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    assert obj.dry_args == [1]


def test_object_args_passing_2(create_name):
    import objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    dryml.save_object(obj, create_name)

    obj_loaded = dryml.load_object(create_name)

    assert obj_loaded.dry_args == [1]


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
    assert obj1.get_hash(no_id=False) != obj2.get_hash(no_id=False)


def test_object_hash_2():
    "Test that object hashes are are same for two elements of the same class"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    obj2 = objs.HelloStr(msg="Test")
    assert obj1.get_hash() == obj2.get_hash()


def test_object_hash_3(create_name):
    "Test that object hashes are the same after saving and restoring"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    obj2 = dryml.load_object(create_name)
    assert obj1.get_hash() == obj2.get_hash()


def test_object_hash_4(create_name):
    "Test that loaded objects are identical hash wise"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    obj2 = dryml.load_object(create_name)
    assert obj1.get_hash(no_id=False) == obj2.get_hash(no_id=False)


def test_object_hash_5(create_name):
    "Test that object hashes are the same after saving and restoring"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    obj2 = dryml.load_object(create_name)
    assert obj1.is_same_category(obj2)


def test_object_hash_6(create_name):
    "Test that loaded objects are identical hash wise"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    obj2 = dryml.load_object(create_name)
    assert obj1.is_identical(obj2)


def test_object_file_hash_1(create_name):
    "Test that object hashes are the same after saving and restoring"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    with dryml.DryObjectFile(create_name) as dry_file:
        assert obj1.get_hash() == dry_file.get_hash()


def test_object_file_hash_2(create_name):
    "Test that loaded objects are identical hash wise"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    with dryml.DryObjectFile(create_name) as dry_file:
        assert obj1.get_hash_str(no_id=False) == \
            dry_file.get_hash_str(no_id=False)


def test_object_file_hash_3(create_name):
    "Test that object and object factory hashes are the same"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self(create_name)

    f = dryml.DryObjectFactory(objs.HelloStr, msg="Test")

    assert obj1.get_hash() == f.get_hash()


def test_change_obj_cls_1():
    "Test that we can change an object's class"
    import objects as objs
    obj1 = objs.TestClassA(item=[5])
    obj2 = dryml.change_object_cls(obj1, objs.TestClassA2)

    assert type(obj2) is objs.TestClassA2
    assert obj1.dry_kwargs['item'] == obj2.dry_kwargs['item']
