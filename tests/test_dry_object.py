import pytest
import dryml
import io
import os

test_objs_text = """
import dryml

class SimpleObject(dryml.DryObject):
    def __init__(self, i = 0, **kwargs):
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

    import test_objs as objs

    # Define simple class
    temp_buffer = io.BytesIO()
    obj = objs.SimpleObject(10)

    # Test that save to buffer works
    assert obj.save_self(temp_buffer)

    obj2 = dryml.load_object(temp_buffer)

    # Test that restore from buffer creates identical object in this context.
    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1

def test_basic_object_2():
    with open('./tests/test_objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    import test_objs as objs

    obj = objs.SimpleObject(10)

    assert obj.save_self('test')

    obj2 = dryml.load_object('test')

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1

    os.remove('test.dry')


@pytest.mark.skip(reason="Currently, I don't know how to properly test updating an object definition")
def test_basic_object_def_update_1():
    with open('test_objs.py', 'w') as f:
        f.write(test_objs_text.format(version=1))

    def build_and_save_obj_1():
        import test_objs as objs

        obj = objs.SimpleObject(10)

        buffer = io.BytesIO()

        assert obj.save_self(buffer)

        return obj, buffer

    obj1, buffer = build_and_save_obj_1()

    with open('test_objs.py', 'w') as f:
        f.write(test_objs_text.format(version=2))

    def build_obj_2(buffer):
        import test_objs as objs

        obj2 = dryml.load_object(buffer)

        return obj2

    obj2 = build_obj_2(buffer)

    assert obj1 == obj2

    assert obj1.version() == 1
    assert obj2.version() == 2

def test_object_args_passing_1():
    import objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    assert obj.dry_args == [1]

def test_object_args_passing_2():
    import objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    dryml.save_object(obj, 'test')

    obj_loaded = dryml.load_object('test')

    assert obj_loaded.dry_args == [1]

    os.remove('test.dry')

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

def test_object_hash_3():
    "Test that object hashes are the same after saving and restoring"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self('test')

    obj2 = dryml.load_object('test')
    assert obj1.get_hash() == obj2.get_hash()

def test_object_hash_4():
    "Test that loaded objects are identical hash wise"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self('test')

    obj2 = dryml.load_object('test')
    assert obj1.get_hash(no_id=False) == obj2.get_hash(no_id=False)

def test_object_hash_5():
    "Test that object hashes are the same after saving and restoring"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self('test')

    obj2 = dryml.load_object('test')
    assert obj1.is_same_category(obj2)

def test_object_hash_6():
    "Test that loaded objects are identical hash wise"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self('test')

    obj2 = dryml.load_object('test')
    assert obj1.is_identical(obj2)

def test_object_file_hash_1():
    "Test that object hashes are the same after saving and restoring"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self('test')

    with dryml.DryObjectFile('test') as dry_file:
        assert obj1.get_hash() == dry_file.get_hash()

def test_object_file_hash_2():
    "Test that loaded objects are identical hash wise"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self('test')

    with dryml.DryObjectFile('test') as dry_file:
        assert obj1.get_hash_str(no_id=False) == dry_file.get_hash_str(no_id=False)

def test_object_file_hash_1():
    "Test that object and object factory hashes are the same"
    import objects as objs
    obj1 = objs.HelloStr(msg="Test")
    assert obj1.save_self('test')

    f = dryml.DryObjectFactory(objs.HelloStr, msg="Test")

    assert obj1.get_hash() == f.get_hash()
