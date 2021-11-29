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

def test_object_config_1():
    import objects as objs

    obj = objs.HelloStr(msg="Test")
    msg = obj.get_message()
    assert msg == "Hello! Test"

    obj = objs.HelloInt(msg=10)
    msg = obj.get_message()
    assert msg == "Hello! 10"
