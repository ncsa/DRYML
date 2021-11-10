import pytest
import dryml
import io
import os

def test_basic_object_1():
    import objects as objs

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
    import objects as objs

    obj = objs.SimpleObject(10)

    assert obj.save_self('test')

    obj2 = dryml.load_object('test')

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1

    os.remove('test.dry')

@pytest.mark.skip(reason="Currently, I don't know how to properly test updating an object definition")
def test_basic_object_def_update_1():
    def build_and_save_obj_1():
        import objects as objs

        obj = objs.SimpleObject(10)

        buffer = io.BytesIO()

        assert obj.save_self(buffer)

        return obj, buffer

    obj1, buffer = build_and_save_obj_1()

    def build_obj_2(buffer):
        import objects2 as objs

        obj2 = dryml.load_object(buffer)

        return obj2

    obj2 = build_obj_2(buffer)

    assert obj1 == obj2

    assert obj1.version() == 1
    assert obj2.version() == 2
