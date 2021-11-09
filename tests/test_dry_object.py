import pytest
import dryml
import io
import objects

def test_basic_object_1():
    # Define simple class
    temp_buffer = io.BytesIO()
    obj = objects.SimpleObject(10)

    # Test that save to buffer works
    assert obj.save_self(temp_buffer)

    obj2 = dryml.load_object(temp_buffer)

    # Test that restore from buffer creates identical object in this context.
    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1
