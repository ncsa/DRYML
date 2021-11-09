import pytest
import dryml
import io
import objects
import os

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

def test_basic_object_2():
    obj = objects.SimpleObject(10)

    assert obj.save_self('test')

    obj2 = dryml.load_object('test')

    assert obj == obj2

    assert obj.version() == 1
    assert obj2.version() == 1

    os.remove('test.dry')
