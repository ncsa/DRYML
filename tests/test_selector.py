import pytest
import dryml
import io
import os
import objects

def test_selector_1():
    "Class selection"
    obj1 = objects.TestClassA(base_msg="Test1", item=5)
    obj2 = objects.TestClassB([1,2,3], base_msg="Test1")

    sel = dryml.DrySelector(cls=objects.TestClassA)

    # Test selectors work with built classes
    assert sel(obj1)
    assert not sel(obj2)

    dryml.save_object(obj1, 'test1')
    dryml.save_object(obj2, 'test2')

    # Test selectors work with loaded classes

    obj1_loaded = dryml.load_object('test1')
    obj2_loaded = dryml.load_object('test2')

    assert sel(obj1_loaded)
    assert not sel(obj2_loaded)

    os.remove('test1.dry')
    os.remove('test2.dry')

def test_selector_2():
    "args selection"
    obj1 = objects.TestClassB(1, base_msg="Test1")
    obj2 = objects.TestClassB([1,2,3], base_msg="Test2")

    sel = dryml.DrySelector(cls=objects.TestClassB, args=[1])

    assert sel(obj1)
    assert not sel(obj2)
