from io import StringIO
import os

import core_objects as objects
import pytest
from dryml.core import Satisfies
from dryml.core.definition import Definition, SKIP_ARGS, selector_match
from dryml.core.symbol import ImportRef
import dryml.core as core


def test_selector_1():
    selector = [objects.TestClass1]
    definition = [objects.TestClass1]
    assert selector_match(selector, definition)


def test_selector_2():
    selector = Definition(
        objects.TestClass1,
        10,
        test='a')
    definition = selector
    obj = definition.build()

    assert selector(definition)
    assert selector(obj)


def test_selector_3():
    selector = Definition(
        objects.TestClass1,
        10)

    definition = Definition(
        objects.TestClass1,
        10,
        test='a')

    assert selector(definition)
    assert selector(definition)


def test_selector_4():
    with pytest.raises(TypeError, match="Anonymous function"):
        Definition(lambda x: x == objects.TestClass1, 10)


def test_selector_5():
    # Definitions with different numbers of arguments should be considered different.
    selector = Definition(
        objects.TestClass3,
        1, 2)

    definition = Definition(
        objects.TestClass3,
        1, 2, 3)

    assert not selector(definition)


def test_selector_6():
    sel_1 = Definition(
        objects.TestClass1, Satisfies(lambda x: x == 10, name="is-10"), test='a')
    def_2 = Definition(
        objects.TestClass1, 10, test='a')
    assert sel_1(def_2)


def test_selector_7():
    sel_1 = Definition(
        objects.TestClass1, 10, test=Satisfies(lambda x: x == 'a', name="is-a"))
    def_2 = Definition(
        objects.TestClass1, 10, test='a')
    assert sel_1(def_2)


def test_selector_8():
    def_1 = Definition(
        objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1, 20, test='a')
    sel_1 = Definition(
        objects.TestClass1, SKIP_ARGS, test='a')

    assert sel_1(def_1)
    assert sel_1(def_2)


def test_selector_9():
    obj_1 = objects.TestClass1(10, test='a')
    def_1 = Definition(
        objects.TestClass1, 20, test='b')
    assert not def_1(obj_1, verbose=True, full_diagnostic=True)


def test_selector_10():
    def_2 = Definition(objects.TestClass1, 10, test='a')
    def_1 = Definition(
        objects.TestClass1, 20, test='b')
    assert not def_1(def_2, verbose=True, full_diagnostic=True)


def test_selector_11():
    obj_1 = objects.TestClass1(10, test='a')
    def_1 = Definition(
        objects.TestClass2,
        test='a')

    assert not def_1(obj_1, verbose=True)


def test_selector_12():
    obj_1 = objects.TestClass1(
        10,
        test=objects.TestClass1(
            20,
            test=objects.TestClass1(30, test='c')))
    with core.definition_mode():
        def_1 = objects.TestClass2(
            10,
            test=objects.TestClass1(
                30,
                test=objects.TestClass2(
                    30,
                    test=Satisfies(lambda x: x != 'c', name="not-c")
                )
            )
        )
    assert not def_1(obj_1, verbose=True, full_diagnostic=True)


def test_selector_13():
    "Class selection"
    obj1 = objects.TestClassA(base_msg="Test1", item=5)
    obj2 = objects.TestClassB([1, 2, 3], base_msg="Test1")

    sel = Definition(objects.TestClassA, SKIP_ARGS)

    # Test selectors work with built classes
    assert sel(obj1)
    assert sel(obj1.definition)
    assert not sel(obj2)
    assert not sel(obj2.definition)

    core.save_object(obj1, repo='test1.dry')
    core.save_object(obj2, repo='test2.dry')

    # Test selectors work with loaded classes

    obj1_loaded = core.load_object(repo='test1.dry')
    obj2_loaded = core.load_object(repo='test2.dry')

    assert sel(obj1_loaded)
    assert not sel(obj2_loaded)

    os.remove('test1.dry')
    os.remove('test2.dry')


def test_selector_14():
    "args selection"
    obj1 = objects.TestClassB(1, base_msg="Test1")
    obj2 = objects.TestClassB([1, 2, 3], base_msg="Test2")

    sel = Definition(objects.TestClassB, 1)

    assert sel(obj1)
    assert sel(obj1.definition)
    sel(obj2, verbose=True)
    assert not sel(obj2)
    assert not sel(obj2.definition)


def test_selector_15():
    "kwargs selection"
    obj1 = objects.TestClassA(base_msg="Test1", item='a')
    obj2 = objects.TestClassA(base_msg="Test2", item=[10, 10, 10])

    sel = Definition(
        objects.TestClassA,
        item='a')

    assert sel(obj1)
    assert sel(obj1.definition)
    assert not sel(obj2)
    assert not sel(obj2.definition)

    sel = Definition(
        objects.TestClassA,
        item=[10, 10, 10])

    assert not sel(obj1)
    assert not sel(obj1.definition)
    assert sel(obj2)
    assert sel(obj2.definition)


def test_selector_16():
    "superclass selection"
    obj1 = objects.TestClassA(base_msg="Test1", item='a')
    obj2 = objects.TestClassA(base_msg="Test2",
                              item=[10, 10, 10])
    obj3 = objects.TestClassB(0, base_msg="Test3")
    obj4 = objects.TestClassB([10, 10], base_msg="Test4")
    obj5 = objects.HelloInt(msg=5)
    obj6 = objects.HelloInt(msg=20)
    obj7 = objects.HelloStr(msg='test')
    obj8 = objects.HelloStr(msg='2test')

    sel = Definition(
        objects.TestBase,
        SKIP_ARGS)

    assert sel(
        obj1,
        cls_str_compare=False,
        verbose=True)
    assert sel(
        obj1.definition,
        cls_str_compare=False,
        verbose=True)
    assert sel(
        obj2,
        cls_str_compare=False,
        verbose=True)
    assert sel(
        obj2.definition,
        cls_str_compare=False,
        verbose=True)
    assert sel(
        obj3,
        cls_str_compare=False,
        verbose=True)
    assert sel(
        obj3.definition,
        cls_str_compare=False,
        verbose=True)
    assert sel(
        obj4,
        cls_str_compare=False,
        verbose=True)
    assert sel(
        obj4.definition,
        cls_str_compare=False,
        verbose=True)
    assert not sel(
        obj5,
        cls_str_compare=False,
        verbose=True)
    assert not sel(
        obj5.definition,
        cls_str_compare=False,
        verbose=True)
    assert not sel(
        obj6,
        cls_str_compare=False,
        verbose=True)
    assert not sel(
        obj6.definition,
        cls_str_compare=False,
        verbose=True)
    assert not sel(
        obj7,
        cls_str_compare=False,
        verbose=True)
    assert not sel(
        obj7.definition,
        cls_str_compare=False,
        verbose=True)
    assert not sel(
        obj8,
        cls_str_compare=False,
        verbose=True)
    assert not sel(
        obj8.definition,
        cls_str_compare=False,
        verbose=True)

    sel = Definition(
        objects.HelloObject,
        SKIP_ARGS)

    assert not sel(
        obj1,
        cls_str_compare=False)
    assert not sel(
        obj1.definition,
        cls_str_compare=False)
    assert not sel(
        obj2,
        cls_str_compare=False)
    assert not sel(
        obj2.definition,
        cls_str_compare=False)
    assert not sel(
        obj3,
        cls_str_compare=False)
    assert not sel(
        obj3.definition,
        cls_str_compare=False)
    assert not sel(
        obj4,
        cls_str_compare=False)
    assert not sel(
        obj4.definition,
        cls_str_compare=False)
    assert sel(
        obj5,
        cls_str_compare=False)
    assert sel(
        obj5.definition,
        cls_str_compare=False)
    assert sel(
        obj6,
        cls_str_compare=False)
    assert sel(
        obj6.definition,
        cls_str_compare=False)
    assert sel(
        obj7,
        cls_str_compare=False)
    assert sel(
        obj7.definition,
        cls_str_compare=False)
    assert sel(
        obj8,
        cls_str_compare=False)
    assert sel(
        obj8.definition,
        cls_str_compare=False)


def test_selector_matches_live_class_against_import_ref():
    ref = ImportRef.from_object(objects.TestClass1)

    assert selector_match(objects.TestClass1, ref)
    assert selector_match(ref, objects.TestClass1)


def test_definition_selector_matches_concrete_definition_refs():
    selector = Definition(objects.TestClass1, 10, test="a")
    target = selector.concretize()

    assert isinstance(target.cls, ImportRef)
    assert selector(target)
    assert not target(selector)


def test_definition_selector_matches_class_arg_refs():
    selector = Definition(objects.TestClass1, objects.TestClass2)
    target = selector.concretize()

    assert isinstance(target.args[0], ImportRef)
    assert selector(target)
