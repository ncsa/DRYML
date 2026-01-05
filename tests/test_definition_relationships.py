import core2_objects as objects
from dryml.core2.definition import Definition


def test_definition_eq_1():
    def_1 = Definition(
        objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1, 10, test='a')
    assert def_1 == def_2


def test_definition_eq_2():
    def_1 = Definition(
        objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1, 20, test='a')
    assert def_1 != def_2


def test_definition_eq_3():
    sel_1 = Definition(
        objects.TestClass1, lambda x: x == 10, test='a')
    def_2 = Definition(
        objects.TestClass1, 10, test='a')
    assert sel_1 != def_2
    assert def_2 != sel_1


def test_definition_eq_4():
    sel_1 = Definition(
        objects.TestClass1, 10, test=lambda x: x == 'a')
    def_2 = Definition(
        objects.TestClass1, 10, test='a')
    assert sel_1 != def_2
    assert def_2 != sel_1

