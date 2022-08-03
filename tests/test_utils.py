from dryml.utils import get_class_str, get_class_from_str, \
    apply_func
from dryml import DrySelector
import objects


def test_class_utils_1():
    cls = objects.HelloInt

    class_str = get_class_str(cls)

    cls_2 = get_class_from_str(class_str)

    assert cls is cls_2


def test_class_utils_2():
    obj = objects.HelloInt(msg=5)

    class_str = get_class_str(obj)

    cls_2 = get_class_from_str(class_str)

    assert type(obj) is cls_2


def test_apply_func_1():
    obj1 = objects.TestNest(objects.TestClassF1())

    def f(o):
        o.val = 20
    apply_func(obj1, f, sel=DrySelector(objects.TestClassF1))

    assert obj1.A.val == 20


def test_apply_func_2():
    obj1 = objects.TestClassC(
        objects.TestClassF1(),
        B=objects.TestClassC(
            objects.TestClassF1(),
            B=objects.TestClassF1()))

    def f(o):
        o.val = 20

    apply_func(obj1, f, sel=DrySelector(objects.TestClassF1))

    assert obj1.A.val == 20
    assert obj1.B.A.val == 20
    assert obj1.B.B.val == 20
    assert not hasattr(obj1, 'val')
    assert not hasattr(obj1.B, 'val')
