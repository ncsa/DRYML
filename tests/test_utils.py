from dryml.core2.util import get_class_str, get_class_from_str, \
    get_unique_objects, apply_func
from dryml.core2.definition import Definition, SKIP_ARGS
import core2_objects as objects


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


def test_get_unique_objs_1():
    obj_f1_1 = objects.TestClassF1() 
    obj_f1_2 = objects.TestClassF1() 
    obj_f1_3 = objects.TestClassF1() 
    obj_c_1 = objects.TestClassC(
        obj_f1_2,
        B=obj_f1_3) 
    obj_c_2 = objects.TestClassC(
        obj_f1_1,
        B=obj_c_1)

    unique_objs = get_unique_objects(obj_c_2)

    unique_obj_definitions = set(map(lambda obj: obj.definition.concretize(), unique_objs.values()))

    assert len(unique_objs) == 5
    assert obj_f1_1.definition.concretize() in unique_obj_definitions
    assert obj_f1_2.definition.concretize() in unique_obj_definitions
    assert obj_f1_3.definition.concretize() in unique_obj_definitions
    assert obj_c_1.definition.concretize() in unique_obj_definitions
    assert obj_c_2.definition.concretize() in unique_obj_definitions


def test_apply_func_1():
    obj1 = objects.TestNest(objects.TestClassF1())

    def f(o):
        o.val = 20
    apply_func(obj1, f, sel=Definition(objects.TestClassF1, SKIP_ARGS))

    assert obj1.A.val == 20


def test_apply_func_2():
    obj1 = objects.TestClassC(
        objects.TestClassF1(),
        B=objects.TestClassC(
            objects.TestClassF1(),
            B=objects.TestClassF1()))

    def f(o):
        o.val = 20

    apply_func(obj1, f, sel=Definition(objects.TestClassF1, SKIP_ARGS))

    assert obj1.A.val == 20
    assert obj1.B.A.val == 20
    assert obj1.B.B.val == 20
    assert not hasattr(obj1, 'val')
    assert not hasattr(obj1.B, 'val')
