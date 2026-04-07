from dryml.core2.utils.general import get_class_str, get_class_from_str, \
    get_unique_objects, get_unique_concrete_definitions
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


def test_list_unique_objs_1():
    obj_f1_1 = objects.TestClassF1() 
    obj_f1_2 = objects.TestClassF1() 
    obj_f1_3 = objects.TestClassF1() 
    obj_c_1 = objects.TestClassC(
        obj_f1_2,
        B=obj_f1_3) 
    obj_c_2 = objects.TestClassC(
        obj_f1_1,
        B=obj_c_1)

    unique_obj_definitions = set(get_unique_concrete_definitions(obj_c_2))

    assert len(unique_obj_definitions) == 5
    assert obj_f1_1.definition in unique_obj_definitions
    assert obj_f1_2.definition in unique_obj_definitions
    assert obj_f1_3.definition in unique_obj_definitions
    assert obj_c_1.definition in unique_obj_definitions
    assert obj_c_2.definition in unique_obj_definitions
