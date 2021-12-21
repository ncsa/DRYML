from dryml.utils import get_class_str, get_class_from_str
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
