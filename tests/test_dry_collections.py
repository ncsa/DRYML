from dryml import DryList
import objects


def test_dry_list_1():
    obj1 = objects.HelloInt(msg=5)
    obj2 = objects.HelloStr(msg="a test")
    the_list = DryList(obj1, obj2)

    new_list = the_list.get_definition()()
    assert new_list[0].get_definition() == obj1.get_definition()
    assert new_list[1].get_definition() == obj2.get_definition()
