import pytest
import dryml


@pytest.mark.usefixtures("create_name")
def test_basic_component_1(create_name):
    import objects

    test_obj = objects.HelloComponent(msg='test1')

    test_obj.save_self(create_name)

    test_obj2 = dryml.load_object(create_name)

    assert test_obj.definition().get_individual_id() == \
        test_obj2.definition().get_individual_id()

    assert test_obj2.dry_kwargs['msg'] == 'test1'

def test_basic_component_2():
    import objects

    test_obj = objects.HelloComponent(msg='test1')

    new_obj = dryml.change_object_cls(test_obj, cls=objects.HelloComponent, update=True)

    assert test_obj.definition() == new_obj.definition()
