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

    new_obj = dryml.change_object_cls(
        test_obj, cls=objects.HelloComponent, update=True)

    assert test_obj.definition() == new_obj.definition()


def test_component_def():
    import objects

    test_obj = objects.HelloComponentC('test1')

    test_obj2 = test_obj.definition().build()

    assert test_obj.definition() == test_obj2.definition()
    assert test_obj.dry_args[0] == test_obj2.dry_args[0]
