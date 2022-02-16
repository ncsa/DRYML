import pytest
import dryml


@pytest.mark.usefixtures("create_name")
def test_basic_trainable_1(create_name):
    import objects

    test_obj = objects.HelloTrainable(msg='test1')

    test_obj.save_self(create_name)

    test_obj2 = dryml.load_object(create_name)

    assert test_obj.definition().get_individual_id() == \
        test_obj2.definition().get_individual_id()

    assert test_obj2.dry_kwargs['msg'] == 'test1'


def test_basic_trainable_2():
    import objects

    test_obj = objects.HelloTrainable(msg='test1')

    new_obj = dryml.change_object_cls(
        test_obj, cls=objects.HelloTrainable, update=True)

    assert test_obj.definition() == new_obj.definition()


def test_trainable_def_1():
    import objects

    test_obj = objects.HelloTrainableC('test1', description='test obj')

    test_obj2 = test_obj.definition().build()

    assert test_obj.definition() == test_obj2.definition()
    assert test_obj.dry_args[0] == test_obj2.dry_args[0]


def test_trainable_def_2():
    import objects

    test_obj = objects.HelloTrainableC(objects.HelloStr(msg='test'))

    test_obj2 = test_obj.definition().build()

    assert test_obj.definition() == test_obj2.definition()
    assert type(test_obj.A) is objects.HelloStr
    assert type(test_obj2.A) is objects.HelloStr
    assert test_obj.dry_args[0] == test_obj2.dry_args[0]
    assert test_obj.A.definition() == test_obj2.A.definition()
    assert test_obj.A.str_message == test_obj2.A.str_message


def test_trainable_def_3():
    import objects

    test_obj = objects.HelloTrainableD(A='test1')

    assert test_obj.A == 'test1'

    test_obj2 = test_obj.definition().build()

    assert test_obj.definition() == test_obj2.definition()
    assert test_obj.dry_kwargs['A'] == 'test1'
    assert test_obj2.dry_kwargs['A'] == 'test1'
    assert type(test_obj.A) is str
    assert type(test_obj2.A) is str
    assert test_obj.A == test_obj2.A


def test_trainable_def_4():
    import objects

    test_obj = objects.HelloTrainableD(A=objects.HelloStr(msg='test'))

    test_obj2 = test_obj.definition().build()

    assert test_obj.definition() == test_obj2.definition()
    assert type(test_obj.A) is objects.HelloStr
    assert type(test_obj2.A) is objects.HelloStr
    assert test_obj.dry_kwargs['A'] == test_obj2.dry_kwargs['A']
    assert test_obj.A.definition() == test_obj2.A.definition()
    assert test_obj.A.str_message == test_obj2.A.str_message
