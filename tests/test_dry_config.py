import pytest
import dryml
import io
import pickle
import objects
import copy


def test_dry_kwargs_add_items_1():
    test_val = {'a': 1, 1.0: 'test', 5: {'A': 'C', 'B': 20},
                (1, 'a'): 'test2', 'nest': {'Z': {'Y': {'X': 5}}}}

    # Val shouldn't raise error here
    stored_val = dryml.DryKwargs(test_val)

    buffer = io.BytesIO()

    stored_val.save(buffer)
    buffer.seek(0)
    new_val = pickle.loads(buffer.read())

    assert new_val == test_val
    assert dryml.DryKwargs(new_val) == stored_val


@pytest.mark.xfail
def test_dry_kwargs_add_items_2():
    test_val = {'a': lambda x: x}

    dryml.DryKwargs(test_val)


def test_dry_args_add_items_1():
    test_val = [1, 1.0, {'A': 'C', 'B': 20}, (1, 'a'), {'Z': {'Y': {'X': 5}}}]

    # Val shouldn't raise error here
    stored_val = dryml.DryArgs(test_val)

    buffer = io.BytesIO()

    stored_val.save(buffer)
    buffer.seek(0)
    new_val = pickle.loads(buffer.read())

    assert new_val == test_val
    assert dryml.DryArgs(new_val) == stored_val


@pytest.mark.xfail
def test_dry_args_add_items_2():
    test_val = [lambda x: x]

    dryml.DryArgs(test_val)


def test_adapt_val_1():
    test_val = [[('test', 0)]]

    adapted_val = dryml.dry_config.adapt_val(test_val)

    assert test_val == adapted_val


def test_adapt_val_2():
    """
    Test that adapt_val and detect_and_construct leave certain arguments
    Unchanged.
    """
    test_val = [[('test', 0)]]

    test_val_2 = dryml.dry_config.detect_and_construct(
        dryml.dry_config.adapt_val(test_val))

    assert test_val == test_val_2


def test_def_1():
    """
    Test conditions under which a definition is concrete
    """

    obj_def = dryml.DryObjectDef(objects.HelloStr)

    assert not obj_def.is_concrete()

    obj = obj_def.build()

    assert obj.definition().is_concrete()


def test_def_2():
    """
    A definition is only concrete if all of its components are concrete
    """

    obj = objects.TestClassC(
        objects.TestClassC2(10),
        B=objects.TestClassC2(20))

    obj_def = obj.definition()

    assert obj_def.is_concrete()

    obj_def_cpy = copy.copy(obj_def)
    del obj_def_cpy['dry_args'][0]['dry_kwargs']['dry_id']

    assert not obj_def_cpy.is_concrete()

    obj_def_cpy = copy.copy(obj_def)
    del obj_def_cpy['dry_kwargs']['B']['dry_kwargs']['dry_id']

    assert not obj_def_cpy.is_concrete()


def test_def_3():
    """
    A definition is only concrete if all of its components are concrete
    """

    obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

    assert obj.definition().is_concrete()

    obj_def = obj.definition()
    del obj_def['dry_args'][0]['dry_kwargs']['dry_id']

    assert not obj_def.is_concrete()

    obj_def = obj.definition()
    del obj_def['dry_args'][0]['dry_kwargs']['A']['dry_kwargs']['dry_id']

    assert not obj_def.is_concrete()


def test_def_4():
    """
    A case where a fully specified definition wasn't marked as concrete
    """

    obj = dryml.ObjectWrapper(objects.HelloTrainableD, obj_kwargs={'A': 5})

    assert obj.definition().is_concrete()
