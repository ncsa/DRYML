import pytest
import dryml
import io
import pickle


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
