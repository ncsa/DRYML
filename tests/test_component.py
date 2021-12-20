import pytest
import dryml
import os
import uuid


@pytest.fixture
def create_name():
    tempfile = str(uuid.uuid4())
    yield tempfile
    fullpath = f"{tempfile}.dry"
    if os.path.exists(fullpath):
        os.remove(fullpath)


def test_basic_component_1(create_name):
    import objects

    test_obj = objects.HelloComponent(msg='test1')

    test_obj.save_self(create_name)

    test_obj2 = dryml.load_object(create_name)

    assert test_obj.get_individual_hash() == test_obj2.get_individual_hash()

    assert test_obj2.dry_kwargs['msg'] == 'test1'
