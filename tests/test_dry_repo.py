import pytest
import dryml
import io
import os
import tempfile

import objects

@pytest.fixture
def prep_and_clean_test_dir():
    with tempfile.TemporaryDirectory() as directory:
        yield directory

def test_add_retrieve_object_1(prep_and_clean_test_dir):
    obj = objects.HelloStr(msg='test')

    repo = dryml.DryRepo(prep_and_clean_test_dir)

    repo.add_object(obj)

    assert repo.number_of_objects() == 1

    objs = repo.get()

    assert len(objs) == 1

    assert objs[0].get_individual_hash() == obj.get_individual_hash()

def test_add_retrieve_objects_2():
    repo = dryml.DryRepo()

    objs = []
    objs.append(objects.HelloStr(msg='test'))
    objs.append(objects.HelloInt(msg=10))
    objs.append(objects.HelloInt(msg=10))
    objs.append(objects.TestClassA(item=[10,10]))
    objs.append(objects.TestClassB('test'))

    for obj in objs:
        repo.add_object(obj)

    assert len(repo.get(selector=dryml.DrySelector(cls=objects.HelloStr, kwargs={'msg': 'test'}))) == 1
    assert len(repo.get(selector=dryml.DrySelector(cls=objects.HelloInt, kwargs={'msg': 10}))) == 2
    assert len(repo.get(selector=dryml.DrySelector(cls=objects.TestClassA, kwargs={'item': [10,10]}))) == 1
    assert len(repo.get(selector=dryml.DrySelector(cls=objects.TestClassB, args=['test']))) == 1

@pytest.mark.xfail
def test_try_write():
    repo = dryml.DryRepo()
    
    repo.add_object(objects.HelloStr(msg='test'))

    repo.save()

def test_write_1(prep_and_clean_test_dir):
    repo = dryml.DryRepo(prep_and_clean_test_dir, create=True)

    objs = []

    objs.append(objects.HelloStr(msg='test'))
    objs.append(objects.HelloInt(msg=10))
    objs.append(objects.HelloInt(msg=10))
    objs.append(objects.TestClassA(item=[10,10]))
    objs.append(objects.TestClassB('test'))

    for obj in objs:
        repo.add_object(obj)

    repo.save()

    # Delete repo
    del repo

    repo = dryml.DryRepo(prep_and_clean_test_dir)

    assert repo.number_of_objects() == 5

    obj_list = repo.get(selector=dryml.DrySelector(cls=objects.HelloStr, kwargs={'msg': 'test'}))
    assert len(obj_list) == 1
    assert objs[0].get_individual_hash() == obj_list[0].get_individual_hash()
