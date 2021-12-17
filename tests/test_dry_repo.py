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

    obj_list = repo.get(selector=dryml.DrySelector(cls=objects.HelloInt, kwargs={'msg': 10}))
    assert len(obj_list) == 2
    assert objs[1].get_category_hash() == obj_list[0].get_category_hash()
    assert objs[1].get_category_hash() == obj_list[1].get_category_hash()

    obj_list = repo.get(selector=dryml.DrySelector(cls=objects.TestClassA, kwargs={'item': [10,10]}))
    assert len(obj_list) == 1
    assert objs[3].get_individual_hash() == obj_list[0].get_individual_hash()

    obj_list = repo.get(selector=dryml.DrySelector(cls=objects.TestClassB, args=['test']))
    assert len(obj_list) == 1
    assert objs[4].get_individual_hash() == obj_list[0].get_individual_hash()

def test_reload_1(prep_and_clean_test_dir):
    repo = dryml.DryRepo(prep_and_clean_test_dir, create=True)

    objs = []

    objs.append(objects.TestClassA(item=[10]))
    objs.append(objects.TestClassA(item=[10, 10]))
    objs.append(objects.TestClassA(item='a'))

    for obj in objs:
        repo.add_object(obj)

    repo.reload_objs(selector=dryml.DrySelector(cls=objects.TestClassA), as_cls=objects.TestClassA2)

    obj = repo.get(selector=dryml.DrySelector(cls=objects.TestClassA2, kwargs={'item': [10]}))[0]
    assert objs[0].dry_kwargs['item'] == obj.dry_kwargs['item']

    obj = repo.get(selector=dryml.DrySelector(cls=objects.TestClassA2, kwargs={'item': [10, 10]}))[0]
    assert objs[1].dry_kwargs['item'] == obj.dry_kwargs['item']

    obj = repo.get(selector=dryml.DrySelector(cls=objects.TestClassA2, kwargs={'item': 'a'}))[0]
    assert objs[2].dry_kwargs['item'] == obj.dry_kwargs['item']

def test_save_1(prep_and_clean_test_dir):
    repo = dryml.DryRepo(prep_and_clean_test_dir, create=True)

    repo.add_object(objects.HelloStr(msg='test'))

    # Save objects in repository
    repo.save()

    # Delete the repo
    del repo

    # Load the repository objects should not be loaded right away
    repo = dryml.DryRepo(prep_and_clean_test_dir)

    assert len(repo.get(load_objects=False)) == 0

    repo.save()

    assert len(os.listdir(prep_and_clean_test_dir)) == 1

def test_save_2(prep_and_clean_test_dir):
    repo = dryml.DryRepo(prep_and_clean_test_dir, create=True)

    repo.add_object(objects.HelloStr(msg='test'), filepath='test_file')

    # Save objects in repository
    repo.save()

    # Delete the repo
    del repo

    # Load the repository objects should not be loaded right away
    repo = dryml.DryRepo(prep_and_clean_test_dir)

    assert len(repo.get(load_objects=False)) == 0

    repo.save()

    assert len(os.listdir(prep_and_clean_test_dir)) == 1

@pytest.fixture
def prep_and_clean_test_dir2():
    with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
        yield dir1, dir2

def test_save_3(prep_and_clean_test_dir2):
    dir1, dir2 = prep_and_clean_test_dir2
    repo = dryml.DryRepo(dir1, create=True)

    repo.add_object(objects.HelloStr(msg='test'), filepath='test_file')
    repo.add_object(objects.HelloInt(msg=5), filepath=os.path.join(dir2, 'test_file'))

    # Save objects in repository
    repo.save()

    # Delete the repo
    del repo

    assert len(os.listdir(dir1)) == 1
    assert len(os.listdir(dir2)) == 1

    # Load the repository objects should not be loaded right away
    repo = dryml.DryRepo(dir1)

    assert len(repo.get(load_objects=False)) == 0

    del repo

    repo = dryml.DryRepo(dir2)

    assert len(repo.get(load_objects=False)) == 0

def test_save_4(prep_and_clean_test_dir2):
    dir1, dir2 = prep_and_clean_test_dir2
    repo = dryml.DryRepo(dir1, create=True)

    repo.add_object(objects.HelloStr(msg='test'), filepath='test_file')
    repo.add_object(objects.HelloInt(msg=5))

    # Save objects in repository
    repo.save()
    repo.save(directory=dir2)

    # Delete the repo
    del repo

    assert set(os.listdir(dir1)) == set(os.listdir(dir2))
