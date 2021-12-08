import pytest
import dryml
import io
import os

import objects

def test_add_object_1():
    obj = objects.HelloStr(msg='test')

    repo = dryml.DryRepo('test_dir')

    repo.add_object(obj)

    assert repo.number_of_objects() == 1

    objs = repo.get_objs()

    assert len(objs) == 1

    assert objs[0].get_individual_hash() == obj.get_individual_hash()

def test_add_objects_2():
    repo = dryml.DryRepo()

    objs = []
    objs.append(objects.HelloStr(msg='test'))
    objs.append(objects.HelloInt(msg=10))
    objs.append(objects.HelloInt(msg=10))
    objs.append(objects.TestClassA(item=[10,10]))
    objs.append(objects.TestClassB('test'))

    for obj in objs:
        repo.add_object(obj)

    assert len(repo.get_objs(selector=dryml.DrySelector(cls=objects.HelloStr, kwargs={'msg': 'test'}))) == 1
    assert len(repo.get_objs(selector=dryml.DrySelector(cls=objects.HelloInt, kwargs={'msg': 10}))) == 2
    assert len(repo.get_objs(selector=dryml.DrySelector(cls=objects.TestClassA, kwargs={'item': [10,10]}))) == 1
    assert len(repo.get_objs(selector=dryml.DrySelector(cls=objects.TestClassB, args=['test']))) == 1
