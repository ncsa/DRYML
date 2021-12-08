import pytest
import dryml
import io
import os

import objects

def test_add_object():
    obj = objects.HelloStr(msg='test')

    repo = dryml.DryRepo('test_dir')

    repo.add_object(obj)

    assert repo.number_of_objects() == 1

    objs = repo.get_objs()

    assert len(objs) == 1

    assert objs[0].get_individual_hash() == obj.get_individual_hash()
