import pytest
import dryml
import os
import tempfile
import glob

import core_objects as objs
from dryml.core import SKIP_ARGS
from dryml.core.utils.general import dir_store_inspect


@pytest.mark.usefixtures("create_temp_dir")
def test_add_retrieve_object_1(create_temp_dir):
    obj = objs.HelloStr(msg='test')

    repo = dryml.core.Repo(create_temp_dir)

    repo.add_object(obj)

    assert len(repo) == 1

    obj_dict = repo.get()

    obj = list(obj_dict.values())[0]

    assert obj.definition.stable_hash() == \
        obj.definition.stable_hash()


def test_add_retrieve_objects_2():
    repo = dryml.core.Repo()

    obj_list = []
    obj_list.append(objs.HelloStr(msg='test'))
    obj_list.append(objs.HelloInt(msg=10))
    obj_list.append(objs.HelloInt(msg=10))
    obj_list.append(objs.TestClassA(item=[10, 10]))
    obj_list.append(objs.TestClassB('test'))

    repo.add_object(*obj_list)

    assert len(repo.get(selector=objs.HelloInt.d(msg=10))) == 2
    assert len(repo.get(selector=objs.HelloStr.d(msg='test'))) == 1
    assert len(repo.get(selector=objs.TestClassA.d(item=[10, 10]))) == 1
    assert len(repo.get(selector=objs.TestClassB.d('test'))) == 1


def test_add_retrieve_objects_3():
    """
    Should be able to add all objs within an object at once.
    """

    repo = dryml.core.Repo()

    obj = objs.TestNest(objs.TestNest(A=objs.TestNest(10)))

    repo.add_object(obj)

    assert len(repo.get()) == 3
    assert len(repo.get(selector=objs.TestNest.d(SKIP_ARGS))) == 3
    assert len(repo.get(selector=objs.TestNest.d(10))) == 1

    assert len(repo.get(selector=objs.TestNest.d(objs.TestNest.d(SKIP_ARGS)))) == 1


def test_add_retrieve_objects_4():
    """
    Object hierarchy should work during selection
    """

    repo = dryml.core.Repo()

    parent_cls_obj = objs.TestBase()

    repo.add_object(parent_cls_obj)

    assert len(repo.get(
        selector=objs.TestBase.d(SKIP_ARGS))) == 1
    assert len(repo.get(
        selector=objs.TestClassA.d(SKIP_ARGS))) == 0


@pytest.mark.xfail
def test_try_write():
    repo = dryml.core.Repo()

    repo.add_object(objs.HelloStr(msg='test'))

    repo.save()


def test_get_api_1():
    repo = dryml.core.Repo()

    repo.add_object(objs.HelloStr(msg='test'))

    repo.get(sel_kwargs={'verbosity': 10})


def test_get_api_2():
    repo = dryml.core.Repo()

    obj1 = objs.TestNest4(1)
    obj2 = objs.TestNest4(2)
    obj3 = objs.TestNest4(3)
    obj4 = objs.TestNest4(4)

    repo.add_object(obj1)
    repo.add_object(obj2)
    repo.add_object(obj3)
    repo.add_object(obj4)

    # Get container for first object
    res1 = repo[obj1.definition]
    assert type(res1) is objs.TestNest4
    assert obj1.uid == res1.uid
    assert res1.A == 1

    res3 = repo[obj3.definition]
    assert type(res3) is objs.TestNest4
    assert res3.uid == obj3.uid
    assert res3.A == 3


@pytest.mark.usefixtures("create_temp_dir")
def test_get_api_3(create_temp_dir):
    repo = dryml.core.Repo()

    test_obj_def = objs.HelloStr.d(msg='test')

    assert len(repo.get(selector=test_obj_def)) == 0

    obj_dict = repo.get(selector=test_obj_def, build_missing=True)

    test_obj = list(obj_dict.values())[0]

    assert len(repo) == 1
    assert test_obj_def(test_obj)


@pytest.mark.usefixtures("create_temp_dir")
def test_get_api_4(create_temp_dir):
    repo = dryml.core.Repo()

    test_obj_def = objs.HelloStr.d(
        msg='test')

    assert len(repo.get(test_obj_def, build_missing=False)) == 0
    assert len(repo) == 0


@pytest.mark.usefixtures("create_temp_dir")
def test_get_api_5(create_temp_dir):
    repo = dryml.core.Repo(create_temp_dir)

    test_obj_def = objs.TestNest.d(
        objs.TestNest2.d(
            A=5)
        )

    obj = list(repo.get(test_obj_def, build_missing=True).values())[0]

    assert len(repo) == 2

    obj_dict = repo.get(test_obj_def)
    obj2 = list(obj_dict.values())[0]
    assert obj is obj2
    assert len(repo) == 2


@pytest.mark.usefixtures("create_temp_dir")
def test_write_1(create_temp_dir):
    repo = dryml.core.Repo(create_temp_dir)

    obj_list = []

    obj_list.append(objs.HelloStr(msg='test'))
    obj_list.append(objs.HelloInt(msg=10))
    obj_list.append(objs.HelloInt(msg=10))
    obj_list.append(objs.TestClassA(item=[10, 10]))
    obj_list.append(objs.TestClassB('test'))

    repo.add_object(*obj_list)

    repo.save()
    repo.close()

    # Delete repo
    del repo

    repo = dryml.core.Repo(create_temp_dir)
    repo.hydrate_from_stores()

    assert len(repo) == 5

    obj_dict = repo.get(selector=objs.HelloStr.d(msg='test'))
    assert len(obj_dict) == 1
    obj = list(obj_dict.values())[0]
    assert obj_list[0].definition.stable_hash() == \
        obj.definition.stable_hash()

    obj_dict = repo.get(selector=objs.HelloInt.d(msg=10))
    obj_list_2 = list(obj_dict.values())
    assert len(obj_list_2) == 2
    assert obj_list[1].definition.categorical().stable_hash() == \
        obj_list_2[0].definition.categorical().stable_hash()
    assert obj_list[1].definition.categorical().stable_hash() == \
        obj_list_2[1].definition.categorical().stable_hash()

    obj_dict = repo.get(selector=objs.TestClassA.d(item=[10, 10]))
    assert len(obj_dict) == 1
    obj = list(obj_dict.values())[0]
    assert obj_list[3].definition.stable_hash() == \
        obj.definition.stable_hash()

    obj_dict = repo.get(selector=objs.TestClassB.d('test'))
    assert len(obj_dict) == 1
    obj = list(obj_dict.values())[0]
    assert obj_list[4].definition.stable_hash() == \
        obj.definition.stable_hash()




@pytest.fixture
def prep_and_clean_test_dir2():
    with tempfile.TemporaryDirectory() as dir1, \
         tempfile.TemporaryDirectory() as dir2:
        yield dir1, dir2


def test_save_4(prep_and_clean_test_dir2):
    dir1, dir2 = prep_and_clean_test_dir2
    repo = dryml.core.Repo(dir1)

    repo.add_object(objs.HelloStr(msg='test'))
    repo.add_object(objs.HelloInt(msg=5))

    # Save objects in repository
    repo.save()

    # Save to a new location
    new_dirstore = dryml.core.store.dir.DirStore(dir2)
    repo.save(store=new_dirstore)

    # Delete the repo
    del repo

    defs_1 = set(dir_store_inspect(dir1))
    defs_2 = set(dir_store_inspect(dir2))
    assert defs_1 == defs_2


@pytest.mark.usefixtures("create_temp_dir")
def test_save_5(create_temp_dir):
    repo = dryml.core.Repo(create_temp_dir)

    repo.add_object(objs.HelloStr(msg='test'))

    repo.save_objs_on_deletion = True

    # Delete the repo
    del repo

    defs = set(dir_store_inspect(create_temp_dir))
    assert len(defs) == 1


@pytest.mark.usefixtures("create_temp_dir")
def test_save_7(create_temp_dir):
    repo = dryml.core.Repo(create_temp_dir)

    obj1 = objs.TestNest2(A=5)
    obj2 = objs.TestNest(obj1)

    repo.add_object(obj2)

    assert len(repo) == 2

    assert obj1 is repo[obj1.definition]
    assert obj2 is repo[obj2.definition]

    repo.save(obj2)

    assert len(dir_store_inspect(create_temp_dir)) == 2

    del repo

    repo = dryml.core.Repo(create_temp_dir)
    repo.hydrate_from_stores()

    assert len(repo) == 2

    assert len(repo.get(obj1.definition)) == 1
    assert len(repo.get(obj2.definition)) == 1


@pytest.mark.usefixtures("create_temp_dir")
def test_save_8(create_temp_dir):
    repo = dryml.core.Repo(create_temp_dir)

    obj1 = objs.TestNest2(A=5)
    obj2 = objs.TestNest(obj1)

    repo.save(obj2)

    assert len(repo) == 2

    assert len(dir_store_inspect(create_temp_dir)) == 2


@pytest.mark.usefixtures("create_temp_dir")
def test_object_save_restore_with_repo_1(create_temp_dir):
    """
    We test save and restore of nested objects through arguments
    """
    repo = dryml.core.Repo(create_temp_dir)

    # Create the data containing objects
    data_obj1 = objs.TestClassC2(10)
    data_obj1.set_val(20)

    # Add and save object in repo
    repo.add_object(data_obj1)
    repo.save()

    # Enclose them in another object
    obj = objs.TestClassC(data_obj1, B=data_obj1)

    # Load the object from the file
    obj2 = obj.definition.build(repo=repo)

    assert obj.definition == obj2.definition
    assert obj.A is obj.B
    assert obj2.A is obj2.B
    assert obj.A is obj2.A
    assert obj.B is obj2.B
    assert obj.A is obj2.B


@pytest.mark.usefixtures("create_temp_dir")
def test_object_save_restore_with_repo_2(create_temp_dir):
    """
    We test save and restore of nested objects with a repo
    """
    repo = dryml.core.Repo(create_temp_dir)

    # Create the data containing objects
    data_obj1 = objs.TestClassC2(10)
    data_obj1.set_val(20)

    # Add and save object in repo
    repo.add_object(data_obj1)
    repo.save()

    # Enclose them in another object
    obj = objs.TestClassC(data_obj1, B=data_obj1)

    # Save the enclosing object.
    repo.add_object(obj)
    repo.save()

    # There should now be two objects stored.
    assert len(repo) == 2

    # Get top object definition
    obj_def = obj.definition

    repo2 = dryml.core.Repo(create_temp_dir)

    obj2 = obj_def.build(repo=repo2)

    assert obj_def == obj2.definition
    assert obj2.A is obj2.B
    assert obj.A.C == obj2.A.C
    assert obj.B.C == obj2.B.C
    assert obj.A.data == obj2.A.data
    assert obj.B.data == obj2.B.data


@pytest.mark.usefixtures("create_temp_dir")
def test_object_save_restore_with_repo_3(create_temp_dir):
    """
    We test save and restore of nested objects with a repo
    """

    # Create workshop
    repo = dryml.core.Repo(create_temp_dir)

    obj_a = objs.TestNest(10)
    repo.add_object(obj_a)

    def build_def(repo):
        # Create the data containing objects
        obj_dict = repo.get(objs.TestNest.d(10))
        obj_a = list(obj_dict.values())[0]

        mdl_def = objs.TestNest2.d(
            A=10)

        mdl_def = objs.TestNest3.d(
            model=mdl_def)

        mdl_def = objs.TestNest3.d(
            obj_a,
            mdl_def)

        return mdl_def

    model_def = build_def(repo)

    @dryml.compute_context(ctx_context_reqs={'default': {}})
    def test_method(model_def, location):
        # Create repo
        repo = dryml.core.Repo(location)
        # Build the object
        model_obj = model_def.build(repo=repo)

        # Save all objects
        repo.save(model_obj)

    test_method(model_def, create_temp_dir)

    repo.hydrate_from_stores()

    obj_dict = repo.get(model_def, sel_kwargs={'verbosity': 2})
    assert len(obj_dict) == 1
