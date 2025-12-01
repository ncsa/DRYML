import pytest
import dryml
import os
import tempfile

import core2_objects as objs
from dryml.core2 import SKIP_ARGS


@pytest.mark.usefixtures("create_temp_dir")
def test_add_retrieve_object_1(create_temp_dir):
    obj = objs.HelloStr(msg='test')

    repo = dryml.core2.Repo(create_temp_dir)

    repo.add_object(obj)

    assert len(repo) == 1

    obj_dict = repo.get()

    obj = list(obj_dict.values())[0]

    assert obj.definition.stable_hash() == \
        obj.definition.stable_hash()


def test_add_retrieve_objects_2():
    repo = dryml.core2.Repo()

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

    repo = dryml.core2.Repo()

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

    repo = dryml.core2.Repo()

    parent_cls_obj = objs.TestBase()

    repo.add_object(parent_cls_obj)

    assert len(repo.get(
        selector=objs.TestBase.d(SKIP_ARGS))) == 1
    assert len(repo.get(
        selector=objs.TestClassA.d(SKIP_ARGS))) == 0


@pytest.mark.xfail
def test_try_write():
    repo = dryml.core2.Repo()

    repo.add_object(objs.HelloStr(msg='test'))

    repo.save()


def test_get_api_1():
    repo = dryml.core2.Repo()

    repo.add_object(objs.HelloStr(msg='test'))

    repo.get(sel_kwargs={'verbosity': 10})


def test_get_api_2():
    repo = dryml.core2.Repo()

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
    repo = dryml.core2.Repo()

    test_obj_def = objs.HelloStr.d(msg='test')

    assert len(repo.get(selector=test_obj_def)) == 0

    obj_dict = repo.get(selector=test_obj_def, build_missing=True)

    test_obj = list(obj_dict.values())[0]

    assert len(repo) == 1
    assert test_obj_def(test_obj)


@pytest.mark.usefixtures("create_temp_dir")
def test_get_api_4(create_temp_dir):
    repo = dryml.core2.Repo()

    test_obj_def = objs.HelloStr.d(
        msg='test')

    assert len(repo.get(test_obj_def, build_missing=False)) == 0
    assert len(repo) == 0


@pytest.mark.usefixtures("create_temp_dir")
def test_get_api_5(create_temp_dir):
    repo = dryml.core2.Repo()

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
    repo = dryml.core2.Repo(create_temp_dir)

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

    repo = dryml.core2.Repo(create_temp_dir)
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


@pytest.mark.usefixtures("create_temp_dir")
def test_reload_1(create_temp_dir):
    repo = dryml.core2.Repo(create_temp_dir, create=True)

    objs = []

    objs.append(objs.TestClassA(item=[10]))
    objs.append(objs.TestClassA(item=[10, 10]))
    objs.append(objs.TestClassA(item='a'))

    for obj in objs:
        repo.add_object(obj)

    repo.reload_objs(selector=dryml.Selector(cls=objs.TestClassA),
                     as_cls=objs.TestClassA2)

    obj = repo.get(selector=dryml.Selector(
        cls=objs.TestClassA2, kwargs={'item': [10]}))
    assert objs[0].dry_kwargs['item'] == obj.dry_kwargs['item']

    obj = repo.get(selector=dryml.Selector(
        cls=objs.TestClassA2, kwargs={'item': [10, 10]}))
    assert objs[1].dry_kwargs['item'] == obj.dry_kwargs['item']

    obj = repo.get(selector=dryml.Selector(
        cls=objs.TestClassA2, kwargs={'item': 'a'}))
    assert objs[2].dry_kwargs['item'] == obj.dry_kwargs['item']


@pytest.mark.usefixtures("create_temp_dir")
def test_save_1(create_temp_dir):
    repo = dryml.core2.Repo(create_temp_dir, create=True)

    repo.add_object(objs.HelloStr(msg='test'))

    # Save objects in repository
    repo.save()

    # Delete the repo
    del repo

    # Load the repository objects should not be loaded right away
    repo = dryml.core2.Repo(create_temp_dir)

    try:
        repo.get(only_loaded=True)
        assert False
    except KeyError:
        pass

    repo.save()

    assert len(os.listdir(create_temp_dir)) == 1


@pytest.mark.usefixtures("create_temp_dir")
def test_save_2(create_temp_dir):
    repo = dryml.core2.Repo(create_temp_dir, create=True)

    repo.add_object(objs.HelloStr(msg='test'), filepath='test_file')

    # Save objects in repository
    repo.save()

    # Delete the repo
    del repo

    # Load the repository objects should not be loaded right away
    repo = dryml.core2.Repo(create_temp_dir)

    try:
        repo.get(only_loaded=True)
        assert False
    except KeyError:
        pass

    repo.save()

    assert len(os.listdir(create_temp_dir)) == 1


@pytest.fixture
def prep_and_clean_test_dir2():
    with tempfile.TemporaryDirectory() as dir1, \
         tempfile.TemporaryDirectory() as dir2:
        yield dir1, dir2


def test_save_3(prep_and_clean_test_dir2):
    dir1, dir2 = prep_and_clean_test_dir2
    repo = dryml.core2.Repo(dir1, create=True)

    repo.add_object(objs.HelloStr(msg='test'),
                    filepath='test_file')
    repo.add_object(objs.HelloInt(msg=5),
                    filepath=os.path.join(dir2, 'test_file'))

    # Save objects in repository
    repo.save()

    # Delete the repo
    del repo

    assert len(os.listdir(dir1)) == 1
    assert len(os.listdir(dir2)) == 1

    # Load the repository objects should not be loaded right away
    repo = dryml.core2.Repo(dir1)

    try:
        repo.get(only_loaded=True)
        assert False
    except KeyError:
        pass

    del repo

    repo = dryml.core2.Repo(dir2)

    try:
        repo.get(only_loaded=True)
        assert False
    except KeyError:
        pass


def test_save_4(prep_and_clean_test_dir2):
    dir1, dir2 = prep_and_clean_test_dir2
    repo = dryml.core2.Repo(dir1, create=True)

    repo.add_object(objs.HelloStr(msg='test'), filepath='test_file')
    repo.add_object(objs.HelloInt(msg=5))

    # Save objects in repository
    repo.save()
    repo.save(directory=dir2)

    # Delete the repo
    del repo

    assert set(os.listdir(dir1)) == set(os.listdir(dir2))


@pytest.mark.usefixtures("create_temp_dir")
def test_save_5(create_temp_dir):
    repo = dryml.core2.Repo(create_temp_dir, create=True)

    repo.add_object(objs.HelloStr(msg='test'), filepath='test_file')

    repo.save_objs_on_deletion = True

    # Delete the repo
    del repo

    assert len(os.listdir(create_temp_dir)) == 1


@pytest.mark.usefixtures("create_temp_dir")
def test_save_6(create_temp_dir):
    repo = dryml.core2.Repo(create_temp_dir, create=True)

    repo.add_object(objs.HelloStr(msg='test'), filepath='test_file')

    repo.save_and_cache()

    try:
        repo.get(only_loaded=True)
        assert False
    except KeyError:
        pass
    assert len(os.listdir(create_temp_dir)) == 1


@pytest.mark.usefixtures("create_temp_dir")
def test_save_7(create_temp_dir):
    repo = dryml.core2.Repo(create_temp_dir, create=True)

    obj1 = objs.TestNest2(A=5)
    obj2 = objs.TestNest(obj1)

    repo.add_object(obj2)

    assert len(repo) == 2

    assert obj1 is repo[obj1]
    assert obj1 is repo[obj1.dry_id]
    assert obj2 is repo[obj2]
    assert obj2 is repo[obj2.dry_id]

    repo.save(obj2)

    assert len(os.listdir(create_temp_dir)) == 2

    del repo

    repo = dryml.core2.Repo(create_temp_dir)

    assert len(repo) == 2

    assert type(repo.get(obj1)) is not list
    assert type(repo.get(obj2)) is not list


@pytest.mark.usefixtures("create_temp_dir")
def test_save_8(create_temp_dir):
    repo = dryml.core2.Repo(create_temp_dir, create=True)

    obj1 = objs.TestNest2(A=5)
    obj2 = objs.TestNest(obj1)

    repo.save(obj2)

    assert len(repo) == 2

    assert len(os.listdir(create_temp_dir)) == 2


@pytest.mark.usefixtures("create_temp_dir")
def test_delete_1(create_temp_dir):
    repo = dryml.core2.Repo(create_temp_dir, create=True)

    repo.add_object(objs.HelloStr(msg='test'))

    repo.save()

    assert len(os.listdir(create_temp_dir)) == 1

    repo.delete()

    assert len(os.listdir(create_temp_dir)) == 0
    try:
        repo.get(load_objects=True)
        assert False
    except KeyError:
        pass


@pytest.mark.usefixtures("create_temp_dir")
def test_delete_2(create_temp_dir):
    repo = dryml.core2.Repo(create_temp_dir, create=True)

    repo.add_object(objs.HelloStr(msg='test'))

    assert len(repo) == 1

    repo.delete()

    assert len(repo) == 0


@pytest.mark.usefixtures("create_temp_dir")
def test_object_save_restore_with_repo_1(create_temp_dir):
    """
    We test save and restore of nested objects through arguments
    """
    repo = dryml.core2.Repo(create_temp_dir, create=True)

    # Create the data containing objects
    data_obj1 = objs.TestClassC2(10)
    data_obj1.set_val(20)

    # Add and save object in repo
    repo.add_object(data_obj1)
    repo.save()

    # Enclose them in another object
    obj = objs.TestClassC(data_obj1, B=data_obj1)

    # Load the object from the file
    obj2 = obj.definition().build(repo=repo)

    assert dryml.core.config.build_repo is None

    assert obj.definition() == obj2.definition()
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
    repo = dryml.core2.Repo(create_temp_dir, create=True)

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
    obj_def = obj.definition()

    repo2 = dryml.core2.Repo(create_temp_dir)

    obj2 = obj_def.build(repo=repo2)

    assert obj_def == obj2.definition()
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
    repo = dryml.core2.Repo(directory=create_temp_dir)

    obj_a = objs.TestNest(10)
    repo.add_object(obj_a)

    def build_def(repo):
        # Create the data containing objects
        obj_a = dryml.core.utils.head(repo.get(
            selector=dryml.ObjectDef(objs.TestNest, 10)))

        mdl_def = dryml.ObjectDef(
            objs.TestNest2,
            A=10)

        mdl_def = dryml.ObjectDef(
            objs.TestNest3,
            model=mdl_def)

        mdl_def = dryml.ObjectDef(
            objs.TestNest3,
            obj_a,
            mdl_def)

        return mdl_def

    model_def = build_def(repo)

    @dryml.compute_context(ctx_context_reqs={'default': {}})
    def test_method(model_def, location):
        # Create repo
        repo = dryml.core2.Repo(directory=location)

        # Build the object
        model_obj = model_def.build(repo=repo)

        # Save all objects
        repo.save(model_obj)

    test_method(model_def, create_temp_dir)

    repo.load_objects_from_directory()

    repo.get(model_def, sel_kwargs={'verbosity': 2})
