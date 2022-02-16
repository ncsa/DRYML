import pytest
import dryml
import os
import tempfile

import objects


@pytest.mark.usefixtures("create_temp_dir")
def test_container_selector_1(create_temp_dir):
    # Create objects
    obj1 = objects.TestClassA(base_msg="Test1", item=5)
    obj2 = objects.TestClassB([1, 2, 3], base_msg="Test1")

    # Create containers for these objects
    obj1_cont = dryml.dry_repo.DryRepoContainer.from_object(
        obj1, directory=create_temp_dir)
    obj2_cont = dryml.dry_repo.DryRepoContainer.from_object(
        obj2, directory=create_temp_dir)

    # Create selector
    sel = dryml.DrySelector(cls=objects.TestClassA)

    # Test selectors work with built classes
    assert sel(obj1_cont.definition())
    assert not sel(obj2_cont.definition())


@pytest.mark.usefixtures("create_temp_dir")
def test_container_selector_2(create_temp_dir):
    # Create objects
    obj1 = objects.TestClassA(base_msg="Test1", item=5)
    obj2 = objects.TestClassB([1, 2, 3], base_msg="Test1")

    # Create containers for these objects
    obj1_cont = dryml.dry_repo.DryRepoContainer.from_object(
        obj1, directory=create_temp_dir)
    obj2_cont = dryml.dry_repo.DryRepoContainer.from_object(
        obj2, directory=create_temp_dir)

    # Save to disk, and unload these objects
    obj1_cont.save()
    obj2_cont.save()
    obj1_cont.unload()
    obj2_cont.unload()

    # Create selector
    sel = dryml.DrySelector(cls=objects.TestClassA)

    # Test selectors work with built classes
    assert sel(obj1_cont.definition())
    assert not sel(obj2_cont.definition())


@pytest.mark.usefixtures("create_temp_dir")
def test_add_retrieve_object_1(create_temp_dir):
    obj = objects.HelloStr(msg='test')

    repo = dryml.DryRepo(create_temp_dir)

    repo.add_object(obj)

    assert len(repo) == 1

    objs = repo.get()

    assert len(objs) == 1

    assert objs[0].definition().get_individual_id() == \
        obj.definition().get_individual_id()


def test_add_retrieve_objects_2():
    repo = dryml.DryRepo()

    objs = []
    objs.append(objects.HelloStr(msg='test'))
    objs.append(objects.HelloInt(msg=10))
    objs.append(objects.HelloInt(msg=10))
    objs.append(objects.TestClassA(item=[10, 10]))
    objs.append(objects.TestClassB('test'))

    for obj in objs:
        repo.add_object(obj)

    assert len(repo.get(selector=dryml.DrySelector(
        cls=objects.HelloStr, kwargs={'msg': 'test'}))) == 1
    assert len(repo.get(selector=dryml.DrySelector(
        cls=objects.HelloInt, kwargs={'msg': 10}))) == 2
    assert len(repo.get(selector=dryml.DrySelector(
        cls=objects.TestClassA, kwargs={'item': [10, 10]}))) == 1
    assert len(repo.get(selector=dryml.DrySelector(
        cls=objects.TestClassB, args=['test']))) == 1


@pytest.mark.xfail
def test_try_write():
    repo = dryml.DryRepo()

    repo.add_object(objects.HelloStr(msg='test'))

    repo.save()


@pytest.mark.usefixtures("create_temp_dir")
def test_write_1(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    objs = []

    objs.append(objects.HelloStr(msg='test'))
    objs.append(objects.HelloInt(msg=10))
    objs.append(objects.HelloInt(msg=10))
    objs.append(objects.TestClassA(item=[10, 10]))
    objs.append(objects.TestClassB('test'))

    for obj in objs:
        repo.add_object(obj)

    repo.save()

    # Delete repo
    del repo

    repo = dryml.DryRepo(create_temp_dir)

    assert len(repo) == 5

    obj_list = repo.get(selector=dryml.DrySelector(
        cls=objects.HelloStr, kwargs={'msg': 'test'}))
    assert len(obj_list) == 1
    assert objs[0].definition().get_individual_id() == \
        obj_list[0].definition().get_individual_id()

    obj_list = repo.get(selector=dryml.DrySelector(
        cls=objects.HelloInt, kwargs={'msg': 10}))
    assert len(obj_list) == 2
    assert objs[1].definition().get_category_id() == \
        obj_list[0].definition().get_category_id()
    assert objs[1].definition().get_category_id() == \
        obj_list[1].definition().get_category_id()

    obj_list = repo.get(selector=dryml.DrySelector(
        cls=objects.TestClassA, kwargs={'item': [10, 10]}))
    assert len(obj_list) == 1
    assert objs[3].definition().get_individual_id() == \
        obj_list[0].definition().get_individual_id()

    obj_list = repo.get(selector=dryml.DrySelector(
        cls=objects.TestClassB, args=['test']))
    assert len(obj_list) == 1
    assert objs[4].definition().get_individual_id() == \
        obj_list[0].definition().get_individual_id()


@pytest.mark.usefixtures("create_temp_dir")
def test_reload_1(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    objs = []

    objs.append(objects.TestClassA(item=[10]))
    objs.append(objects.TestClassA(item=[10, 10]))
    objs.append(objects.TestClassA(item='a'))

    for obj in objs:
        repo.add_object(obj)

    repo.reload_objs(selector=dryml.DrySelector(cls=objects.TestClassA),
                     as_cls=objects.TestClassA2)

    obj = repo.get(selector=dryml.DrySelector(
        cls=objects.TestClassA2, kwargs={'item': [10]}))[0]
    assert objs[0].dry_kwargs['item'] == obj.dry_kwargs['item']

    obj = repo.get(selector=dryml.DrySelector(
        cls=objects.TestClassA2, kwargs={'item': [10, 10]}))[0]
    assert objs[1].dry_kwargs['item'] == obj.dry_kwargs['item']

    obj = repo.get(selector=dryml.DrySelector(
        cls=objects.TestClassA2, kwargs={'item': 'a'}))[0]
    assert objs[2].dry_kwargs['item'] == obj.dry_kwargs['item']


@pytest.mark.usefixtures("create_temp_dir")
def test_save_1(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    repo.add_object(objects.HelloStr(msg='test'))

    # Save objects in repository
    repo.save()

    # Delete the repo
    del repo

    # Load the repository objects should not be loaded right away
    repo = dryml.DryRepo(create_temp_dir)

    assert len(repo.get(only_loaded=True)) == 0

    repo.save()

    assert len(os.listdir(create_temp_dir)) == 1


@pytest.mark.usefixtures("create_temp_dir")
def test_save_2(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    repo.add_object(objects.HelloStr(msg='test'), filepath='test_file')

    # Save objects in repository
    repo.save()

    # Delete the repo
    del repo

    # Load the repository objects should not be loaded right away
    repo = dryml.DryRepo(create_temp_dir)

    assert len(repo.get(only_loaded=True)) == 0

    repo.save()

    assert len(os.listdir(create_temp_dir)) == 1


@pytest.fixture
def prep_and_clean_test_dir2():
    with tempfile.TemporaryDirectory() as dir1, \
         tempfile.TemporaryDirectory() as dir2:
        yield dir1, dir2


def test_save_3(prep_and_clean_test_dir2):
    dir1, dir2 = prep_and_clean_test_dir2
    repo = dryml.DryRepo(dir1, create=True)

    repo.add_object(objects.HelloStr(msg='test'),
                    filepath='test_file')
    repo.add_object(objects.HelloInt(msg=5),
                    filepath=os.path.join(dir2, 'test_file'))

    # Save objects in repository
    repo.save()

    # Delete the repo
    del repo

    assert len(os.listdir(dir1)) == 1
    assert len(os.listdir(dir2)) == 1

    # Load the repository objects should not be loaded right away
    repo = dryml.DryRepo(dir1)

    assert len(repo.get(only_loaded=True)) == 0

    del repo

    repo = dryml.DryRepo(dir2)

    assert len(repo.get(only_loaded=True)) == 0


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


@pytest.mark.usefixtures("create_temp_dir")
def test_save_5(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    repo.add_object(objects.HelloStr(msg='test'), filepath='test_file')

    repo.save_objs_on_deletion = True

    # Delete the repo
    del repo

    assert len(os.listdir(create_temp_dir)) == 1


@pytest.mark.usefixtures("create_temp_dir")
def test_save_6(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    repo.add_object(objects.HelloStr(msg='test'), filepath='test_file')

    repo.save_and_cache()

    assert len(repo.get(only_loaded=True)) == 0
    assert len(os.listdir(create_temp_dir)) == 1


@pytest.mark.usefixtures("create_temp_dir")
def test_delete_1(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    repo.add_object(objects.HelloStr(msg='test'))

    repo.save()

    assert len(os.listdir(create_temp_dir)) == 1

    repo.delete()

    assert len(os.listdir(create_temp_dir)) == 0
    assert len(repo.get(load_objects=True)) == 0


@pytest.mark.usefixtures("create_temp_dir")
def test_object_save_restore_with_repo_1(create_temp_dir):
    """
    We test save and restore of nested objects through arguments
    """
    import objects

    repo = dryml.DryRepo(create_temp_dir, create=True)

    # Create the data containing objects
    data_obj1 = objects.TestClassC2(10)
    data_obj1.set_val(20)

    # Add and save object in repo
    repo.add_object(data_obj1)
    repo.save()

    # Enclose them in another object
    obj = objects.TestClassC(data_obj1, B=data_obj1)

    # Load the object from the file
    obj2 = obj.definition().build(repo=repo)

    assert obj.definition() == obj2.definition()
    assert obj.A is obj.B
    assert obj2.A is obj2.B
    assert obj.A is obj2.A
    assert obj.B is obj2.B
    assert obj.A is obj2.B
