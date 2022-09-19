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
    sel = dryml.Selector(cls=objects.TestClassA)

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
    sel = dryml.Selector(cls=objects.TestClassA)

    # Test selectors work with built classes
    assert sel(obj1_cont.definition())
    assert not sel(obj2_cont.definition())


@pytest.mark.usefixtures("create_temp_dir")
def test_add_retrieve_object_1(create_temp_dir):
    obj = objects.HelloStr(msg='test')

    repo = dryml.DryRepo(create_temp_dir)

    repo.add_object(obj)

    assert len(repo) == 1

    obj_ret = repo.get()

    assert type(obj_ret) is not list

    assert obj_ret.definition().get_individual_id() == \
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

    assert type(repo.get(selector=dryml.Selector(
        cls=objects.HelloStr, kwargs={'msg': 'test'}))) is not list
    assert len(repo.get(selector=dryml.Selector(
        cls=objects.HelloInt, kwargs={'msg': 10}))) == 2
    assert type(repo.get(selector=dryml.Selector(
        cls=objects.TestClassA, kwargs={'item': [10, 10]}))) is not list
    assert type(repo.get(selector=dryml.Selector(
        cls=objects.TestClassB, args=['test']))) is not list


def test_add_retrieve_objects_3():
    """
    Should be able to add all objects within an object at once.
    """

    repo = dryml.DryRepo()

    obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

    repo.add_object(obj)

    assert len(repo.get()) == 3
    assert type(repo.get(selector=dryml.Selector(
        cls=objects.TestNest, args=(10,)))) is not list
    assert type(repo.get(selector=dryml.Selector(
        cls=objects.HelloTrainableD))) is not list
    obj = repo.get(
        selector=dryml.Selector(
            cls=objects.TestNest,
            args=(dryml.Selector(objects.HelloTrainableD),),),
        sel_kwargs={'verbosity': 2})
    assert type(obj) is not list


def test_add_retrieve_objects_4():
    """
    Object hierarchy should work during selection
    """

    repo = dryml.DryRepo()

    parent_cls_obj = objects.TestBase()

    repo.add_object(parent_cls_obj)

    assert dryml.utils.count(repo.get(
        selector=dryml.Selector(objects.TestBase))) == 1
    try:
        repo.get(selector=dryml.Selector(objects.TestClassA))
        assert False
    except KeyError:
        pass


@pytest.mark.xfail
def test_try_write():
    repo = dryml.DryRepo()

    repo.add_object(objects.HelloStr(msg='test'))

    repo.save()


def test_get_api_1():
    repo = dryml.DryRepo()

    repo.add_object(objects.HelloStr(msg='test'))

    repo.get(sel_kwargs={'verbosity': 10})


def test_get_api_2():
    repo = dryml.DryRepo()

    obj1 = objects.TestNest(1)
    obj2 = objects.TestNest(2)
    obj3 = objects.TestNest(3)
    obj4 = objects.TestNest(4)

    repo.add_object(obj1)
    repo.add_object(obj2)
    repo.add_object(obj3)
    repo.add_object(obj4)

    # Get container for first object
    res1 = repo[obj1]
    assert type(res1) is objects.TestNest
    assert obj1.dry_id == res1.dry_id
    assert res1.A == 1

    res3 = repo[obj3]
    assert type(res3) is objects.TestNest
    assert res3.dry_id == obj3.dry_id
    assert res3.A == 3


@pytest.mark.usefixtures("create_temp_dir")
def test_get_api_3(create_temp_dir):
    repo = dryml.DryRepo()

    test_obj_def = dryml.ObjectDef(
        objects.HelloStr,
        msg='test')

    try:
        test_obj = repo.get(test_obj_def)
        assert False
    except KeyError:
        pass

    test_obj = repo.get(test_obj_def, build_missing_def=True)

    assert len(repo) == 1
    assert dryml.Selector.build(test_obj_def)(test_obj)


@pytest.mark.usefixtures("create_temp_dir")
def test_get_api_4(create_temp_dir):
    repo = dryml.DryRepo()

    test_obj_def = dryml.ObjectDef(
        objects.HelloStr,
        msg='test')

    try:
        repo.get(test_obj_def, build_missing_def=False)
        assert False
    except KeyError:
        pass

    assert len(repo) == 0


@pytest.mark.usefixtures("create_temp_dir")
def test_get_api_5(create_temp_dir):
    repo = dryml.DryRepo()

    test_obj_def = dryml.ObjectDef(
        objects.TestNest,
        dryml.ObjectDef(
            objects.TestNest2,
            A=5)
        )

    obj = repo.get(test_obj_def, build_missing_def=True)

    assert len(repo) == 2

    obj2 = repo.get(test_obj_def)
    assert obj is obj2
    assert len(repo) == 2


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

    obj = repo.get(selector=dryml.Selector(
        cls=objects.HelloStr, kwargs={'msg': 'test'}))
    assert type(obj) is not list
    assert objs[0].definition().get_individual_id() == \
        obj.definition().get_individual_id()

    obj_list = repo.get(selector=dryml.Selector(
        cls=objects.HelloInt, kwargs={'msg': 10}))
    assert len(obj_list) == 2
    assert objs[1].definition().get_category_id() == \
        obj_list[0].definition().get_category_id()
    assert objs[1].definition().get_category_id() == \
        obj_list[1].definition().get_category_id()

    obj = repo.get(selector=dryml.Selector(
        cls=objects.TestClassA, kwargs={'item': [10, 10]}))
    assert type(obj) is not list
    assert objs[3].definition().get_individual_id() == \
        obj.definition().get_individual_id()

    obj = repo.get(selector=dryml.Selector(
        cls=objects.TestClassB, args=['test']))
    assert type(obj) is not list
    assert objs[4].definition().get_individual_id() == \
        obj.definition().get_individual_id()


@pytest.mark.usefixtures("create_temp_dir")
def test_reload_1(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    objs = []

    objs.append(objects.TestClassA(item=[10]))
    objs.append(objects.TestClassA(item=[10, 10]))
    objs.append(objects.TestClassA(item='a'))

    for obj in objs:
        repo.add_object(obj)

    repo.reload_objs(selector=dryml.Selector(cls=objects.TestClassA),
                     as_cls=objects.TestClassA2)

    obj = repo.get(selector=dryml.Selector(
        cls=objects.TestClassA2, kwargs={'item': [10]}))
    assert objs[0].dry_kwargs['item'] == obj.dry_kwargs['item']

    obj = repo.get(selector=dryml.Selector(
        cls=objects.TestClassA2, kwargs={'item': [10, 10]}))
    assert objs[1].dry_kwargs['item'] == obj.dry_kwargs['item']

    obj = repo.get(selector=dryml.Selector(
        cls=objects.TestClassA2, kwargs={'item': 'a'}))
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

    try:
        repo.get(only_loaded=True)
        assert False
    except KeyError:
        pass

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

    try:
        repo.get(only_loaded=True)
        assert False
    except KeyError:
        pass

    del repo

    repo = dryml.DryRepo(dir2)

    try:
        repo.get(only_loaded=True)
        assert False
    except KeyError:
        pass


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

    try:
        repo.get(only_loaded=True)
        assert False
    except KeyError:
        pass
    assert len(os.listdir(create_temp_dir)) == 1


@pytest.mark.usefixtures("create_temp_dir")
def test_save_7(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    obj1 = objects.TestNest2(A=5)
    obj2 = objects.TestNest(obj1)

    repo.add_object(obj2)

    assert len(repo) == 2

    assert obj1 is repo[obj1]
    assert obj1 is repo[obj1.dry_id]
    assert obj2 is repo[obj2]
    assert obj2 is repo[obj2.dry_id]

    repo.save(obj2)

    assert len(os.listdir(create_temp_dir)) == 2

    del repo

    repo = dryml.DryRepo(create_temp_dir)

    assert len(repo) == 2

    assert type(repo.get(obj1)) is not list
    assert type(repo.get(obj2)) is not list


@pytest.mark.usefixtures("create_temp_dir")
def test_save_8(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    obj1 = objects.TestNest2(A=5)
    obj2 = objects.TestNest(obj1)

    repo.save(obj2)

    assert len(repo) == 2

    assert len(os.listdir(create_temp_dir)) == 2


@pytest.mark.usefixtures("create_temp_dir")
def test_delete_1(create_temp_dir):
    repo = dryml.DryRepo(create_temp_dir, create=True)

    repo.add_object(objects.HelloStr(msg='test'))

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
    repo = dryml.DryRepo(create_temp_dir, create=True)

    repo.add_object(objects.HelloStr(msg='test'))

    assert len(repo) == 1

    repo.delete()

    assert len(repo) == 0


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

    assert dryml.config.build_repo is None

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

    # Save the enclosing object.
    repo.add_object(obj)
    repo.save()

    # There should now be two objects stored.
    assert len(repo) == 2

    # Get top object definition
    obj_def = obj.definition()

    repo2 = dryml.DryRepo(create_temp_dir)

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
    import objects

    # Create workshop
    repo = dryml.DryRepo(directory=create_temp_dir)

    obj_a = objects.TestNest(10)
    repo.add_object(obj_a)

    def build_def(repo):
        # Create the data containing objects
        obj_a = dryml.utils.head(repo.get(
            selector=dryml.ObjectDef(objects.TestNest, 10)))

        mdl_def = dryml.ObjectDef(
            objects.TestNest2,
            A=10)

        mdl_def = dryml.ObjectDef(
            objects.TestNest3,
            model=mdl_def)

        mdl_def = dryml.ObjectDef(
            objects.TestNest3,
            obj_a,
            mdl_def)

        return mdl_def

    model_def = build_def(repo)

    @dryml.compute_context(ctx_context_reqs={'default': {}})
    def test_method(model_def, location):
        # Create repo
        repo = dryml.DryRepo(directory=location)

        # Build the object
        model_obj = model_def.build(repo=repo)

        # Save all objects
        repo.save(model_obj)

    test_method(model_def, create_temp_dir)

    repo.load_objects_from_directory()

    repo.get(model_def, sel_kwargs={'verbosity': 2})
