import dryml
import os
import glob
import core2_objects as objects


def test_save_1(primary_store_set):
    # Create repo and save object
    repo = dryml.core2.Repo(stores=primary_store_set.stores)
    repo.add_objects(objects.HelloStr(msg='test'))
    assert len(repo.strong_obj_cache) == 1
    repo.save()
    repo.close(flush=True)

    primary_store_set.rewind_all()

    # Load the repository objects should not be loaded right away
    repo = dryml.core2.Repo(stores=primary_store_set.stores)

    assert len(repo.get(restore_state=False)) == 0

    # But storage *does* have content; hydrate index into cache
    repo.hydrate_from_stores()
    assert len(repo.light_index) == 1

    # Save again should be no-op-ish and not corrupt anything
    repo.save()
    repo.close(flush=True)

    primary_store_set.rewind_all()

    # Still exactly one stored object after reopening again
    repo = dryml.core2.Repo(stores=primary_store_set.stores)
    repo.hydrate_from_stores()
    assert len(repo.light_index) == 1

    # Test that we can load a single object
    objs_loaded = repo.get(restore_state=True)
    assert len(objs_loaded) == 1


def test_save_1(primary_store_set):
    repo = dryml.core2.Repo(stores=primary_store_set.stores)

    repo.add_objects(objects.HelloStr(msg='test'))

    # Save objects in repository
    repo.save()

    assert len(dryml.core2.Repo.dir_store_inspect(primary_store_set.stores[0].base_dir)) == 1

    # Delete the repo
    del repo

    dryml.core2.repo._global_repo.clear_cache(weak=True)

    # Load the repository objects should not be loaded right away
    repo = dryml.core2.Repo(stores=primary_store_set.stores)

    try:
        result = repo.get(build_missing=False)
        assert len(result) == 0
    except KeyError:
        pass

    repo.save()

    assert len(dryml.core2.Repo.dir_store_inspect(primary_store_set.stores[0].base_dir)) == 1


def test_save_2(primary_store_set):
    repo = dryml.core2.Repo(stores=primary_store_set.stores)

    repo.add_objects(objects.HelloStr(msg='test'))

    # Save objects in repository
    repo.save()
    assert len(dryml.core2.Repo.dir_store_inspect(primary_store_set.stores[0].base_dir)) == 1

    # Delete the repo
    del repo
    dryml.core2.repo._global_repo.clear_cache(weak=True)

    # Load the repository objects should not be loaded right away
    repo = dryml.core2.Repo(stores=primary_store_set.stores)

    try:
        result = repo.get(build_missing=False)
        assert len(result) == 0
    except KeyError:
        pass

    repo.save()

    assert len(dryml.core2.Repo.dir_store_inspect(primary_store_set.stores[0].base_dir)) == 1


def test_object_save_restore_1(primary_store_set):
    """
    We test save and restore of nested objects through arguments
    """

    ic(id(dryml.core2.repo._global_repo))

    # Create the data containing objects
    data_obj1 = objects.TestClassC2(10)
    data_obj1.set_val(20)

    data_obj2 = objects.TestClassC2(20)
    data_obj2.set_val(40)

    # Enclose them in another object
    obj = objects.TestClassC(data_obj1, B=data_obj2)

    ic(list(dryml.core2.repo._global_repo.strong_obj_cache.keys()))
    ic(list(dryml.core2.repo._global_repo.weak_obj_cache.keys()))

    # Save to the backend
    obj.save(repo=primary_store_set.stores)

    # For file-like backends (buffer / zip_buffer), rewind before reading
    primary_store_set.rewind_all()

    # Load back
    obj2 = dryml.core2.load_object(repo=primary_store_set.stores)

    assert obj.definition == obj2.definition
    assert obj.A.data == obj2.A.data
    assert obj.B.data == obj2.B.data


def test_object_save_restore_2(primary_store_set):
    """
    We test save and restore of nested objects through arguments
    This time, we make sure identical objects are loaded as
    the same object.
    """
    # Create the data containing objects
    data_obj1 = objects.TestClassC2(10)
    data_obj1.set_val(20)

    # Enclose them in another object
    obj = objects.TestClassC(data_obj1, B=data_obj1)

    # Save to the backend
    obj.save(repo=primary_store_set.stores)

    # For file-like backends (buffer / zip_buffer), rewind before reading
    primary_store_set.rewind_all()

    # Load the object from the file
    obj2 = dryml.core2.load_object(repo=primary_store_set.stores)

    assert obj.definition == obj2.definition
    assert obj.A is obj.B


def test_object_save_restore_3(primary_store_set):
    """
    We test save and restore of nested objects through arguments
    Deeper nesting
    """
    # Create the data containing objects
    data_obj1 = objects.TestClassC2(10)
    data_obj1.set_val(20)

    data_obj2 = objects.TestClassC2(20)
    data_obj2.set_val(40)

    data_obj3 = objects.TestClassC2('test')
    data_obj3.set_val('test')

    data_obj4 = objects.TestClassC2(0.5)
    data_obj4.set_val(30.5)

    obj1 = objects.TestClassC(data_obj1, B=data_obj2)
    obj2 = objects.TestClassC(data_obj3, B=data_obj4)

    # Enclose them in another object
    obj = objects.TestClassC(obj1, B=obj2)

    # Save to the backend
    obj.save(repo=primary_store_set.stores)

    # For file-like backends (buffer / zip_buffer), rewind before reading
    primary_store_set.rewind_all()

    # Load the object from the file
    obj2 = dryml.core2.load_object(repo=primary_store_set.stores)

    assert obj.definition == obj2.definition
    assert obj.A.A.data == obj2.A.A.data
    assert obj.A.B.data == obj2.A.B.data
    assert obj.B.A.data == obj2.B.A.data
    assert obj.B.B.data == obj2.B.B.data


def test_object_save_restore_4(primary_store_set):
    """
    Test saving/restoring arguments/kwargs
    """
    # Create the data containing objects
    data_obj1 = objects.TestClassC2(10)
    data_obj1.set_val(20)

    data_obj2 = objects.TestClassC2(20)
    data_obj2.set_val(40)

    data_obj3 = objects.TestClassC2('test')
    data_obj3.set_val('test')

    data_obj4 = objects.TestClassC2(0.5)
    data_obj4.set_val(30.5)

    obj1 = objects.TestClassC(data_obj1, B=data_obj2)
    obj2 = objects.TestClassC(data_obj3, B=data_obj4)

    args = (obj1, obj2)
    args_def = (obj1.definition, obj2.definition)

    # Save objects to a buffer
    dryml.core2.save_object(args, repo=primary_store_set.stores)

    primary_store_set.rewind_all()

    # Load objects from buffer
    new_args = dryml.core2.load_object(args_def, repo=primary_store_set.stores)

    assert type(new_args[0]) is objects.TestClassC
    assert type(new_args[1]) is objects.TestClassC

    assert obj1.A.data == new_args[0].A.data
    assert obj1.B.data == new_args[0].B.data
    assert obj2.A.data == new_args[1].A.data
    assert obj2.B.data == new_args[1].B.data


def test_object_save_restore_5(primary_store_set):
    """
    Test saving/restoring arguments/kwargs
    """
    # Create the data containing objects
    model_obj = objects.TestNest2(A=10)
    opt_obj = objects.TestNest3(20, model=model_obj)
    loss_obj = objects.TestNest2(A='func')
    train_fn_obj = objects.TestNest3(
        optimizer=opt_obj,
        loss=loss_obj,
        epochs=10)

    trainable_obj = objects.TestNest3(
        model=model_obj,
        train_fn=train_fn_obj
    )

    args = (trainable_obj,)

    args_defs = (trainable_obj.definition,)

    dryml.core2.save_object(args, repo=primary_store_set.stores)

    primary_store_set.rewind_all()

    new_args = dryml.core2.load_object(args_defs, repo=primary_store_set.stores)

    recon_trainable_obj = new_args[0]
    assert type(recon_trainable_obj) is objects.TestNest3

    assert recon_trainable_obj['model'] is \
        recon_trainable_obj['train_fn']['optimizer']['model']
    assert recon_trainable_obj['train_fn']['epochs'] == 10
    assert recon_trainable_obj['model'].A == 10
    assert recon_trainable_obj['train_fn']['optimizer'][0] == 20


def test_save_load_1(primary_store_set):
    # Test save/load to/from a directory
    obj1 = objects.TestClass5(10, test='a')

    repo = dryml.core2.repo.Repo(stores=primary_store_set.stores)
    dryml.core2.save_object(obj1, repo=repo, main=True)
    repo.flush()

    ic(os.listdir(repo.stores[0].base_dir))
    assert len(os.listdir(repo.stores[0].base_dir)) == 2
    assert len(os.listdir(repo.stores[0].object_root_dir)) == 1

    del repo

    obj1_2 = dryml.core2.load_object(obj1.definition, repo=primary_store_set.stores)
    assert obj1_2.x == 10
    assert obj1_2.test == 'a'


def test_save_load_2(primary_store_set):
    # Test save/load to/from a directory
    obj1 = objects.TestClass5(10, test='a')
    obj2 = objects.TestClass5(20, test='b')
    obj3 = objects.TestClass5(obj1, test=obj2)
    obj4 = objects.TestClass5(obj3, test=obj2)
    assert obj3.x is obj1
    assert obj3.test is obj2
    assert obj4.test is obj2
    assert obj4.x is obj3

    repo = dryml.core2.repo.Repo(stores=primary_store_set.stores)
    dryml.core2.save_object(obj4, repo=repo, main=True)
    repo.flush()

    assert len(os.listdir(repo.stores[0].base_dir)) == 2
    obj_dirs = glob.glob(os.path.join(repo.stores[0].object_root_dir, '*', '*'))
    assert len(obj_dirs) == 4
    del repo

    obj4_2 = dryml.core2.load_object(obj4.definition, repo=primary_store_set.stores)
    assert obj4_2 is not obj4
    obj3_2 = obj4_2.x
    obj2_2 = obj4_2.test
    assert obj3_2.test is obj2_2
    assert obj4


def test_save_load_3(primary_store_set):
    # Test save/load to/from a directory another nested object
    obj1 = objects.TestClass5(10, test='a')
    obj2 = objects.TestClass5(20, test='b')
    obj3 = objects.TestClass5(30, test='c')
    obj4 = objects.TestClass5(40, test='d')

    obj5 = objects.TestClass5(obj1, test=obj2)
    obj6 = objects.TestClass5(obj2, test=obj3)
    obj7 = objects.TestClass5(obj3, test=obj4)

    obj8 = objects.TestClass5(obj5, test=obj6)
    obj9 = objects.TestClass5(obj6, test=obj7)

    obj10 = objects.TestClass5(obj8, test=obj9)

    obj11 = objects.TestClass5(obj10, test=obj10)

    repo = dryml.core2.repo.Repo(stores=primary_store_set.stores)
    dryml.core2.save_object(obj11, repo=repo, main=True)
    repo.flush()

    assert len(os.listdir(repo.stores[0].base_dir)) == 2
    obj_dirs = glob.glob(os.path.join(repo.stores[0].object_root_dir, '*', '*'))
    assert len(obj_dirs) == 11

    del repo

    obj11_2 = dryml.core2.load_object(obj11.definition, repo=primary_store_set.stores)
    obj10_2 = obj11_2.x
    assert obj11_2.test is obj10_2
    obj6_2 = obj10_2.x.test
    assert obj6_2 is obj10_2.test.x
    obj2_2 = obj6_2.x
    assert obj2_2 is obj10_2.x.x.test
    obj3_2 = obj6_2.test
    assert obj3_2 is obj10_2.test.test.x
