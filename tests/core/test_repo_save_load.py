import dryml
import os
import glob
from tests.core import core_objects as objects
import pytest
import tempfile
import numpy as np

from dryml.core import definition_mode
from dryml.core.repo import Repo, default_repo
from dryml.core.dtype import dtype
from dryml.core.tensor_spec import TensorSpec
from dryml.core.cardinality import Cardinality


def _persistent_query_index_count(store) -> int:
    path = getattr(store, "query_index_path", None)
    return int(path is not None and os.path.exists(path))


def _assert_expected_store_root_entries(store):
    entries = set(os.listdir(store.base_dir))
    assert {"def.pkl", "objects"} <= entries
    assert entries <= {"def.pkl", "aliases.pkl", "objects", ".dryml"}


def test_save_1(primary_store_set):
    # Create repo and save object
    repo = Repo(stores=primary_store_set.stores)
    repo.add_objects(objects.HelloStr(msg='test'))
    assert len(repo.strong_obj_cache) == 1
    repo.save()
    repo.close(flush=True)

    primary_store_set.rewind_all()

    # Load the repository objects should not be loaded right away
    repo = dryml.core.Repo(stores=primary_store_set.stores)

    assert len(repo.find_defs(None, refresh=False)) == _persistent_query_index_count(primary_store_set.stores[0])
    assert repo._num_constructions == 0

    # Auto discovery finds definitions without materializing objects.
    defs = repo.find_defs(None)
    assert len(defs) == 1
    assert repo._num_constructions == 0
    if _persistent_query_index_count(primary_store_set.stores[0]) == 0:
        assert len(repo.light_index) == 1

    # Save again should be no-op-ish and not corrupt anything
    repo.save()
    repo.close(flush=True)

    primary_store_set.rewind_all()

    # Still exactly one stored object after reopening again
    repo = dryml.core.Repo(stores=primary_store_set.stores)
    assert len(repo.find_defs(None)) == 1

    # Test that we can load a single object
    objs_loaded = repo.get(restore_state=True)
    assert len(objs_loaded) == 1


def test_save_2(primary_store_set):
    repo = dryml.core.Repo(stores=primary_store_set.stores)

    repo.add_objects(objects.HelloStr(msg='test'))

    # Save objects in repository
    repo.save()

    assert len(dryml.core.Repo.dir_store_inspect(primary_store_set.stores[0].base_dir)) == 1

    # Delete the repo
    del repo

    dryml.core.repo._global_repo.clear_cache(weak=True)

    # Load the repository objects should not be loaded right away
    repo = dryml.core.Repo(stores=primary_store_set.stores)

    assert len(repo.find_defs(None, refresh=False)) == _persistent_query_index_count(primary_store_set.stores[0])
    result = repo.get(build_missing=False)
    assert len(result) == 1

    repo.save()

    assert len(dryml.core.Repo.dir_store_inspect(primary_store_set.stores[0].base_dir)) == 1


def test_save_3(primary_store_set):
    repo = dryml.core.Repo(stores=primary_store_set.stores)

    repo.add_objects(objects.HelloStr(msg='test'))

    # Save objects in repository
    repo.save()
    assert len(dryml.core.Repo.dir_store_inspect(primary_store_set.stores[0].base_dir)) == 1

    # Delete the repo
    del repo
    dryml.core.repo._global_repo.clear_cache(weak=True)

    # Load the repository objects should not be loaded right away
    repo = dryml.core.Repo(stores=primary_store_set.stores)

    assert len(repo.find_defs(None, refresh=False)) == _persistent_query_index_count(primary_store_set.stores[0])
    result = repo.get(build_missing=False)
    assert len(result) == 1

    repo.save()

    assert len(dryml.core.Repo.dir_store_inspect(primary_store_set.stores[0].base_dir)) == 1


@pytest.fixture
def prep_and_clean_test_dir2():
    with tempfile.TemporaryDirectory() as dir1, \
         tempfile.TemporaryDirectory() as dir2:
        yield dir1, dir2


def test_save_4(prep_and_clean_test_dir2):
    from dryml.core.repo import make_store

    dir1, dir2 = prep_and_clean_test_dir2
    store1 = make_store(dir1)
    store2 = make_store(dir2)
    repo = dryml.core.Repo([store1, store2])

    repo.add_objects(objects.HelloStr(msg='test'))
    store1 = make_store(dir1)
    repo.set_default_store(store2)
    repo.add_objects(objects.HelloInt(msg=5))

    # Save objects in repository
    repo.save()

    # Delete the repo
    del repo

    assert len(dryml.core.Repo.dir_store_inspect(dir1)) == 1
    assert len(dryml.core.Repo.dir_store_inspect(dir2)) == 1

    # Load the repository objects should not be loaded right away
    repo = dryml.core.Repo(dir1)

    assert len(repo.find_defs(None)) == 1
    with pytest.raises(ValueError, match="load_or_build"):
        repo.get(build_missing=True)

    del repo

    repo = dryml.core.Repo(dir2)

    assert len(repo.find_defs(None)) == 1
    with pytest.raises(ValueError, match="load_or_build"):
        repo.get(build_missing=True)

    repo = dryml.core.Repo([dir1, dir2])
    assert len(repo.find_defs(None)) == 2
    assert len(repo.get()) == 2

    assert len(dryml.core.Repo.dir_store_inspect(dir1)) == 1
    assert len(dryml.core.Repo.dir_store_inspect(dir2)) == 1


def test_object_save_restore_1(primary_store_set):
    """
    We test save and restore of nested objects through arguments
    """

    # Create the data containing objects
    data_obj1 = objects.TestClassC2(10)
    data_obj1.set_val(20)

    data_obj2 = objects.TestClassC2(20)
    data_obj2.set_val(40)

    # Enclose them in another object
    obj = objects.TestClassC(data_obj1, B=data_obj2)

    # Save to the backend
    obj.save(repo=primary_store_set.stores)

    # For file-like backends (buffer / zip_buffer), rewind before reading
    primary_store_set.rewind_all()

    # Load back
    obj2 = dryml.core.load_object(repo=primary_store_set.stores)

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
    obj2 = dryml.core.load_object(repo=primary_store_set.stores)

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
    obj2 = dryml.core.load_object(repo=primary_store_set.stores)

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
    dryml.core.save_object(args, repo=primary_store_set.stores)

    primary_store_set.rewind_all()

    # Load objects from buffer
    new_args = dryml.core.load_object(args_def, repo=primary_store_set.stores)

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

    dryml.core.save_object(args, repo=primary_store_set.stores)

    primary_store_set.rewind_all()

    new_args = dryml.core.load_object(args_defs, repo=primary_store_set.stores)

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

    repo = dryml.core.repo.Repo(stores=primary_store_set.stores)
    dryml.core.save_object(obj1, repo=repo, main=True)
    repo.flush()

    _assert_expected_store_root_entries(repo.stores[0])
    assert len(os.listdir(repo.stores[0].object_root_dir)) == 1

    del repo

    obj1_2 = dryml.core.load_object(obj1.definition, repo=primary_store_set.stores)
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

    repo = dryml.core.repo.Repo(stores=primary_store_set.stores)
    dryml.core.save_object(obj4, repo=repo, main=True)
    repo.flush()

    _assert_expected_store_root_entries(repo.stores[0])
    obj_dirs = glob.glob(os.path.join(repo.stores[0].object_root_dir, '*', '*'))
    assert len(obj_dirs) == 1
    del repo

    obj4_2 = dryml.core.load_object(obj4.definition, repo=primary_store_set.stores)
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

    repo = dryml.core.repo.Repo(stores=primary_store_set.stores)
    dryml.core.save_object(obj11, repo=repo, main=True)
    repo.flush()

    _assert_expected_store_root_entries(repo.stores[0])
    obj_dirs = glob.glob(os.path.join(repo.stores[0].object_root_dir, '*', '*'))
    assert len(obj_dirs) == 1

    del repo

    obj11_2 = dryml.core.load_object(obj11.definition, repo=primary_store_set.stores)
    obj10_2 = obj11_2.x
    assert obj11_2.test is obj10_2
    obj6_2 = obj10_2.x.test
    assert obj6_2 is obj10_2.test.x
    obj2_2 = obj6_2.x
    assert obj2_2 is obj10_2.x.x.test
    obj3_2 = obj6_2.test
    assert obj3_2 is obj10_2.test.test.x


def f_test(x):
    return x+10

default_compare = lambda a, b: a == b

@pytest.fixture(
    params=[
        pytest.param(
            (lambda: 'a', default_compare),
            id='str',
        ),
        pytest.param(
            (lambda: 20, default_compare),
            id='int',
        ),
        pytest.param(
            (lambda: 3.5, default_compare),
            id='float',
        ),
        pytest.param(
            (lambda: np.array([1., 2., 3.], dtype=np.float32), lambda a, b: np.all(a == b)),
            id='np_array',
        ),
        pytest.param(
            (lambda: dtype("float32"), default_compare),
            id='dtype',
        ),
        pytest.param(
            (lambda: TensorSpec(shape=(1, 2, 3), dtype=dtype("float32")), lambda a, b: a == b),
            id='tensor_spec',
        ),
        pytest.param(
            (lambda: Cardinality.UNKNOWN, default_compare),
            id='cardinality_unknown'
        ),
        pytest.param(
            (lambda: Cardinality.INFINITE, default_compare),
            id='cardinality_infinite'
        ),
        pytest.param(
            (lambda: Cardinality(10), default_compare),
            id='cardinality_infinite'
        ),
        pytest.param(
            (
                lambda: f_test,
                lambda a, b: (
                    a(0) == 10 and
                    b(0) == 10 and
                    a(15) == 25 and
                    b(15) == 25
                )
            ),
            id='function'
        )
    ]
)
def leaf_case(request):
    make_obj, check_equal = request.param
    return make_obj(), check_equal


def test_save_load_leaf_roundtrip(primary_store_set, leaf_case):
    test_obj, test_check = leaf_case
    # Test save/load to/from a directory
    obj1 = objects.TestClass5(10, test=test_obj)
    obj1.save(primary_store_set.stores)

    obj1_2 = dryml.core.load_object(obj1.definition, repo=primary_store_set.stores)
    assert obj1_2.x == 10
    assert test_check(test_obj, obj1_2.test)


def test_save_load_revision_1(primary_store_set):
    # Test that we can save revisions and load them again
    obj = objects.TestClassC2(10)
    obj.set_val(1)

    assert obj.data == 1

    repo = dryml.core.repo.Repo(stores=primary_store_set.stores)

    repo.save_object(obj, revision='A')

    obj.set_val(2)

    assert obj.data == 2

    repo.save_object(obj, revision='B')

    store = primary_store_set.stores[0]
    store_dir = store.base_dir

    assert len(dryml.core.Repo.dir_store_inspect(store_dir)) == 1
    active_dir = store._active_state_dir(store.object_dir(obj.definition))
    assert len(glob.glob(os.path.join(active_dir, '*.pkl'))) == 3

    # Remove the objects
    del repo
    del obj

    # Re-create repo
    repo = dryml.core.repo.Repo(stores=primary_store_set.stores)

    with dryml.core.definition_mode():
        obj_def = objects.TestClassC2(10)

    obj = repo.load_object(obj_def, revision='A')

    assert obj.data == 1

    obj = repo.load_object(obj_def, revision='B')

    assert obj.data == 2


def test_save_load_revision_2(primary_store_set):
    # Test that we can save nested revisions and load them again

    repo = Repo(stores=primary_store_set.stores)

    with default_repo(repo):
        obj1 = objects.TestClassC2(10)
        obj1_cdef = obj1.definition

        obj1.set_val(1)
        repo.save_object(obj1, revision='A')

        obj1.set_val(2)
        repo.save_object(obj1, revision='B')

        obj2 = objects.TestClassC2(20)
        obj2_cdef = obj2.definition

        obj2.set_val(1)
        repo.save_object(obj2, revision='A')

        obj2.set_val(2)
        repo.save_object(obj2, revision='B')

        obj3 = objects.TestNest3(obj1, obj2)

        repo.save_object(obj3)

    # Clear objects
    del obj1, obj2, obj3, repo

    repo = Repo(stores=primary_store_set.stores)

    with definition_mode():
        test_nest_def = objects.TestNest3(
            obj1_cdef,
            obj2_cdef)

    obj = repo.load_object(
        test_nest_def,
        revision = {
            obj1_cdef: 'A',
            obj2_cdef: 'A' })

    assert obj.args[0].C == 10
    assert obj.args[0].data == 1
    assert obj.args[1].C == 20
    assert obj.args[1].data == 1

    obj = repo.load_object(
        test_nest_def,
        revision = {
            obj1_cdef: 'A',
            obj2_cdef: 'B' })

    assert obj.args[0].C == 10
    assert obj.args[0].data == 1
    assert obj.args[1].C == 20
    assert obj.args[1].data == 2

    obj = repo.load_object(
        test_nest_def,
        revision = {
            obj1_cdef: 'B',
            obj2_cdef: 'A' })

    assert obj.args[0].C == 10
    assert obj.args[0].data == 2
    assert obj.args[1].C == 20
    assert obj.args[1].data == 1

    obj = repo.load_object(
        test_nest_def,
        revision = {
            obj1_cdef: 'B',
            obj2_cdef: 'B' })

    assert obj.args[0].C == 10
    assert obj.args[0].data == 2
    assert obj.args[1].C == 20
    assert obj.args[1].data == 2
