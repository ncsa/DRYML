import pytest

import core2_objects as objects
from dryml.core2.repo import Repo, RepoLoadError, load_alias, make_store, save_object


def test_repo_alias_loads_object_in_same_repo(primary_store_set):
    repo = Repo(stores=primary_store_set.stores)
    obj = objects.TestClass1(10, test="train")

    repo.save_object(obj, alias="train_data_1")

    assert repo.get_alias("train_data_1") == obj.definition
    assert repo.load_alias("train_data_1") is obj


def test_repo_alias_persists_across_reopen(primary_store_set):
    repo = Repo(stores=primary_store_set.stores)
    obj = objects.TestClass1(10, test="train")
    repo.save_object(obj, alias="train_data_1")
    repo.close(flush=True)

    repo2 = Repo(stores=primary_store_set.fresh_stores())
    loaded = repo2.load_alias("train_data_1")

    assert loaded.definition == obj.definition
    assert loaded.x == 10
    assert loaded.test == "train"


def test_top_level_load_alias(primary_store_set):
    obj = objects.TestClass1(20, test="eval")
    save_object(obj, repo=primary_store_set.stores, alias="eval_data")

    loaded = load_alias("eval_data", repo=primary_store_set.fresh_stores())

    assert loaded.definition == obj.definition

    assert loaded.x == 20
    assert loaded.test == "eval"


def test_repo_delete_alias_persists(primary_store_set):
    repo = Repo(stores=primary_store_set.stores)
    obj = objects.TestClass1(30, test="delete")
    repo.save_object(obj, alias="old_alias")
    repo.flush()

    assert repo.delete_alias("old_alias") == obj.definition
    repo.close(flush=True)

    repo2 = Repo(stores=primary_store_set.fresh_stores())
    with pytest.raises(KeyError):
        repo2.get_alias("old_alias")


def test_repo_unknown_alias_raises(primary_store_set):
    repo = Repo(stores=primary_store_set.stores)

    with pytest.raises(KeyError):
        repo.get_alias("missing")


def test_repo_rejects_bad_alias_name(primary_store_set):
    repo = Repo(stores=primary_store_set.stores)
    obj = objects.TestClass1(40, test="bad")

    with pytest.raises(TypeError):
        repo.set_alias(123, obj)

    with pytest.raises(ValueError):
        repo.set_alias("", obj)


def test_repo_alias_conflict_across_stores_raises(store_resource_factory):
    res1 = store_resource_factory("directory", prefix="alias_conflict_a")
    res2 = store_resource_factory("directory", prefix="alias_conflict_b")

    store1 = make_store(res1.resource)
    store2 = make_store(res2.resource)

    repo1 = Repo(stores=store1)
    repo1.save_object(objects.TestClass1(1, test="a"), alias="shared")
    repo1.close(flush=True)

    repo2 = Repo(stores=store2)
    repo2.save_object(objects.TestClass1(2, test="b"), alias="shared")
    repo2.close(flush=True)

    with pytest.raises(RepoLoadError):
        Repo(stores=[make_store(res1.resource), make_store(res2.resource)])
