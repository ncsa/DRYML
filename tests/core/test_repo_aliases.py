import pytest

from tests.core import core_objects as objects
from dryml.core import ObjectRef
from dryml.core.repo import Repo, RepoLoadError, load_alias, make_store, save_object
from dryml.core.store.dir import DirStore


def _require_writable_reference_store(primary_store_set):
    """Skip file-like Zip fixtures that cannot atomically publish aliases."""
    if not primary_store_set.stores[0].publication_capabilities.writable:
        pytest.skip("file-like ZipStore deliberately rejects mutable reference authority")


def _require_writable(repo):
    if not repo.default_store.publication_capabilities.writable:
        pytest.skip("reference publication requires a writable Store")


def test_repo_alias_loads_object_in_same_repo(primary_store_set):
    _require_writable_reference_store(primary_store_set)
    repo = Repo(stores=primary_store_set.stores)
    _require_writable(repo)
    obj = objects.TestClass1(10, test="train")

    repo.save_object(obj, alias="train_data_1")

    assert repo.get_alias("train_data_1") == obj.object_ref
    assert repo.load_alias("train_data_1").definition == obj.definition


def test_repo_alias_persists_across_reopen(primary_store_set):
    _require_writable_reference_store(primary_store_set)
    repo = Repo(stores=primary_store_set.stores)
    _require_writable(repo)
    obj = objects.TestClass1(10, test="train")
    repo.save_object(obj, alias="train_data_1")
    repo.close(flush=True)

    repo2 = Repo(stores=primary_store_set.fresh_stores())
    loaded = repo2.load_alias("train_data_1")

    assert repo2.get_alias("train_data_1") == obj.object_ref
    assert loaded.definition == obj.definition
    assert loaded.x == 10
    assert loaded.test == "train"


def test_top_level_load_alias(primary_store_set):
    _require_writable_reference_store(primary_store_set)
    if not primary_store_set.stores[0].publication_capabilities.writable:
        pytest.skip("reference publication requires a writable Store")
    obj = objects.TestClass1(20, test="eval")
    save_object(obj, repo=primary_store_set.stores, alias="eval_data")

    loaded = load_alias("eval_data", repo=primary_store_set.fresh_stores())

    assert loaded.definition == obj.definition

    assert loaded.x == 20
    assert loaded.test == "eval"


def test_repo_alias_deletion_is_not_a_legacy_cdef_mutation(primary_store_set):
    _require_writable_reference_store(primary_store_set)
    repo = Repo(stores=primary_store_set.stores)
    _require_writable(repo)
    obj = objects.TestClass1(30, test="delete")
    repo.save_object(obj, alias="old_alias")
    with pytest.raises(NotImplementedError, match="reference aliases"):
        repo.delete_alias("old_alias")
    assert repo.get_alias("old_alias") == obj.object_ref


def test_repo_unknown_alias_raises(primary_store_set):
    repo = Repo(stores=primary_store_set.stores)

    with pytest.raises(KeyError):
        repo.get_alias("missing")


def test_repo_rejects_bad_alias_name(primary_store_set):
    repo = Repo(stores=primary_store_set.stores)
    _require_writable(repo)
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

    repo = Repo(stores=[make_store(res1.resource), make_store(res2.resource)])
    with pytest.raises(RepoLoadError, match="conflicts across connected Stores"):
        repo.get_alias("shared")


def test_dirstore_object_alias_replaces_only_its_current_record(tmp_path):
    store_path = tmp_path / "shared-store"
    repo = Repo(stores=DirStore(store_path, query_index="memory"))
    first = repo.save_object(objects.TestClass1(1, test="first"))
    replacement = repo.save_object(objects.TestClass1(3, test="replacement"))

    repo.set_alias("current", first.object)
    repo.set_alias("current", replacement.object)

    record = DirStore(store_path, query_index="memory").read_object_alias("current")
    assert isinstance(record.object_ref, ObjectRef)
    assert record.object_ref == replacement.object
