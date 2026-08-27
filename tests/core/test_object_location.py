import os

from tests.core import core_objects as objects

from dryml.core import Repo
from dryml.core.repo import default_repo
from dryml.core.store.dir import DirStore


def test_object_save_accepts_explicit_store_and_sets_location(tmp_path):
    store = DirStore(tmp_path / "artifacts")
    repo = Repo()
    obj = objects.HelloStr(msg="located")

    obj.save(repo=repo, store=store)

    assert repo.location(obj) == store.object_dir(obj.definition)
    with default_repo(repo):
        assert obj.location == store.object_dir(obj.definition)
    assert os.path.exists(os.path.join(repo.location(obj), "def.pkl"))


def test_repo_save_object_uses_selected_store_for_location(tmp_path):
    primary = DirStore(tmp_path / "primary")
    secondary = DirStore(tmp_path / "secondary")
    repo = Repo(stores=primary)
    obj = objects.HelloStr(msg="secondary")

    repo.save_object(obj, store=secondary)

    assert repo.location(obj) == secondary.object_dir(obj.definition)
    assert secondary.has(obj.definition)
    assert not primary.has(obj.definition)


def test_repo_set_object_store_sets_location_without_saving(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo()
    obj = objects.HelloStr(msg="bound")

    repo.set_object_store(obj, store)

    assert repo.location(obj) == store.object_dir(obj.definition)
    assert not store.has(obj.definition)
