from pathlib import Path

from tests.core import core_objects as objects
from dryml.core.repo import Repo, make_store


def test_object_save_load_still_works_without_environment_records(store_resource_factory):
    res = store_resource_factory("directory", prefix="environment_guard")
    store = make_store(res.resource)
    obj = objects.HelloStr(msg="environment guard")
    repo = Repo([store])
    state = repo.save_object(obj)

    assert not (Path(store.base_dir) / "records").exists()
    assert not (Path(store.base_dir) / "environment").exists()

    loaded_repo = Repo([make_store(res.resource)])
    loaded = loaded_repo.load_state_ref(state, reuse_live="never")
    assert loaded.definition == obj.definition
    assert loaded.get_message() == "Hello! environment guard"


def test_saves_and_reads_do_not_observe_environment_before_u3(store_resource_factory, monkeypatch):
    res = store_resource_factory("directory", prefix="environment_observation_guard")
    store = make_store(res.resource)
    repo = Repo([store])
    calls = []

    def forbidden_observer():
        calls.append(True)
        raise AssertionError("U2 capture must not be wired to Store save/read paths")

    monkeypatch.setattr("dryml.environments.introspection.inspect_current", forbidden_observer)
    state = repo.save_object(objects.HelloStr(msg="first"))
    repo.save_object(objects.HelloStr(msg="first"))
    repo.load_state_ref(state, reuse_live="never")

    assert calls == []
