import pytest

from tests.core import core_objects as objects
from dryml.core.repo import Repo, make_store


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_object_save_load_uses_complete_snapshot_metadata(store_resource_factory):
    res = store_resource_factory("directory", prefix="environment_guard")
    store = make_store(res.resource)
    obj = objects.HelloStr(msg="environment guard")
    repo = Repo([store])
    state = repo.save_object(obj)

    snapshot = repo.get_snapshot_directory(state, store=store)
    assert (snapshot / "metadata.json").is_file()
    assert repo.get_snapshot_metadata(state, store=store).state_ref == state

    loaded_repo = Repo([make_store(res.resource)])
    loaded = loaded_repo.load_state_ref(state, reuse_live="never")
    assert loaded.definition == obj.definition
    assert loaded.get_message() == "Hello! environment guard"


def test_new_save_observes_once_but_repeat_and_read_do_not(store_resource_factory, monkeypatch):
    res = store_resource_factory("directory", prefix="environment_observation_guard")
    store = make_store(res.resource)
    repo = Repo([store])
    calls = []

    def forbidden_observer():
        calls.append(True)
        raise OSError("synthetic observation failure")

    monkeypatch.setattr("dryml.environments.introspection.inspect_current", forbidden_observer)
    obj = objects.HelloStr(msg="first")
    state = repo.save_object(obj)
    repo.save_object(obj)
    repo.load_state_ref(state, reuse_live="never")

    assert calls == [True]
    assert repo.get_snapshot_metadata(state).environment_status == "unavailable"
