import dryml

from dryml.core2 import Repo, Serializable, definition_mode
from dryml.core2.definition import ConcreteDefinition, Definition
from dryml.core2.repo import default_repo, get_default_repo
from dryml.core2.store.dir import DirStore
from dryml.core2.utils.general import pickle_load, pickle_save


class SessionThing(Serializable):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        pickle_save(self.value, f"{dest_dir}/value.pkl")

    def restore_state_from_dir_imp(self, src_dir, revision=None):
        self.value = pickle_load(f"{src_dir}/value.pkl")


def teardown_function():
    dryml.reset_config()


def test_configured_repo_is_default_but_default_repo_context_overrides():
    repo = Repo()
    override = Repo()

    dryml.configure(repo=repo)

    assert get_default_repo() is repo
    with default_repo(override):
        assert get_default_repo() is override
    assert get_default_repo() is repo


def test_config_context_temporarily_overrides_and_restores():
    dryml.configure(object_mode="load_or_build", cache="strong")

    with dryml.config(object_mode="fresh", cache="none"):
        assert dryml.status()["object_mode"] == "fresh"
        assert dryml.status()["cache"] == "none"

    assert dryml.status()["object_mode"] == "load_or_build"
    assert dryml.status()["cache"] == "strong"


def test_definition_mode_compatibility_uses_session_modes():
    dryml.configure(object_mode="load_or_build")

    with definition_mode():
        defn = SessionThing(1)

    with definition_mode(concrete=True):
        cdef = SessionThing(1)

    assert isinstance(defn, Definition)
    assert isinstance(cdef, ConcreteDefinition)
    assert dryml.status()["object_mode"] == "load_or_build"


def test_definition_mode_false_forces_fresh_inside_definition_mode():
    with definition_mode():
        with definition_mode(False):
            obj = SessionThing(1)

    assert isinstance(obj, SessionThing)
    assert obj.value == 1


def test_load_or_build_constructor_restores_saved_state(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = SessionThing(1, repo=repo)
    obj.value = 99
    repo.save_object(obj)
    repo.clear_cache(strong=True, weak=True)

    dryml.configure(repo=repo, object_mode="load_or_build", cache="strong")
    loaded = SessionThing(1)

    assert loaded.value == 99
    assert repo.strong_obj_cache[loaded.definition] is loaded


def test_load_or_build_falls_back_to_fresh_when_missing(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    dryml.configure(repo=repo, object_mode="load_or_build")

    obj = SessionThing(5)

    assert obj.value == 5


def test_fresh_context_does_not_load_root_object(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = SessionThing(1, repo=repo)
    obj.value = 99
    repo.save_object(obj)
    repo.clear_cache(strong=True, weak=True)

    dryml.configure(repo=repo, object_mode="load_or_build")
    with dryml.config(object_mode="fresh"):
        fresh = SessionThing(1)

    assert fresh.value == 1


def test_repo_load_or_build_restores_existing_state(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = SessionThing(1, repo=repo)
    obj.value = 42
    repo.save_object(obj)
    repo.close(flush=True)

    repo2 = Repo(stores=DirStore(store.base_dir))
    loaded = repo2.load_or_build(obj.definition)

    assert loaded.value == 42
