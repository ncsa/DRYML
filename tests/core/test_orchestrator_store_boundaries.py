import pytest

from dryml import session
from dryml.core import Object, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.utils.general import pickle_load, pickle_save
from dryml.runtime.errors import RuntimeTransitionError


class StoreBoundaryObject(Serializable):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        pickle_save(self.value, f"{dest_dir}/value.pkl")

    def restore_state_from_dir_imp(self, src_dir, revision=None):
        self.value = pickle_load(f"{src_dir}/value.pkl")


@pytest.fixture(autouse=True)
def reset_runtime():
    session.reset()
    yield
    session.reset()


def test_strict_rejects_live_store_mutation_and_restoration_before_io(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = StoreBoundaryObject("saved", repo=repo)
    repo.save_object(obj)

    session.set_mode("orchestrator")

    for call in (
            lambda: repo.save_object(obj),
            lambda: repo.location(obj, store=store),
            lambda: store.restore_object(obj),
            lambda: obj.save_state_to_dir(str(tmp_path / "direct")),
            lambda: obj.restore_state_from_dir(str(tmp_path / "missing")),
    ):
        with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization"):
            call()


def test_definition_alias_and_main_reference_publication_remain_available(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = StoreBoundaryObject("reference", repo=repo)
    cdef = obj.definition

    session.set_mode("orchestrator")

    repo.set_main_def(cdef)
    repo.set_alias("reference", cdef)
    repo.flush()

    assert repo.get_alias("reference") == cdef
    assert Repo(stores=DirStore(store.base_dir)).main_def == cdef
