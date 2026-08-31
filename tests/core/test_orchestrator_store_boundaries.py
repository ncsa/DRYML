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

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pickle_save(self.value, f"{dest_dir}/value.pkl")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
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
            lambda: obj.save_state_to_dir(str(tmp_path / "direct"), codec="pkl"),
            lambda: obj.restore_state_from_dir(str(tmp_path / "missing"), codec="pkl"),
    ):
        with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization"):
            call()


def test_typed_alias_resolution_remains_available_without_materialization(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    state = repo.save_object(StoreBoundaryObject("reference", repo=repo), alias="reference")
    repo.set_state_alias("checkpoint", state)

    session.set_mode("orchestrator")

    object_ref = repo.resolve_object_alias("reference")
    state_ref = repo.resolve_state_alias(object_ref, "checkpoint")

    assert object_ref == state.object
    assert state_ref == state
