import pytest

from dryml import session
from dryml.core import Definition, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.store.records import NodeNameRecord
from dryml.core.utils.general import pickle_load, pickle_save
from dryml.runtime.errors import RuntimeTransitionError

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


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
    """Recipe and exact saved names resolve as Nodes under the strict floor."""
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    cdef = Definition(StoreBoundaryObject, "reference").concretize(repo=repo)
    recipe = store.publish_node(cdef, kind="recipe")
    store.compare_and_set_node_name(NodeNameRecord("reference", "recipe", recipe.digest), expected=None)
    state = repo.save_object(StoreBoundaryObject("reference", repo=repo), alias="checkpoint")

    session.set_mode("orchestrator")

    recipe_name = store.read_node_name("reference")
    saved_name = store.read_node_name("checkpoint")
    assert recipe_name.root_kind == "recipe"
    assert saved_name.root_kind == "saved"
    reference = store.read_node(recipe_name.root_digest, kind=recipe_name.root_kind)
    state_ref = store.read_node(saved_name.root_digest, kind=saved_name.root_kind)

    assert reference.graph_equal(cdef)
    assert state_ref.object_id == state.object_id
    assert state_ref.state.key == state.state.key
    assert state_ref.graph_equal(state)
