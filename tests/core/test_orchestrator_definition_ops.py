import sys

import pytest

from dryml import session
from dryml.core import Definition, Object, Repo
from dryml.core.store.dir import DirStore
from dryml.runtime.errors import RuntimeTransitionError

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class DefinitionOnlyObject(Object):
    prepared = 0

    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        cls.prepared += 1
        return args, kwargs

    def __init__(self, value):
        super().__init__()
        self.value = value


@pytest.fixture(autouse=True)
def reset_runtime():
    loaded = tuple(name for name in ("tensorflow", "torch", "jax", "jaxlib") if name in sys.modules)
    assert loaded == (), f"earlier lightweight tests leaked optional frameworks: {loaded}"
    session.reset()
    DefinitionOnlyObject.prepared = 0
    yield
    session.reset()


def test_definition_identity_and_structural_query_paths_remain_available(tmp_path):
    """Named saved Nodes and structural/cache queries stay metadata-only."""
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(store)
    definition = Definition(DefinitionOnlyObject, "planned")
    cdef = definition.concretize(repo=repo)
    live = DefinitionOnlyObject("planned", repo=repo)
    state = repo.save_object(live, alias="planned")
    prepared = DefinitionOnlyObject.prepared

    session.set_mode("orchestrator")

    assert cdef.creation_digest()
    name = store.read_node_name("planned")
    assert name.root_kind == "saved"
    assert store.read_node(name.root_digest, kind=name.root_kind).graph_equal(state)
    assert repo.definition_graph(cdef).roots == (cdef,)
    matches = repo.query().sel(cdef).cdefs().cached(scope=repo).collect()
    assert len(matches) == 1
    assert matches.one().graph_equal(cdef)
    assert DefinitionOnlyObject.prepared == prepared


def test_definition_build_rejects_before_preparation_or_constructor():
    session.set_mode("orchestrator")

    with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization"):
        Definition(DefinitionOnlyObject, "blocked").build(repo=Repo())

    assert DefinitionOnlyObject.prepared == 0
