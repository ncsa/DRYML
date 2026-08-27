import sys

import pytest

from dryml import session
from dryml.core import Definition, Object, Repo
from dryml.runtime.errors import RuntimeTransitionError


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


def test_definition_identity_and_structural_query_paths_remain_available():
    repo = Repo()
    definition = Definition(DefinitionOnlyObject, "planned")
    cdef = definition.concretize(repo=repo)
    repo.set_alias("planned", cdef)

    session.set_mode("orchestrator")

    assert cdef.stable_hash()
    assert repo.get_alias("planned") == cdef
    assert repo.definition_graph(cdef).roots == (cdef,)
    assert list(repo.find_defs(cdef, scope="cached")) == []


def test_definition_build_rejects_before_preparation_or_constructor():
    session.set_mode("orchestrator")

    with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization"):
        Definition(DefinitionOnlyObject, "blocked").build(repo=Repo())

    assert DefinitionOnlyObject.prepared == 0
