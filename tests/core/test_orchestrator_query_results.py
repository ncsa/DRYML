import pytest

from dryml import session
from dryml.core import Object, Repo
from dryml.core.query.result import DefinitionResultSet, ObjectResultSet
from dryml.runtime.errors import RuntimeTransitionError


class QueryBoundaryObject(Object):
    def __init__(self, value):
        super().__init__()
        self.value = value


@pytest.fixture(autouse=True)
def reset_runtime():
    session.reset()
    yield
    session.reset()


def test_strict_fences_object_yielding_definition_and_retained_results():
    repo = Repo()
    obj = QueryBoundaryObject("cached", repo=repo)
    repo.pin(obj)
    definitions = DefinitionResultSet(
        repo, (obj.definition,), materializable=True, replicas={obj.definition: ()}
    )
    retained = ObjectResultSet(repo, {obj.definition: obj})

    session.set_mode("orchestrator")

    for call in (
            definitions.objects,
            retained.one,
            retained.first,
            lambda: retained[obj.definition],
            lambda: next(iter(retained.values())),
            lambda: retained.apply(lambda value: value),
    ):
        with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization"):
            call()


def test_lazy_graph_iterator_rechecks_before_yielding_live_objects():
    repo = Repo()
    obj = QueryBoundaryObject("lazy", repo=repo)
    iterator = repo.iter_graph(obj)

    session.set_mode("orchestrator")

    with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization"):
        next(iterator)


def test_guarded_object_views_preserve_mapping_view_behavior():
    repo = Repo()
    obj = QueryBoundaryObject("view", repo=repo)
    retained = ObjectResultSet(repo, {obj.definition: obj})

    items = retained.items()
    values = retained.values()

    assert len(items) == 1
    assert len(values) == 1
    assert list(items) == list(items) == [(obj.definition, obj)]
    assert list(values) == list(values) == [obj]
