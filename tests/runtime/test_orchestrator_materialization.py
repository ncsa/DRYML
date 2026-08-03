"""Strict-orchestrator materialization and local-workload boundaries."""

from __future__ import annotations

import pytest

from dryml import session
from dryml.core import Definition, Object, Repo
from dryml.core.repo import load_object
from dryml.core.canonical import from_canonical
from dryml.core.materialization import build_materialization_plan, execute_materialization_plan
from dryml.core.policies import RepoLoadOptions
from dryml.core.query.result import ObjectResultSet
from dryml.runtime.errors import RuntimeTransitionError


class BoundaryObject(Object):
    """Minimal Object whose initializer makes materialization observable."""

    initialized = 0

    def __init__(self, value):
        super().__init__()
        type(self).initialized += 1
        self.value = value


@pytest.fixture(autouse=True)
def reset_session():
    """Keep the facade mode and construction counter isolated per assertion."""

    session.reset()
    BoundaryObject.initialized = 0
    yield
    session.reset()


def _strict_orchestrator():
    session.set_mode("orchestrator")


def _materialization_error(call):
    with pytest.raises(
        RuntimeTransitionError,
        match="Orchestration mode prohibits Object materialization",
    ):
        call()


def test_live_metaclass_construction_is_fenced_after_definition_mode_projection():
    repo = Repo()
    cdef = Definition(BoundaryObject, "live").concretize(repo=repo)

    _strict_orchestrator()

    assert isinstance(BoundaryObject("definition"), Definition)
    _materialization_error(lambda: BoundaryObject("live", repo=repo, __cdef__=cdef))
    assert BoundaryObject.initialized == 0


def test_plan_uses_metadata_only_reuse_and_executor_is_fenced():
    repo = Repo()
    live = BoundaryObject("cached", repo=repo)
    repo.pin(live)
    plan = build_materialization_plan(
        repo,
        live.definition,
        RepoLoadOptions(restore_state=False),
        memo={},
    )

    action = plan.actions[live.definition]
    assert action.kind == "reuse"
    assert not hasattr(action, "obj")

    _strict_orchestrator()
    _materialization_error(
        lambda: execute_materialization_plan(
            repo, plan, memo={}, revision={}, root=live.definition
        )
    )


def test_cache_and_retained_result_access_are_fenced_after_orchestration_starts():
    repo = Repo()
    live = BoundaryObject("cached", repo=repo)
    repo.pin(live)
    retained = ObjectResultSet(repo, {live.definition: live})
    called = []

    _strict_orchestrator()

    _materialization_error(lambda: repo.get_cached(live.definition))
    _materialization_error(lambda: retained[live.definition])
    _materialization_error(retained.one)
    _materialization_error(retained.one_or_none)
    _materialization_error(retained.first)
    _materialization_error(lambda: retained.apply(called.append))
    assert called == []


def test_public_object_loaders_are_fenced_even_for_an_existing_object():
    repo = Repo()
    live = BoundaryObject("existing", repo=repo)

    _strict_orchestrator()

    _materialization_error(lambda: repo.load_object(live))
    _materialization_error(lambda: load_object(live, repo=repo))
    _materialization_error(lambda: live.load(repo=repo))
    assert from_canonical(live, repo=repo) is live


def test_definition_only_work_remains_available_in_strict_orchestration():
    _strict_orchestrator()

    definition = Definition(BoundaryObject, "definition")

    assert definition.cls is BoundaryObject
    assert definition.concretize().stable_hash()


def test_canonical_reconstruction_and_lazy_graph_retrieval_are_fenced_per_next():
    repo = Repo()
    live = BoundaryObject("graph", repo=repo)
    repo.pin(live)
    graph = repo.iter_graph(live)

    _strict_orchestrator()

    _materialization_error(lambda: from_canonical(live.definition, repo=repo, build_missing=True))
    assert from_canonical(live, repo=repo) is live
    _materialization_error(lambda: next(graph))


def test_lazy_graph_rechecks_orchestration_between_yields():
    repo = Repo()
    first = BoundaryObject("first", repo=repo)
    second = BoundaryObject("second", repo=repo)
    graph = repo.iter_graph((first, second))

    assert next(graph) is first
    _strict_orchestrator()

    _materialization_error(lambda: next(graph))
