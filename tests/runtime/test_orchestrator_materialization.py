"""Strict-orchestrator materialization and local-workload boundaries."""

from __future__ import annotations

import threading

import pytest

from dryml import session
from dryml.core import Definition, Object, Repo
from dryml.core.query.result import DefinitionResultSet, ObjectResultSet
from dryml.core.repo import load_alias, load_object
from dryml.core.store.dir import DirStore
from dryml.core.canonical import from_canonical
from dryml.core.materialization import build_materialization_plan, execute_materialization_plan
from dryml.core.policies import RepoLoadOptions
from dryml.runtime.errors import PublicationBusyError, RuntimeTransitionError


class BoundaryObject(Object):
    """Minimal Object whose initializer makes materialization observable."""

    initialized = 0

    def __init__(self, value):
        super().__init__()
        type(self).initialized += 1
        self.value = value


class BlockingBoundaryObject(Object):
    """Object whose initializer exposes an in-flight materialization lease."""

    started = None
    release = None

    def __init__(self):
        super().__init__()
        type(self).started.set()
        type(self).release.wait(timeout=5)


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


@pytest.mark.parametrize(
    "endpoint",
    (
        "definition_build",
        "definition_results_objects",
        "definition_results_apply",
        "object_load",
        "object_restore",
        "store_restore",
        "repo_load",
        "repo_load_or_build",
        "repo_getitem",
        "repo_load_alias",
        "top_level_load_alias",
        "top_level_load_object",
        "repo_find",
        "repo_get",
        "repo_get_callable",
        "repo_apply",
        "repo_apply_graph",
    ),
)
def test_public_materialization_endpoint_matrix_rejects_before_effects(
    endpoint, tmp_path
):
    repo = Repo()
    live = BoundaryObject("matrix", repo=repo)
    repo.pin(live)
    repo.alias_index["boundary"] = live.definition
    definitions = DefinitionResultSet(
        repo,
        (live.definition,),
        materializable=True,
        replicas={live.definition: ()},
    )
    store = DirStore(tmp_path / "store", query_index="none")
    effects = []
    calls = {
        "definition_build": lambda: Definition(BoundaryObject, "new").build(repo=repo),
        "definition_results_objects": definitions.objects,
        "definition_results_apply": lambda: definitions.apply(effects.append),
        "object_load": lambda: live.load(repo=repo),
        "object_restore": lambda: live.restore_state_from_dir(str(tmp_path / "missing")),
        "store_restore": lambda: store.restore_object(live),
        "repo_load": lambda: repo.load(live.definition),
        "repo_load_or_build": lambda: repo.load_or_build(live.definition),
        "repo_getitem": lambda: repo[live.definition],
        "repo_load_alias": lambda: repo.load_alias("boundary"),
        "top_level_load_alias": lambda: load_alias("boundary", repo=repo),
        "top_level_load_object": lambda: load_object(live.definition, repo=repo),
        "repo_find": lambda: repo.find(live.definition, scope="cached"),
        "repo_get": lambda: repo.get(live.definition),
        "repo_get_callable": lambda: repo.get(lambda obj: effects.append(obj) or True),
        "repo_apply": lambda: repo.apply(effects.append),
        "repo_apply_graph": lambda: repo.apply_graph(live, effects.append),
    }

    _strict_orchestrator()

    _materialization_error(calls[endpoint])
    assert effects == []


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


def test_live_constructor_lease_blocks_concurrent_orchestrator_publication():
    started = threading.Event()
    release = threading.Event()
    failures = []
    BlockingBoundaryObject.started = started
    BlockingBoundaryObject.release = release

    def construct():
        try:
            BlockingBoundaryObject()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=construct)
    thread.start()
    assert started.wait(timeout=5)
    try:
        with pytest.raises(PublicationBusyError, match="lease"):
            session.set_mode("orchestrator")
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []
