import pytest

from dryml import session
from dryml.core import Definition, Object, Repo
from dryml.core.canonical import from_canonical
from dryml.core.materialization import build_materialization_plan, execute_materialization_plan
from dryml.core.policies import RepoLoadOptions
from dryml.runtime import materialization_scope
from dryml.runtime.errors import RuntimeTransitionError


class MatrixObject(Object):
    initialized = 0

    def __init__(self, value):
        super().__init__()
        type(self).initialized += 1
        self.value = value


@pytest.fixture(autouse=True)
def reset_runtime():
    session.reset()
    MatrixObject.initialized = 0
    yield
    session.reset()


def test_strict_rejects_constructor_cache_and_plan_execution_before_hooks():
    repo = Repo()
    live = MatrixObject("cached", repo=repo)
    repo.pin(live)
    initialized = MatrixObject.initialized

    session.set_mode("orchestrator")

    plan = build_materialization_plan(
        repo, live.definition, RepoLoadOptions(restore_state=False)
    )
    assert plan.actions[live.definition].reuse_source == "cache"
    assert not hasattr(plan.actions[live.definition], "obj")

    for call in (
            lambda: MatrixObject("definition", repo=repo, __cdef__=live.definition),
            lambda: repo.get_cached(live.definition),
            lambda: live.definition.args,
            lambda: from_canonical(live.definition, repo=repo),
            lambda: execute_materialization_plan(repo, plan, memo={}, revision={}, root=live.definition),
    ):
        with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization"):
            call()
    assert MatrixObject.initialized == initialized


def test_explicit_warn_scope_admits_private_construction_without_lifting_public_floor():
    repo = Repo()
    session.set_mode("orchestrator")

    with materialization_scope("warn"):
        built = Definition(MatrixObject, "admitted").build(repo=repo)

    assert built.value == "admitted"
    assert isinstance(MatrixObject("still-planned"), Definition)
