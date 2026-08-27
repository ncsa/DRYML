import asyncio

import pytest

import dryml
from dryml import session
from dryml.core import ConcreteDefinition, Object, Repo
from dryml.core.session import _construction_config
from dryml.runtime import materialization_admission, materialization_scope
from dryml.runtime.errors import RuntimeTransitionError


class ModeObject(Object):
    def __init__(self, value):
        super().__init__()
        self.value = value


@pytest.fixture(autouse=True)
def reset_modes():
    session.reset()
    dryml.reset_config()
    yield
    session.reset()
    dryml.reset_config()


def test_orchestrator_projects_definition_without_replacing_core_configuration():
    repo = Repo()
    dryml.configure(repo=repo, object_mode="fresh", cache="strong")

    session.set_mode("orchestrator")

    status = dryml.status()
    assert status["repo"] is repo
    assert status["cache"] == "strong"
    assert status["object_mode"] == "definition"
    assert isinstance(ModeObject("planned"), dryml.Definition)

    with dryml.config(object_mode="concrete"):
        assert isinstance(ModeObject("concrete"), ConcreteDefinition)


def test_generic_admission_does_not_lift_public_definition_floor():
    session.set_mode("orchestrator")

    with materialization_scope("off"):
        with materialization_admission(operation="retained_object_callback"):
            assert dryml.status()["object_mode"] == "definition"
            assert isinstance(ModeObject("still-planned"), dryml.Definition)


def test_private_construction_mode_cannot_escape_to_copied_async_context():
    async def scenario():
        release = asyncio.Event()

        async def construct_later():
            await release.wait()
            return ModeObject("task")

        with materialization_scope("off"):
            with materialization_admission(operation="parent"):
                with _construction_config():
                    task = asyncio.create_task(construct_later())

        release.set()
        return await task

    session.set_mode("orchestrator")

    assert isinstance(asyncio.run(scenario()), dryml.Definition)


@pytest.mark.parametrize("mode", ("fresh", "load_or_build"))
def test_public_materializing_modes_reject_before_contextvar_mutation(mode):
    session.set_mode("orchestrator")
    before = dryml.status()

    with pytest.raises(RuntimeTransitionError, match="object-mode floor"):
        dryml.configure(object_mode=mode)

    assert dryml.status() == before
