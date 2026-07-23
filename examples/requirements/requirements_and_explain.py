"""Resolve declared requirements and inspect a bounded dispatch explanation."""

from __future__ import annotations

import importlib

import dryml
import dryml.annotations as annotations
from dryml.core import Definition
from dryml.dispatch import Dispatcher
from dryml.environments import CurrentEnvironmentSpec
from dryml.worlds import LocalResourceInventory, WorldSpec


@dryml.env.req(requirements=("dryml>=0",))
class ReportModel:
    """A lightweight target used only to demonstrate requirement collection."""

    @dryml.world.req(cpus={"min": 1})
    def report(self) -> None:
        """Declare one method-level resource requirement."""


def planning_target() -> int:
    """Provide an importable, lightweight target for non-launching planning."""

    return 1


def main() -> None:
    """Run the deterministic documentation assertions."""

    resolution = annotations.resolve_definition_method_requirements(
        Definition(ReportModel), "report"
    )
    assert tuple(resolution.environment_requirement.requirements) == ("dryml>=0",)
    assert resolution.world_requirement is not None

    requested_world = WorldSpec.from_data(
        {"roles": {"main": {"replicas": 1, "process": {}}}}
    )
    previous_environment = dryml.environments.set_current(CurrentEnvironmentSpec())
    previous_world = dryml.worlds.set_current(requested_world)
    try:
        importable_target = importlib.import_module(
            "examples.requirements.requirements_and_explain"
        ).planning_target
        explanation = Dispatcher().explain(
            importable_target,
            inventory=LocalResourceInventory((0,)),
        )
        summary = {
            "launchable": explanation.launchable,
            "environment_source": explanation.resolution.environment_selection.source,
            "world_source": explanation.resolution.world_selection.source,
            "diagnostic_count": len(explanation.resolution.diagnostics),
        }
        assert summary["launchable"] is True
        assert summary["environment_source"] == "current"
        assert summary["world_source"] == "current"
    finally:
        if previous_environment is None:
            dryml.environments.reset_current()
        else:
            dryml.environments.set_current(previous_environment)
        if previous_world is None:
            dryml.worlds.reset_current()
        else:
            dryml.worlds.set_current(previous_world)


if __name__ == "__main__":
    main()
