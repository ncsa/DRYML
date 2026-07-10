from __future__ import annotations

import dryml

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, normalize_user_operation, resolve_dispatch_plan
from dryml.environments import ContainerEnvironmentSpec, inspect_current
from dryml.operations import make_function_call_spec
from dryml.worlds import LocalResourceInventory


@dryml.world.req(cpus={"min": 2})
def cpu_target():
    return None


@dryml.world.req(roles={"trainer": {"replicas": {"exact": 2}, "resources": {"cpus": {"exact": 1}}}})
def multi_worker_target():
    return None


def test_no_override_hard_world_requirement_is_synthesized_once():
    inventory = LocalResourceInventory((0, 1, 2, 3))

    resolution = resolve_dispatch_plan(
        normalize_user_operation(cpu_target, allow_pickle=True),
        inventory=inventory,
        requirement_policy="strict",
        single_worker_only=True,
    )

    assert resolution.world_selection.source == "synthesized"
    assert resolution.world_synthesis is not None and resolution.world_synthesis.ok
    assert resolution.inventory_summary == inventory.summary()
    assert resolution.launchable


def test_dispatch_reports_inventory_discovery_failure_as_structured_synthesis_failure(monkeypatch):
    import dryml.worlds.synthesis as synthesis

    def fail_inventory(*_args, **_kwargs):
        raise RuntimeError("malformed local inventory")

    monkeypatch.setattr(synthesis, "local_inventory", fail_inventory)
    explanation = Dispatcher().explain(cpu_target, allow_pickle=True)

    assert explanation.launchable is False
    assert explanation.resolution.world_synthesis is not None
    assert explanation.resolution.world_synthesis.status == "error"
    assert explanation.resolution.world_synthesis.diagnostics[0].code == "inventory_discovery_failed"


def test_unsupported_resolver_environment_is_structurally_nonlaunchable_under_ignore():
    explanation = Dispatcher().explain(
        make_function_call_spec("operator:add", args=[1, 2]),
        environment_candidates=(ContainerEnvironmentSpec("example/image"),),
        requirement_policy="ignore",
    )

    assert explanation.resolution.environment_selection.source == "resolver"
    assert explanation.launchable is False
    assert any(item.code == "dryml.dispatch.environment_launch_unsupported" for item in explanation.resolution.diagnostics)


def test_attached_record_does_not_bypass_unsupported_environment_launch():
    explanation = Dispatcher().explain(
        make_function_call_spec("operator:add", args=[1, 2]),
        environment={"spec": ContainerEnvironmentSpec("example/image").to_data(), "record": inspect_current().to_data()},
        requirement_policy="ignore",
    )

    assert explanation.launchable is False
    assert any(item.code == "dryml.dispatch.environment_launch_unsupported" for item in explanation.resolution.diagnostics)


def test_explanation_formats_synthesized_inventory_summary():
    explanation = Dispatcher().explain(
        cpu_target,
        allow_pickle=True,
        inventory=LocalResourceInventory((2, 3), {"gpu": ("gpu-a",)}),
        requirement_policy="strict",
    )

    assert "inventory_cpus=2" in str(explanation)
    assert "inventory_accelerators=['gpu']" in str(explanation)


def test_plan_allocates_a_synthesized_one_worker_world(tmp_path):
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        cpu_target,
        allow_pickle=True,
        inventory=LocalResourceInventory((4, 5)),
        requirement_policy="strict",
    )

    assert plan.resolution.world_selection.source == "synthesized"
    assert plan.envelope.allocation_view["cpus"] == [4, 5]


def test_plan_world_synthesizes_an_omitted_multi_worker_world(tmp_path):
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan_world(
        multi_worker_target,
        allow_pickle=True,
        inventory=LocalResourceInventory((0, 1)),
        requirement_policy="strict",
    )

    assert len(plan.worker_plans) == 2
    assert len(plan.world_spec["payload"]["roles"]["trainer"]) == 2
