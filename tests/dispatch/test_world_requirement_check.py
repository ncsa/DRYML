from __future__ import annotations

import dryml

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
from dryml.dispatch.errors import DispatchPlanningError
from dryml.operations import make_function_call_spec
from dryml.worlds import LocalResourceInventory


@dryml.world.req(accelerators={"gpu": {"min": 1}})
def gpu_target():
    return None


def test_strict_world_incompatibility_blocks_selected_cpu_world_without_search():
    cpu_world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}}
    resolution = resolve_dispatch_plan(normalize_user_operation(gpu_target, allow_pickle=True), world=cpu_world, requirement_policy="strict")
    assert resolution.world_selection.source == "explicit"
    assert resolution.world_check.status == "incompatible"
    assert resolution.launchable is False


def test_ignore_skips_world_requirement_but_preserves_selected_world():
    cpu_world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}}
    resolution = resolve_dispatch_plan(normalize_user_operation(gpu_target, allow_pickle=True), world=cpu_world, requirement_policy="ignore")
    assert resolution.world_check.status == "skipped"
    assert resolution.world_check.details == ({"reason": "requirement_policy_ignore"},)
    assert resolution.launchable is True


def test_single_subprocess_plan_rejects_multi_worker_world_under_every_policy(tmp_path):
    world = {"roles": {"main": {"replicas": 2, "process": {}}}}
    with __import__("pytest").raises(DispatchPlanningError, match=r"plan_world\(\) or run_world\(\)"):
        Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
            make_function_call_spec("operator:add", args=[1, 2]),
            world=world,
            requirement_policy="ignore",
        )


def test_single_subprocess_plan_rejects_resources_it_cannot_allocate(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"devices": {"gpu": 1}}}}}}
    dispatcher = Dispatcher(store=DirStore(tmp_path / "store", query_index="none"))

    explanation = dispatcher.explain(make_function_call_spec("operator:add", args=[1, 2]), world=world, requirement_policy="ignore")
    assert explanation.launchable is False
    assert any(item.code == "dryml.dispatch.single_subprocess_resources_unsupported" for item in explanation.resolution.diagnostics)
    with __import__("pytest").raises(DispatchPlanningError, match="not launchable"):
        dispatcher.plan(make_function_call_spec("operator:add", args=[1, 2]), world=world, requirement_policy="ignore")


def test_single_subprocess_plan_allocates_selected_gpu_world(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 2, "accelerators": {"gpu": 1}}}}}}
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        make_function_call_spec("operator:add", args=[1, 2]),
        world=world,
        inventory=LocalResourceInventory((4, 5), {"gpu": ("gpu-a",)}),
        requirement_policy="ignore",
    )

    assert plan.envelope.allocation_view["cpus"] == [4, 5]
    assert plan.envelope.allocation_view["accelerators"] == {"gpu": ["gpu-a"]}
    assert plan.envelope.allocation_view["metadata"]["backend"] == "local_subprocess"
