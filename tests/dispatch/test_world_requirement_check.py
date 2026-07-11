from __future__ import annotations

from contextlib import nullcontext

import dryml
import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
from dryml.dispatch.errors import DispatchPlanningError
from dryml.operations import make_function_call_spec
from dryml.worlds import LocalResourceInventory


@dryml.world.req(accelerators={"gpu": {"min": 1}})
def gpu_target():
    return None


@dryml.world.req(cpus={"max": 0})
def zero_cpu_target():
    return None


@dryml.world.req(topology={"collectives": True})
def collective_target():
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


def test_single_subprocess_rejects_unenacted_process_metadata():
    world = {"roles": {"main": {"replicas": 1, "process": {"metadata": {"label": "training"}}}}}

    explanation = Dispatcher().explain(make_function_call_spec("operator:add", args=[1, 2]), world=world, requirement_policy="ignore")

    assert not explanation.launchable
    assert any(item.code == "dryml.dispatch.single_subprocess_process_settings_unsupported" for item in explanation.resolution.diagnostics)


def test_single_subprocess_allows_zero_valued_unsupported_resources(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"devices": {"gpu": 0}, "named": {"scratch": 0}}}}}}

    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        make_function_call_spec("operator:add", args=[1, 2]),
        world=world,
        inventory=LocalResourceInventory((0,)),
        requirement_policy="ignore",
    )

    assert plan.envelope.allocation_view["cpus"] == [0]


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


def test_single_subprocess_plan_accepts_local_subprocess_backend(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}, "backend": {"kind": "local_subprocess", "parameters": {}}}
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        make_function_call_spec("operator:add", args=[1, 2]),
        world=world,
        inventory=LocalResourceInventory((4,)),
        requirement_policy="ignore",
    )

    assert plan.envelope.allocation_view["metadata"]["backend"] == "local_subprocess"
    assert plan.envelope.launch["world_allocation_spec"]["payload"]["backend"]["kind"] == "local_subprocess"
    assert plan.envelope.launch["world_allocation_spec"]["metadata"]["world_id"] == plan.envelope.launch["world_id"]


def test_single_subprocess_rejects_unknown_memory_inventory(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"memory": "1GiB"}}}}}
    with __import__("pytest").raises(DispatchPlanningError, match="memory request cannot be proven"):
        Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
            make_function_call_spec("operator:add", args=[1, 2]),
            world=world,
            inventory=LocalResourceInventory((0,)),
            requirement_policy="ignore",
        )


def test_explain_rejects_infeasible_explicit_world_before_planning(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 2}}}}}
    dispatcher = Dispatcher(store=DirStore(tmp_path / "store", query_index="none"))

    explanation = dispatcher.explain(
        make_function_call_spec("operator:add", args=[1, 2]),
        world=world,
        inventory=LocalResourceInventory((0,)),
        requirement_policy="ignore",
    )

    assert explanation.launchable is False
    assert any(item.code == "dryml.dispatch.local_allocation_failed" for item in explanation.resolution.diagnostics)
    with __import__("pytest").raises(DispatchPlanningError, match="not launchable"):
        dispatcher.plan(
            make_function_call_spec("operator:add", args=[1, 2]),
            world=world,
            inventory=LocalResourceInventory((0,)),
            requirement_policy="ignore",
        )


def test_plan_world_validates_actual_allocation_against_requirement(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 0}}}}}

    with __import__("pytest").raises(DispatchPlanningError, match="actual local allocation does not satisfy"):
        Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan_world(
            zero_cpu_target,
            world=world,
            inventory=LocalResourceInventory((0,)),
            requirement_policy="strict",
        )


def test_actual_allocation_requirement_respects_warn_and_ignore(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 0}}}}}
    dispatcher = Dispatcher(store=DirStore(tmp_path / "store", query_index="none"))

    for policy in ("warn", "ignore"):
        assert dispatcher.explain(zero_cpu_target, world=world, inventory=LocalResourceInventory((0,)), allow_pickle=True, requirement_policy=policy).launchable
        expectation = pytest.warns(RuntimeWarning) if policy == "warn" else nullcontext()
        with expectation:
            assert dispatcher.plan(zero_cpu_target, world=world, inventory=LocalResourceInventory((0,)), allow_pickle=True, requirement_policy=policy)
        expectation = pytest.warns(RuntimeWarning) if policy == "warn" else nullcontext()
        with expectation:
            assert dispatcher.plan_world(zero_cpu_target, world=world, inventory=LocalResourceInventory((0,)), allow_pickle=True, requirement_policy=policy)


def test_zero_memory_request_does_not_require_known_memory_inventory(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"memory": "0B"}}}}}

    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        make_function_call_spec("operator:add", args=[1, 2]),
        world=world,
        inventory=LocalResourceInventory((0,)),
        requirement_policy="ignore",
    )

    assert plan.envelope.allocation_view["memory"] == 0


def test_local_subprocess_allocation_process_identity_uses_requested_world_id(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}, "backend": {"kind": "local_subprocess", "parameters": {}}}

    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        make_function_call_spec("operator:add", args=[1, 2]),
        world=world,
        inventory=LocalResourceInventory((0,)),
        requirement_policy="ignore",
    )

    allocation = plan.envelope.launch["world_allocation_spec"]
    assert allocation["metadata"]["world_id"] == plan.envelope.launch["world_id"]
    assert allocation["payload"]["roles"]["main"][0]["env"]["DRYML_WORLD_ID"] == plan.envelope.launch["world_id"]


def test_single_subprocess_topology_remains_structural_under_relaxed_policies():
    world = {"roles": {"main": {"replicas": 1, "process": {}}}}

    for policy in ("warn", "ignore"):
        resolution = resolve_dispatch_plan(
            normalize_user_operation(collective_target, allow_pickle=True),
            world=world,
            requirement_policy=policy,
            single_worker_only=True,
        )
        assert resolution.launchable is False
        assert any(item.code == "dryml.dispatch.single_subprocess_topology_unsupported" for item in resolution.diagnostics)


def test_local_world_topology_remains_structural_under_relaxed_policies():
    world = {"roles": {"main": {"replicas": 1, "process": {}}}}

    for policy in ("warn", "ignore"):
        resolution = resolve_dispatch_plan(
            normalize_user_operation(collective_target, allow_pickle=True),
            world=world,
            requirement_policy=policy,
        )
        assert resolution.launchable is False
        assert any(item.code == "dryml.dispatch.local_world_topology_unsupported" for item in resolution.diagnostics)


def test_local_world_rejects_unenacted_role_process_settings(tmp_path):
    world = {
        "roles": {
            "trainer": {
                "replicas": 1,
                "process": {"environment": "other-python", "runtime": "other-runtime"},
            }
        }
    }

    with pytest.raises(DispatchPlanningError, match="cannot enact role-specific process"):
        Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan_world(
            make_function_call_spec("operator:add", args=[1, 2]),
            world=world,
            inventory=LocalResourceInventory((0,)),
            requirement_policy="ignore",
        )


def test_oversubscribed_local_world_has_bounded_expansion():
    from dryml.dispatch.local_world import allocate_local_world

    world = {"roles": {"main": {"replicas": 4097, "process": {}}}}

    with pytest.raises(DispatchPlanningError, match="worker count exceeds"):
        allocate_local_world(world, inventory=LocalResourceInventory((0,)), oversubscribe=True)


def test_local_world_rejects_zero_worker_world_and_subprocess_enacts_process_env(tmp_path):
    from dryml.dispatch.local_world import allocate_local_world

    with pytest.raises(DispatchPlanningError, match="requires at least one worker"):
        allocate_local_world({"roles": {"main": {"replicas": 0, "process": {}}}}, inventory=LocalResourceInventory((0,)))

    world = {"roles": {"main": {"replicas": 1, "process": {"env": {"AUDIT_PROCESS_ENV": "enabled"}}}}}
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        make_function_call_spec("operator:add", args=[1, 2]),
        world=world,
        inventory=LocalResourceInventory((0,)),
        requirement_policy="ignore",
    )

    assert plan.envelope.allocation_view["env"]["AUDIT_PROCESS_ENV"] == "enabled"


def test_planning_metadata_bounds_deep_nested_data():
    from dryml.dispatch.requirements import _bounded_data

    value = "leaf"
    for _ in range(1100):
        value = {"nested": value}

    bounded = _bounded_data(value)
    for _ in range(9):
        bounded = bounded["nested"]
    assert bounded == {"__dryml_truncated__": "depth_or_size"}
