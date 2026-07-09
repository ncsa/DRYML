from __future__ import annotations

import json

import dryml

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.dispatch.errors import DispatchPlanningError


@dryml.world.req(accelerators={"gpu": {"min": 1}})
def gpu_target():
    return None


def test_explain_returns_nonlaunching_structured_failure_with_plan_parity(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"roles": {"main": {"replicas": 1, "process": {}}}}
    dispatcher = Dispatcher(store=store)

    explanation = dispatcher.explain(gpu_target, allow_pickle=True, world=world)
    assert explanation.launchable is False
    assert explanation.resolution.world_check.status == "incompatible"
    assert json.loads(json.dumps(explanation.to_data()))["launchable"] is False
    with __import__("pytest").raises(DispatchPlanningError, match="not launchable"):
        dispatcher.plan(gpu_target, allow_pickle=True, world=world)


def test_explain_does_not_write_operation_records_for_importable_target(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")

    explanation = Dispatcher(store=store).explain(gpu_target, allow_pickle=True, requirement_policy="ignore")
    assert explanation.launchable is True
    assert not store.records.specs_dir.exists()
