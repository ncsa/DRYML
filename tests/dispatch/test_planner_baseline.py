from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import dryml.operations as ops
from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.dispatch.errors import DispatchPlanningError


def _load_targets():
    path = Path(__file__).parents[1] / "fixtures" / "requirements_targets.py"
    spec = importlib.util.spec_from_file_location("dryml_requirement_targets", path)
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


targets = _load_targets()


def _store(tmp_path):
    return DirStore(tmp_path / "store", query_index="none")


def _operation():
    return ops.make_function_call_spec("operator:add", args=[1, 2])


def test_plan_defaults_environment_to_current_environment_spec_data(tmp_path):
    plan = Dispatcher(store=_store(tmp_path)).plan(_operation())

    assert plan.dispatch_spec["payload"]["environment"]["policy"] == "current"
    assert plan.dispatch_spec["payload"]["environment"]["spec"]["kind"] == "current"
    assert plan.envelope.environment_spec["kind"] == "current"


def test_plan_defaults_world_to_single_worker_policy(tmp_path):
    plan = Dispatcher(store=_store(tmp_path)).plan(_operation())

    assert plan.dispatch_spec["payload"]["world"] == {"policy": "single_worker"}


def test_plan_accepts_explicit_environment_and_world(tmp_path):
    environment = {"kind": "current", "metadata": {"name": "baseline"}}
    world = {"policy": "explicit-test"}

    plan = Dispatcher(store=_store(tmp_path)).plan(_operation(), environment=environment, world=world)

    assert plan.dispatch_spec["payload"]["environment"]["spec"] == environment
    assert plan.envelope.environment_spec == environment
    assert plan.dispatch_spec["payload"]["world"] == world


def test_plan_accepts_operation_spec_mapping_and_attaches_operation_id(tmp_path):
    plan = Dispatcher(store=_store(tmp_path)).plan(_operation())

    operation = plan.envelope.operation_spec

    assert operation["kind"] == "function_call"
    assert operation["id"].startswith("op-v1-")
    assert plan.dispatch_spec["payload"]["operation_id"] == operation["id"]


def test_callable_without_allow_pickle_raises_planning_error(tmp_path):
    with pytest.raises(DispatchPlanningError, match="allow_pickle=True"):
        Dispatcher(store=_store(tmp_path)).plan(targets.plain_importable_function)


def test_callable_with_allow_pickle_uses_existing_pickle_transport(tmp_path):
    plan = Dispatcher(store=_store(tmp_path)).plan(targets.plain_importable_function, allow_pickle=True, args=(2,))

    assert plan.envelope.launch["call_transport"] == "pickle_small"
    assert plan.execution_recipe["payload"]["constraints"]["portable"] is False
    assert plan.envelope.operation_spec["payload"]["function"] == "dryml.dispatch.operations:import_function"


def test_planner_report_contains_requirement_gather_merge_steps(monkeypatch, tmp_path):
    import dryml.reporting as reporting

    names = []

    def record(name, message, **kwargs):
        names.append(name)

    monkeypatch.setattr(reporting, "step", record)
    monkeypatch.setattr(reporting, "detail", record)

    Dispatcher(store=_store(tmp_path)).plan(_operation())

    assert "dryml.dispatch.requirements.gather" in names
    assert "dryml.dispatch.requirements.merge" in names


def test_planner_does_not_resolve_target_annotations_yet(tmp_path):
    plan = Dispatcher(store=_store(tmp_path)).plan(targets.run_training, allow_pickle=True)

    assert "pandas>=2" not in repr(plan.dispatch_spec)
    assert plan.dispatch_spec["payload"]["environment"]["policy"] == "current"
