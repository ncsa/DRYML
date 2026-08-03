from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import dryml.operations as ops
from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher, PickledCallable
from dryml.environments import CurrentEnvironmentSpec


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

    assert plan.dispatch_spec["payload"]["environment"]["policy"] == "fallback"
    assert plan.dispatch_spec["payload"]["environment"]["spec"]["kind"] == "current"
    assert plan.envelope.environment_spec["kind"] == "current"


def test_plan_defaults_world_to_checked_single_worker_fallback(tmp_path):
    plan = Dispatcher(store=_store(tmp_path)).plan(_operation())

    world = plan.dispatch_spec["payload"]["world"]
    assert world["policy"] == "fallback"
    assert world["spec"]["roles"]["main"]["replicas"] == 1


def test_plan_accepts_explicit_environment_and_world(tmp_path):
    environment = CurrentEnvironmentSpec().to_data()
    world = {"roles": {"main": {"replicas": 1, "process": {}}}}

    plan = Dispatcher(store=_store(tmp_path)).plan(_operation(), environment=environment, world=world)

    assert plan.dispatch_spec["payload"]["environment"]["spec"]["kind"] == "current"
    assert plan.envelope.environment_spec["kind"] == "current"
    assert plan.dispatch_spec["payload"]["world"]["policy"] == "explicit"
    assert plan.dispatch_spec["payload"]["world"]["spec"]["roles"]["main"]["replicas"] == 1


def test_plan_accepts_operation_spec_mapping_and_attaches_operation_id(tmp_path):
    plan = Dispatcher(store=_store(tmp_path)).plan(_operation())

    operation = plan.envelope.operation_spec

    assert operation["kind"] == "function_call"
    assert operation["id"].startswith("op-v1-")
    assert plan.dispatch_spec["payload"]["operation_id"] == operation["id"]


def test_importable_callable_without_allow_pickle_uses_import_path(tmp_path):
    plan = Dispatcher(store=_store(tmp_path)).plan(targets.plain_importable_function)

    assert plan.envelope.launch["call_transport"] == "import_ref"
    assert plan.envelope.operation_spec["payload"]["function"] == "dryml_requirement_targets:plain_importable_function"
    assert plan.execution_recipe["payload"]["constraints"]["portable"] is True


def test_explicit_pickled_callable_uses_existing_pickle_transport(tmp_path):
    plan = Dispatcher(store=_store(tmp_path)).plan(PickledCallable(targets.plain_importable_function), allow_pickle=True, args=(2,))

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


def test_planner_resolves_target_annotations_into_authoritative_metadata(tmp_path):
    plan = Dispatcher(store=_store(tmp_path)).plan(
        targets.run_training,
        allow_pickle=True,
        requirement_policy="ignore",
    )

    requirements = plan.dispatch_spec["payload"]["metadata"]["dryml.requirements"]
    assert "pandas>=2" in requirements["environment_requirement"]["requirements"]
    assert plan.dispatch_spec["payload"]["environment"]["policy"] == plan.resolution.environment_selection.source


def test_ordinary_planning_does_not_request_dynamic_tracing(monkeypatch, tmp_path):
    """Default planning keeps the opt-in trace modality out of its analysis call."""

    import dryml.dispatch.requirements as dispatch_requirements

    original_analyze = dispatch_requirements.analyze
    contexts = []

    def analyze_without_trace(*args, **kwargs):
        context = kwargs["context"]
        contexts.append(context)
        assert context.allow_dynamic_execution is False
        assert "dynamic_trace" not in context.algorithms
        return original_analyze(*args, **kwargs)

    monkeypatch.setattr(dispatch_requirements, "analyze", analyze_without_trace)
    Dispatcher(store=_store(tmp_path)).plan(
        targets.run_training,
        allow_pickle=True,
        requirement_policy="ignore",
    )

    assert contexts
