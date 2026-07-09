from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, PickledCallable


pytestmark = pytest.mark.future_behavior


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


def test_function_level_env_requirement_is_collected_during_dispatch_planning(tmp_path):
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(targets.run_training, allow_pickle=True)
    requirements = plan.dispatch_spec["payload"]["metadata"]["dryml.requirements"]["environment_requirement"]["requirements"]
    assert "pandas>=2" in requirements


def test_dispatch_resolves_class_and_method_requirements(tmp_path):
    target = PickledCallable(targets.LightningModel().train)
    explanation = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).explain(target, allow_pickle=True)
    resolved = explanation.requirements.to_data()
    assert "torch>=2" in resolved["environment_requirement"]["requirements"]
    assert "lightning>=2" in resolved["environment_requirement"]["requirements"]
    assert resolved["world_requirement"]["roles"]["main"]["resources"]["accelerators"]["gpu"]["min"] == 1


def test_dispatch_explicit_world_must_satisfy_hard_world_requirement(tmp_path):
    cpu_only_world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}}
    explanation = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).explain(
        PickledCallable(targets.LightningModel().train), allow_pickle=True, world=cpu_only_world
    )
    assert explanation.launchable is False
    assert explanation.resolution.world_check.status == "incompatible"


def test_classmethod_and_staticmethod_both_decorator_orders_are_collected(tmp_path):
    explanation = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).explain(targets.ClassMethodTargets.outer_decorated, allow_pickle=True)
    requirements = explanation.requirements.to_data()["environment_requirement"]["requirements"]
    assert "outer-classmethod>=1" in requirements
