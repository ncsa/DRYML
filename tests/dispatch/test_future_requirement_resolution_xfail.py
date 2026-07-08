from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher


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


@pytest.mark.xfail(reason="Sprint 7: dispatch requirement resolution not implemented yet", strict=True)
def test_function_level_env_requirement_is_collected_during_dispatch_planning(tmp_path):
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(targets.run_training, allow_pickle=True)
    requirements = plan.dispatch_spec["payload"]["requirements"]["environment"]["requirements"]
    assert "pandas>=2" in requirements


@pytest.mark.xfail(reason="Sprint 3/Sprint 7: class + method requirement resolution not implemented yet", strict=True)
def test_dispatch_resolves_class_and_method_requirements(tmp_path):
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(targets.LightningModel(), "train")
    resolved = plan.dispatch_spec["payload"]["requirements"]
    assert "torch>=2" in resolved["environment"]["requirements"]
    assert "lightning>=2" in resolved["environment"]["requirements"]
    assert resolved["world"]["roles"]["main"]["resources"]["accelerators"]["gpu"]["min"] == 1


@pytest.mark.xfail(reason="Sprint 7: explicit world candidate compatibility check not implemented yet", strict=True)
def test_dispatch_explicit_world_must_satisfy_hard_world_requirement(tmp_path):
    cpu_only_world = {"policy": "single_worker", "spec": {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}}}
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(targets.LightningModel().train, allow_pickle=True, world=cpu_only_world)
    assert plan.dispatch_spec["payload"]["requirement_report"]["ok"] is False


@pytest.mark.xfail(reason="Sprint 3: precise classmethod/staticmethod collection not implemented yet", strict=True)
def test_classmethod_and_staticmethod_both_decorator_orders_are_collected(tmp_path):
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(targets.ClassMethodTargets.outer_decorated, allow_pickle=True)
    requirements = plan.dispatch_spec["payload"]["requirements"]["environment"]["requirements"]
    assert "outer-classmethod>=1" in requirements
