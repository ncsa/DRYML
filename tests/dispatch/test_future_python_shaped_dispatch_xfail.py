from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher


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


def test_dispatch_submit_importable_function_builds_function_call_spec(tmp_path):
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(targets.plain_importable_function)
    assert plan.envelope.operation_spec["kind"] == "function_call"
    assert plan.envelope.operation_spec["payload"]["function"].endswith(":plain_importable_function")
    assert plan.envelope.launch.get("call_transport") != "pickle_small"


def test_dispatch_submit_lambda_with_allow_pickle_embeds_plan_time_metadata(tmp_path):
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        targets.local_lambda_with_annotation, allow_pickle=True, requirement_policy="ignore"
    )
    assert plan.envelope.launch["call_transport"] == "pickle_small"
    assert "callable_metadata" in plan.envelope.launch
    assert plan.envelope.launch["callable_metadata"]["qualname"] == "<lambda>"
