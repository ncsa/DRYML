from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from dryml.core2.repo import Repo
from dryml.core2.store.dir import DirStore
from dryml.dispatch import PickledCallable, normalize_user_operation
from dryml.dispatch.errors import DispatchPlanningError
from dryml.operations import attach_operation_id, make_function_call_spec


def _load_requirement_targets():
    path = Path(__file__).parents[1] / "fixtures" / "requirements_targets.py"
    spec = importlib.util.spec_from_file_location("dryml_requirement_targets", path)
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CallableInstance:
    def __call__(self, value):
        return value + 1


class BoundMethodTarget:
    def method(self):
        return "bound"


def test_module_level_function_builds_import_path_spec_and_metadata():
    targets = _load_requirement_targets()

    normalized = normalize_user_operation(targets.plain_importable_function, args=[1], kwargs={"flag": True})

    assert normalized.operation_spec["kind"] == "function_call"
    assert normalized.operation_spec["payload"]["function"] == "dryml_requirement_targets:plain_importable_function"
    assert normalized.operation_spec["payload"]["args"] == [1]
    assert normalized.operation_spec["payload"]["kwargs"] == {"flag": True}
    assert normalized.transport == "import_path"
    assert normalized.live_annotation_targets == (targets.plain_importable_function,)
    metadata = normalized.operation_spec["metadata"]
    assert metadata["dryml.dispatch.normalized"] is True
    assert metadata["dryml.dispatch.transport"] == "import_path"
    assert metadata["dryml.code_target"]["import_path"] == "dryml_requirement_targets:plain_importable_function"


def test_non_importable_callables_fail_without_pickle_and_work_with_pickle():
    targets = _load_requirement_targets()
    local_function = targets.make_local_training_function()

    with pytest.raises(DispatchPlanningError, match="allow_pickle=True") as exc_info:
        normalize_user_operation(local_function)
    assert exc_info.value.context["reason"] == "local_function"

    pickled = normalize_user_operation(local_function, allow_pickle=True, args=("x",))
    assert pickled.transport == "pickle_small"
    assert pickled.launch["call_transport"] == "pickle_small"
    assert pickled.launch["same_environment_only"] is True
    assert pickled.launch["callable_metadata"]["qualname"].endswith("local_training")
    assert pickled.live_annotation_targets == (local_function,)

    with pytest.raises(DispatchPlanningError, match="allow_pickle=True") as lambda_exc:
        normalize_user_operation(targets.local_lambda_with_annotation)
    assert lambda_exc.value.context["reason"] == "lambda"
    assert normalize_user_operation(targets.local_lambda_with_annotation, allow_pickle=True).transport == "pickle_small"


def test_callable_instance_and_bound_method_policy_is_explicit():
    with pytest.raises(DispatchPlanningError, match="allow_pickle=True") as exc_info:
        normalize_user_operation(CallableInstance())
    assert exc_info.value.context["reason"] == "missing_module_or_qualname"

    assert normalize_user_operation(CallableInstance(), allow_pickle=True).transport == "pickle_small"

    with pytest.raises(DispatchPlanningError, match="bound instance method"):
        normalize_user_operation(BoundMethodTarget().method, allow_pickle=True)

    explicit = normalize_user_operation(PickledCallable(BoundMethodTarget().method), args=(2,))
    assert explicit.transport == "pickle_small"
    assert explicit.launch["same_environment_only"] is True
    assert explicit.live_annotation_targets[0].__self__.__class__ is BoundMethodTarget


def test_cdef_checked_before_mapping_and_method_metadata_populated(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(stores=[store])
    box = mod.Box(3)
    repo.save(box, store=store, record_policy="none")

    normalized = normalize_user_operation(box.definition, "plus", store=store, args=(4,))

    assert normalized.operation_spec["kind"] == "method_call"
    assert normalized.operation_spec["payload"]["method"] == "plus"
    assert normalized.operation_spec["payload"]["subject"].startswith("cdef-v4-")
    assert normalized.transport == "method_call"
    assert normalized.subject_class in (None, mod.Box)
    assert normalized.operation_spec["metadata"]["dryml.code_target"]["kind"] == "definition_method"


def test_operation_spec_path_and_invalid_mapping_errors():
    op = attach_operation_id(make_function_call_spec("operator:add", args=[1, 2], metadata={"user": "kept"}))
    normalized = normalize_user_operation(op)
    assert normalized.operation_spec["payload"] == op["payload"]
    assert normalized.operation_spec["metadata"]["user"] == "kept"
    assert normalized.transport == "operation_spec"
    assert normalized.code_target.import_path == "operator:add"

    with pytest.raises(DispatchPlanningError, match="method_name"):
        normalize_user_operation(op, "train")
    with pytest.raises(DispatchPlanningError, match="already contains arguments"):
        normalize_user_operation(op, args=(1,))
    with pytest.raises(DispatchPlanningError, match="function_call requires function"):
        normalize_user_operation({"schema": "dryml.operation.v1", "schema_version": 1, "kind": "function_call", "payload": {}})
    with pytest.raises(DispatchPlanningError, match="unsupported dispatch target"):
        normalize_user_operation(123)


def test_argument_and_method_name_validation():
    targets = _load_requirement_targets()
    assert normalize_user_operation(targets.plain_importable_function, args=[1], kwargs={"x": 2}).operation_spec["payload"]["args"] == [1]
    with pytest.raises(DispatchPlanningError, match="args"):
        normalize_user_operation(targets.plain_importable_function, args="not-a-tuple")
    with pytest.raises(DispatchPlanningError, match="kwargs"):
        normalize_user_operation(targets.plain_importable_function, kwargs=[("x", 1)])
    with pytest.raises(DispatchPlanningError, match="method_name must be a string"):
        normalize_user_operation(targets.plain_importable_function, 4)
    with pytest.raises(DispatchPlanningError, match="method_name must not be empty"):
        normalize_user_operation(targets.plain_importable_function, "")
    with pytest.raises(DispatchPlanningError, match="only valid"):
        normalize_user_operation(targets.plain_importable_function, "train")


def test_pickled_normalization_cleans_temporary_directory_on_serialization_failure(tmp_path, monkeypatch):
    import dryml.dispatch.normalize as normalize_module

    work_dir = tmp_path / "pickle-work"
    monkeypatch.setattr(normalize_module.tempfile, "mkdtemp", lambda **_kwargs: str(work_dir))

    def fail_after_creating_artifact(*_args, **_kwargs):
        work_dir.mkdir()
        raise RuntimeError("pickle failed")

    monkeypatch.setattr(normalize_module, "write_pickled_callable", fail_after_creating_artifact)

    with pytest.raises(RuntimeError, match="pickle failed"):
        normalize_user_operation(CallableInstance(), allow_pickle=True)

    assert not work_dir.exists()
