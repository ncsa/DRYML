from __future__ import annotations

import sys

import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.dispatch.errors import DispatchPlanningError
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec, make_method_call_spec


def _env(target_module):
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()


def test_explicit_function_call_operation_spec_remains_compatible(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    op = attach_operation_id(make_function_call_spec("dispatch_target:add", args=[2, 3], metadata={"owner": "user"}))

    plan = Dispatcher(store=store).plan(op, environment=_env(target_module))

    assert plan.envelope.operation_spec["payload"] == op["payload"]
    assert plan.envelope.operation_spec["metadata"]["owner"] == "user"
    assert plan.envelope.operation_spec["metadata"]["dryml.dispatch.transport"] == "operation_spec"

    result = Dispatcher(store=store).run(op, environment=_env(target_module))
    assert result.status == "ok"
    assert result.result_canonical == 5


def test_explicit_operation_spec_preserves_user_metadata_but_replaces_reserved_keys(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    op = attach_operation_id(
        make_function_call_spec(
            "operator:add",
            args=[1, 2],
            metadata={
                "owner": "user",
                "dryml.dispatch.transport": "stale",
                "dryml.code_target": {"kind": "stale", "import_path": "stale:target"},
            },
        )
    )

    plan = Dispatcher(store=store).plan(op)
    metadata = plan.envelope.operation_spec["metadata"]

    assert metadata["owner"] == "user"
    assert metadata["dryml.dispatch.transport"] == "operation_spec"
    assert metadata["dryml.dispatch.user_target_kind"] == "operation_spec"
    assert metadata["dryml.code_target"]["kind"] == "import_path"
    assert metadata["dryml.code_target"]["import_path"] == "operator:add"


def test_explicit_method_call_operation_spec_still_plans(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    op = attach_operation_id(make_method_call_spec("cdef-v4-" + "0" * 64, "plus", args=[1]))

    plan = Dispatcher(store=store).plan(op)

    assert plan.envelope.operation_spec["kind"] == "method_call"
    assert plan.envelope.operation_spec["payload"] == op["payload"]
    assert plan.envelope.operation_spec["metadata"]["dryml.code_target"]["kind"] == "definition_method"


def test_explicit_operation_spec_rejects_method_name_and_extra_args(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    op = attach_operation_id(make_function_call_spec("operator:add", args=[1, 2]))

    with pytest.raises(DispatchPlanningError, match="method_name"):
        Dispatcher(store=store).plan(op, "train")
    with pytest.raises(DispatchPlanningError, match="already contains arguments"):
        Dispatcher(store=store).plan(op, args=(3,))
    with pytest.raises(DispatchPlanningError, match="function_call requires function"):
        Dispatcher(store=store).plan({"schema": "dryml.operation.v1", "schema_version": 1, "kind": "function_call", "payload": {}})
