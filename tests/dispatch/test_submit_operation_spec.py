from __future__ import annotations

import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.dispatch.errors import DispatchPlanningError
from dryml.operations import attach_operation_id, make_function_call_spec, make_method_call_spec


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


def test_explicit_method_call_operation_spec_requires_resolvable_stored_subject(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    op = attach_operation_id(make_method_call_spec("cdef-v4-" + "0" * 64, "plus", args=[1]))

    with pytest.raises(DispatchPlanningError, match="not launchable"):
        Dispatcher(store=store).plan(op)


def test_explicit_operation_spec_rejects_method_name_and_extra_args(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    op = attach_operation_id(make_function_call_spec("operator:add", args=[1, 2]))

    with pytest.raises(DispatchPlanningError, match="method_name"):
        Dispatcher(store=store).plan(op, "train")
    with pytest.raises(DispatchPlanningError, match="already contains arguments"):
        Dispatcher(store=store).plan(op, args=(3,))
    with pytest.raises(DispatchPlanningError, match="function_call requires function"):
        Dispatcher(store=store).plan({"schema": "dryml.operation.v1", "schema_version": 1, "kind": "function_call", "payload": {}})
