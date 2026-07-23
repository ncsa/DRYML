from __future__ import annotations

import importlib
import sys

import pytest

from dryml.core.definition import Definition
from dryml.core.repo import Repo
from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher, run
from dryml.dispatch.errors import DispatchPlanningError
from dryml.environments import PythonExecutableSpec


def _env(target_module):
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()


def test_cdef_method_plan_and_run(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(stores=[store])
    box = mod.Box(10)
    repo.save(box, store=store, record_policy="none")

    planned = Dispatcher(store=store).plan(box.definition, "plus", args=(7,), environment=_env(target_module))

    assert planned.envelope.operation_spec["kind"] == "method_call"
    assert planned.envelope.operation_spec["payload"]["method"] == "plus"
    assert planned.envelope.operation_spec["payload"]["subject"].startswith("cdef-v4-")
    assert planned.envelope.operation_spec["metadata"]["dryml.dispatch.transport"] == "method_call"
    assert planned.envelope.operation_spec["metadata"]["dryml.code_target"]["method_name"] == "plus"

    result = Dispatcher(store=store).run(box.definition, "plus", args=(7,), environment=_env(target_module))
    assert result.status == "ok"
    assert result.result_canonical == 17


def test_object_instance_method_path_persists_before_dispatch(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    box = mod.Box(11)

    result = run(box, "plus", store=store, args=(6,), environment=_env(target_module))

    assert result.status == "ok"
    assert result.result_canonical == 17


def test_definition_or_cdef_method_errors_are_actionable(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    box = mod.Box(1)

    with pytest.raises(DispatchPlanningError, match="method_name is required"):
        Dispatcher(store=store).plan(box.definition)
    with pytest.raises(DispatchPlanningError, match="method_name must be a string"):
        Dispatcher(store=store).plan(box.definition, 5)
    with pytest.raises(DispatchPlanningError, match="method_name must not be empty"):
        Dispatcher(store=store).plan(box.definition, "")
    with pytest.raises(DispatchPlanningError, match="subject CDef is not present"):
        Dispatcher(store=store).plan(box.definition, "plus")


def test_definition_and_cdef_method_dispatch_without_store_is_actionable(target_module):
    mod = importlib.import_module("dispatch_target")
    subjects = ((Definition(mod.Box, 1), "definition_method"), (mod.Box(1).definition, "cdef_method"))

    for entrypoint in ("plan", "plan_world"):
        for subject, target_form in subjects:
            with pytest.raises(DispatchPlanningError) as exc_info:
                getattr(Dispatcher(), entrypoint)(subject, "plus")

            assert "Store" in str(exc_info.value)
            assert "materializ" in str(exc_info.value)
            assert "execut" in str(exc_info.value)
            assert exc_info.value.context == {"reason": "store_required", "target_form": target_form}


def test_definition_method_dispatch_requires_existing_stored_subject(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    box = mod.Box(2)

    with pytest.raises(DispatchPlanningError, match="subject CDef is not present"):
        Dispatcher(store=store).plan(box.definition, "plus", environment=_env(target_module))
