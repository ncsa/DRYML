import importlib
import os
import sys

from dryml import session
from dryml.core.repo import Repo
from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.formats.refs import format_cdef_id
from dryml.operations import attach_operation_id, make_function_call_spec, make_method_call_spec
from dryml.runtime import NoAllocation, active_runtime
from dryml.worlds import LocalResourceInventory


def _env(target_module):
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()


def test_method_call_materializes_subject(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(stores=[store])
    box = mod.Box(7)
    repo.save(box, store=store, record_policy="descriptive")
    cdef_id = format_cdef_id(box.definition.stable_hash())
    op = attach_operation_id(make_method_call_spec(cdef_id, "plus", args=[5]))

    result = Dispatcher(store=store).run(op, environment=_env(target_module))

    assert result.status == "ok"
    assert result.result_canonical == 12


def test_function_call_cdef_materialize_ref_and_literal(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(stores=[store])
    box = mod.Box(9)
    repo.save(box, store=store, record_policy="descriptive")
    cdef_id = format_cdef_id(box.definition.stable_hash())

    arguments = attach_operation_id(
        make_function_call_spec(
            "dispatch_target:argument_values",
            args=[cdef_id, f"ref({cdef_id})", {"$literal": cdef_id}],
        )
    )

    result = Dispatcher(store=store).run(arguments, environment=_env(target_module))
    assert result.result_canonical[0] == 9
    assert result.result_canonical[1] == cdef_id
    assert result.result_canonical[2] == cdef_id


def _run_orchestrated_cdef_dispatch(tmp_path, target_module, monkeypatch, *, gpus, inventory):
    import dryml.session.state as session_state

    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(stores=[store])
    box = mod.Box(9)
    repo.save(box, store=store, record_policy="descriptive")
    cdef_id = format_cdef_id(box.definition.stable_hash())
    operation = attach_operation_id(
        make_function_call_spec("dispatch_target:cdef_materialization_status", args=[cdef_id])
    )

    monkeypatch.setattr(session_state, "local_inventory", lambda: inventory)
    session.set_mode("orchestrator")
    session.worker_env_request(
        PythonExecutableSpec(
            sys.executable,
            pythonpath_policy="explicit",
            extra_pythonpath=(str(target_module.parent),),
        )
    )
    snapshot = session.worker_world_request(cpus=1, gpus=gpus)
    try:
        assert active_runtime().allocation is NoAllocation
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        plan = Dispatcher(store=store).plan(operation, inventory=inventory)
        assert plan.resolution.environment_selection.source == "session_requested"
        assert plan.resolution.world_selection.source == "session_requested"
        result = Dispatcher(store=store).run(operation, inventory=inventory)
    finally:
        session.reset()

    assert snapshot.runtime is not None
    assert snapshot.runtime.allocation is NoAllocation
    expected_accelerators = {} if gpus == 0 else {"gpu": ["synthetic-gpu-0"]}
    assert result.status == "ok"
    assert result.result_canonical == {
        "value": 9,
        "constructor_mode": "worker",
        "runtime_mode": "worker",
        "accelerators": expected_accelerators,
        "cuda_visible_devices": "" if gpus == 0 else "synthetic-gpu-0",
        "enforcement": "strict",
        "selected_environment": "python",
        "selected_world": ["worker"],
        "selected_runtime": "worker",
    }
    return plan, result


def test_strict_orchestrator_dispatch_materializes_cdef_in_cpu_worker(tmp_path, target_module, monkeypatch):
    plan, _ = _run_orchestrated_cdef_dispatch(
        tmp_path,
        target_module,
        monkeypatch,
        gpus=0,
        inventory=LocalResourceInventory((0,)),
    )

    resources = plan.resolution.world_selection.candidate["roles"]["worker"]["process"]["resources"]
    assert resources == {"cpus": 1}
    assert plan.envelope.allocation_view["accelerators"] == {}


def test_strict_orchestrator_dispatch_assigns_synthetic_accelerator_only_to_worker(tmp_path, target_module, monkeypatch):
    plan, result = _run_orchestrated_cdef_dispatch(
        tmp_path,
        target_module,
        monkeypatch,
        gpus=1,
        inventory=LocalResourceInventory((0,), {"gpu": ("synthetic-gpu-0",)}),
    )

    assert plan.envelope.allocation_view["accelerators"] == {"gpu": ["synthetic-gpu-0"]}
    assert result.result_canonical["runtime_mode"] == "worker"
