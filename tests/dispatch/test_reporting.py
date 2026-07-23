import sys

import dryml
from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher, LocalResourceInventory
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec


def test_capture_reporter_sees_local_world_lifecycle(tmp_path, target_module):
    capture = dryml.reporting.CaptureReporter()
    dryml.configure(reporting={"level": "details", "reporter": capture})
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}

    Dispatcher(store=store).run_world(op, world=world, environment=env, inventory=LocalResourceInventory(cpus=(0, 1)), timeout=10)
    names = [event.name for event in capture.events]

    assert "dryml.dispatch.world.plan.start" in names
    assert "dryml.dispatch.world.allocate" in names
    assert "dryml.dispatch.world.allocation.write" in names
    assert "dryml.dispatch.world.launch" in names
    assert "dryml.dispatch.world.handshake.wait" in names
    assert "dryml.dispatch.world.start" in names
    assert "dryml.dispatch.world.complete" in names
