import sys

import dryml
from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec


def test_capture_reporter_sees_dispatch_lifecycle(tmp_path, target_module):
    capture = dryml.reporting.CaptureReporter()
    dryml.configure(reporting={"level": "details", "reporter": capture})
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:add", args=[1, 2]))

    Dispatcher(store=store).run(op, environment=env)
    names = [event.name for event in capture.events]

    assert "dryml.dispatch.plan.start" in names
    assert "dryml.dispatch.worker.launch" in names
    assert "dryml.dispatch.complete" in names
