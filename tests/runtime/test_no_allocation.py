import pytest

import dryml.runtime as runtime
from dryml.runtime.errors import NoAllocationError, RuntimeTransitionError


def test_require_allocation_raises_in_orchestrator_and_probe():
    with pytest.raises(NoAllocationError) as excinfo:
        runtime.require_allocation("training")
    assert excinfo.value.context["mode"] == "orchestrator"
    assert "enter worker/inline" in excinfo.value.context["fix"]

    with runtime.enter_runtime(runtime.RuntimeMode.PROBE):
        with pytest.raises(NoAllocationError):
            runtime.require_allocation("probe-workload")


def test_worker_and_inline_allocation_success_and_cpu_only_not_no_allocation():
    cpu_only = runtime.RuntimeAllocationView(role="trainer", cpus=(0, 1), accelerators={})
    assert not cpu_only.is_no_allocation

    with runtime.enter_runtime(runtime.RuntimeMode.WORKER, cpu_only):
        assert runtime.require_worker_allocation().cpus == (0, 1)
    with runtime.enter_runtime(runtime.RuntimeMode.INLINE, cpu_only):
        assert runtime.require_allocation().cpus == (0, 1)


def test_assert_no_workload_allocation():
    runtime.assert_no_workload_allocation()
    with runtime.enter_runtime(runtime.RuntimeMode.WORKER, runtime.RuntimeAllocationView(cpus=(0,))):
        with pytest.raises(RuntimeTransitionError):
            runtime.assert_no_workload_allocation()
