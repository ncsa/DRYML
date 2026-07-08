import pytest

import dryml.runtime as runtime
from dryml.runtime.errors import RuntimeTransitionError


def _allocation():
    return runtime.RuntimeAllocationView(role="worker", replica=0, rank=0, local_rank=0, cpus=(0,))


def test_default_runtime_state_current_behavior():
    state = runtime.active_runtime()

    assert state.mode is runtime.RuntimeMode.ORCHESTRATOR
    assert state.allocation is runtime.NoAllocation


def test_runtime_modes_exist():
    assert runtime.RuntimeMode.PROBE.value == "probe"
    assert runtime.RuntimeMode.ORCHESTRATOR.value == "orchestrator"
    assert runtime.RuntimeMode.WORKER.value == "worker"
    assert runtime.RuntimeMode.INLINE.value == "inline"


def test_probe_and_orchestrator_reject_workload_allocation():
    for mode in (runtime.RuntimeMode.PROBE, runtime.RuntimeMode.ORCHESTRATOR):
        with pytest.raises(RuntimeTransitionError, match="must not hold workload allocation"):
            with runtime.enter_runtime(mode, allocation=_allocation()):
                pass


def test_worker_and_inline_require_allocation():
    for mode in (runtime.RuntimeMode.WORKER, runtime.RuntimeMode.INLINE):
        with pytest.raises(RuntimeTransitionError, match="requires an explicit allocation"):
            with runtime.enter_runtime(mode):
                pass


def test_worker_and_inline_accept_real_allocation():
    for mode in (runtime.RuntimeMode.WORKER, runtime.RuntimeMode.INLINE):
        with runtime.enter_runtime(mode, allocation=_allocation()) as state:
            assert state.mode is mode
            assert state.allocation is not runtime.NoAllocation


def test_runtime_context_resets_after_exit():
    before = runtime.active_runtime()
    with runtime.enter_runtime(runtime.RuntimeMode.PROBE) as state:
        assert state.mode is runtime.RuntimeMode.PROBE
    assert runtime.active_runtime() == before


def test_nested_runtime_contexts_reset_in_order():
    before = runtime.active_runtime()
    with runtime.enter_runtime(runtime.RuntimeMode.PROBE):
        assert runtime.active_runtime_mode() is runtime.RuntimeMode.PROBE
        with runtime.enter_runtime(runtime.RuntimeMode.WORKER, allocation=_allocation()):
            assert runtime.active_runtime_mode() is runtime.RuntimeMode.WORKER
        assert runtime.active_runtime_mode() is runtime.RuntimeMode.PROBE
    assert runtime.active_runtime() == before
