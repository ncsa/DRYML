import pytest

import dryml.runtime as runtime
from dryml.runtime.errors import RuntimeTransitionError


def test_default_runtime_is_orchestrator_no_allocation():
    state = runtime.active_runtime()
    assert state.mode is runtime.RuntimeMode.NONE
    assert state.allocation is runtime.NoAllocation


def test_nested_enter_runtime_resets_state():
    outer_alloc = runtime.RuntimeAllocationView(role="outer", cpus=(0,))
    inner_alloc = runtime.RuntimeAllocationView(role="inner", cpus=(1,))

    with runtime.enter_runtime(runtime.RuntimeMode.WORKER, outer_alloc):
        assert runtime.active_runtime().allocation.role == "outer"
        with runtime.enter_runtime(runtime.RuntimeMode.INLINE, inner_alloc):
            assert runtime.active_runtime_mode() is runtime.RuntimeMode.INLINE
            assert runtime.active_runtime().allocation.role == "inner"
        assert runtime.active_runtime().allocation.role == "outer"
    assert runtime.active_runtime().allocation is runtime.NoAllocation


def test_invalid_transitions_fail_clearly():
    with pytest.raises(RuntimeTransitionError):
        with runtime.enter_runtime(runtime.RuntimeMode.WORKER):
            pass
    with pytest.raises(RuntimeTransitionError):
        with runtime.enter_runtime(runtime.RuntimeMode.PROBE, runtime.RuntimeAllocationView(cpus=(0,))):
            pass
