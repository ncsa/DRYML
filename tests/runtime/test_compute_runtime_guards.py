import pytest

from dryml.core.object import Compute
from dryml.runtime import RuntimeAllocationView, RuntimeMode, enter_runtime
from dryml.runtime.errors import NoAllocationError


class RuntimeGuardedCompute(Compute):
    pass


def test_compute_pre_init_requires_runtime_allocation():
    with enter_runtime(RuntimeMode.ORCHESTRATOR, enforcement="strict"):
        with pytest.raises(NoAllocationError):
            RuntimeGuardedCompute.__pre_init__()


def test_compute_pre_init_accepts_cpu_only_worker_allocation():
    allocation = RuntimeAllocationView(cpus=(0, 1))

    with enter_runtime(RuntimeMode.WORKER, allocation):
        RuntimeGuardedCompute.__pre_init__()
