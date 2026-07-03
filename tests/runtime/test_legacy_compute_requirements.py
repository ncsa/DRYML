import pytest

from dryml.core2.object import Compute
from dryml.runtime import RuntimeAllocationView, RuntimeMode, enter_runtime
from dryml.runtime.errors import NoAllocationError


class GpuCompute(Compute):
    __compute_reqs__ = {"torch": {"num_gpus": 1}}


def test_compute_requirement_fails_with_no_allocation():
    with pytest.raises(NoAllocationError):
        GpuCompute.__pre_init__()


def test_compute_requirement_fails_with_cpu_only_allocation():
    allocation = RuntimeAllocationView(cpus=(0, 1))

    with enter_runtime(RuntimeMode.WORKER, allocation):
        with pytest.raises(NoAllocationError) as excinfo:
            GpuCompute.__pre_init__()

    assert excinfo.value.context["failures"]["num_gpus"] == {"required": 1, "actual": 0}


def test_compute_requirement_passes_with_matching_gpu_allocation():
    allocation = RuntimeAllocationView(cpus=(0, 1), accelerators={"gpu": (0,)})

    with enter_runtime(RuntimeMode.WORKER, allocation):
        GpuCompute.__pre_init__()
