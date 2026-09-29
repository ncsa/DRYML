import pytest

import dryml
from dryml.runtime import NoAllocation, RuntimeAllocationView, RuntimeMode, RuntimeState
from dryml.runtime.publication import publication as publication_import_path
from dryml.runtime.errors import RuntimeTransitionError


def test_only_supported_runtime_modes_are_exported_and_none_is_passive():
    assert dryml.runtime.RuntimeMode is RuntimeMode
    assert publication_import_path is dryml.runtime.publication
    assert {mode.name for mode in RuntimeMode} == {"NONE", "ORCHESTRATOR", "INLINE"}
    assert not hasattr(RuntimeMode, "PROBE") and not hasattr(RuntimeMode, "WORKER")
    assert RuntimeState().mode is RuntimeMode.NONE
    assert RuntimeState().allocation is NoAllocation
    with pytest.raises(ValueError):
        RuntimeMode.coerce("worker")


def test_inline_requires_and_orchestrator_rejects_an_exact_allocation():
    with pytest.raises(RuntimeTransitionError):
        RuntimeState(RuntimeMode.INLINE)
    allocation = RuntimeAllocationView(role="main", replica=0, cpus=(0,))
    assert RuntimeState(RuntimeMode.INLINE, allocation).allocation is allocation
    with pytest.raises(RuntimeTransitionError):
        RuntimeState(RuntimeMode.ORCHESTRATOR, allocation)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"role": "main", "replica": 0, "rank": -1},
        {"role": "main", "replica": 0, "local_rank": -1},
        {"role": "main", "replica": 0, "memory": -1},
        {"role": "main", "replica": 0, "world_allocation_id": "worldalloc-v1.1-invalid"},
        {"role": "main", "replica": 0, "accelerators": {"gpu": (0,)}, "accelerator_memory": {"gpu": {1: 1}}},
    ),
)
def test_runtime_allocation_view_rejects_malformed_exact_facts(kwargs):
    with pytest.raises(RuntimeTransitionError):
        RuntimeAllocationView(**kwargs)
