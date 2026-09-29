"""Tests for world collection through the generic code scheduler."""

import pytest

from dryml.code import (
    KernelCall,
    StaticDependenciesKernel,
    capture_inspection,
    probe,
)
from dryml.worlds import WorldRequirementsKernel, req
from tests.managed.execution_fixtures import ManagedWorldMatrixValue, ORDERS


def test_world_kernel_collects_live_resolved_targets() -> None:
    """A scheduled domain kernel combines root and helper declarations once."""

    @req(cpus=1, source="root")
    def root() -> None:
        helper()

    @req(memory="1GiB", source="helper")
    def helper() -> None:
        return None

    result = probe(
        root,
        (
            KernelCall(StaticDependenciesKernel(), None),
            KernelCall(WorldRequirementsKernel(), None),
        ),
    )

    requirements = result.require(WorldRequirementsKernel)
    assert requirements.has_value
    assert requirements.value.roles["main"].resources.cpus.min == 1
    assert requirements.value.roles["main"].resources.memory.min == 1024**3


@pytest.mark.parametrize(("member", "written_order"), ORDERS)
def test_world_kernel_collects_managed_direct_carrier_declarations(
        member, written_order) -> None:
    """Each A/F/M order retains its nonempty descriptor world declaration."""

    del written_order
    result = probe(
        getattr(ManagedWorldMatrixValue(), member)._descriptor,
        (
            KernelCall(StaticDependenciesKernel(), None),
            KernelCall(WorldRequirementsKernel(), None),
        ),
    )

    requirements = result.require(WorldRequirementsKernel)
    assert requirements.has_value
    assert requirements.value.roles["main"].resources.cpus.min == 1


def test_world_kernel_snapshot_matches_live_selected_method_collection() -> (
    None
):
    """Bound snapshots keep selected-MRO semantics and exclude siblings."""

    @req(cpus=1, source="base")
    class Base:
        @req(memory="2GiB", source="base-method")
        def run(self) -> None:
            return None

        @req(named={"sibling": 1}, source="sibling")
        def sibling(self) -> None:
            return None

    @req(cpus={"min": 1, "max": 2}, source="leaf")
    class Leaf(Base):
        @req(memory={"min": "1GiB", "max": "3GiB"}, source="leaf-method")
        def run(self) -> None:
            return None

    capture = capture_inspection(Leaf().run)
    bound = WorldRequirementsKernel._from_capture(capture)
    calls = (KernelCall(StaticDependenciesKernel(), None),)
    live = probe(
        Leaf().run, (*calls, KernelCall(WorldRequirementsKernel(), None))
    )
    snapshot = probe(capture.target, (*calls, KernelCall(bound, None)))

    requirement = snapshot.require(WorldRequirementsKernel).value
    assert requirement == live.require(WorldRequirementsKernel).value
    assert requirement.roles["main"].resources.memory.min == 1024**3
    assert requirement.roles["main"].resources.memory.max == 3 * 1024**3
    assert "sibling" not in requirement.roles["main"].resources.named


def test_world_view_rejects_source_ordinals_and_wrong_snapshot() -> None:
    """Owner decoding rejects malformed source links before scheduling."""

    @req(cpus=1)
    def subject() -> None:
        return None

    capture = capture_inspection(subject)
    kernel = WorldRequirementsKernel._from_capture(capture)
    from dryml.worlds.kernel import _view_to_data

    data = _view_to_data(kernel._view)
    data["occurrences"][0]["source"] = 9
    try:
        WorldRequirementsKernel._from_data(capture.target, data)
    except Exception as error:
        assert "occurrence" in str(error)
    else:
        raise AssertionError("malformed source ordinal was accepted")

    oversized = _view_to_data(kernel._view)
    oversized["occurrences"] *= 4097
    try:
        WorldRequirementsKernel._from_data(capture.target, oversized)
    except Exception as error:
        assert "limit" in str(error)
    else:
        raise AssertionError("oversized occurrence table was accepted")

    def other() -> None:
        return None

    other_capture = capture_inspection(other)
    result = probe(
        other_capture.target,
        (
            KernelCall(StaticDependenciesKernel(), None),
            KernelCall(kernel, None),
        ),
    )
    assert result.outcomes[-1].status == "failed"
