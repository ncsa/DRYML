"""Tests for environment collection through the generic code scheduler."""

import functools

import pytest

from dryml.code import (
    KernelCall,
    StaticDependenciesKernel,
    capture_inspection,
    probe,
)
from dryml.environments import EnvironmentRequirementsKernel, req


def test_environment_kernel_collects_live_resolved_targets() -> None:
    """A scheduled domain kernel combines root and helper declarations once."""

    @req(requirements=("root>=1",), source="root")
    def root() -> None:
        helper()

    @req(tags=("helper",), source="helper")
    def helper() -> None:
        return None

    result = probe(
        root,
        (
            KernelCall(StaticDependenciesKernel(), None),
            KernelCall(EnvironmentRequirementsKernel(), None),
        ),
    )

    requirements = result.require(EnvironmentRequirementsKernel)
    assert requirements.has_value
    assert requirements.value.requirements == ("root>=1",)
    assert requirements.value.tags == ("helper",)


def test_environment_snapshot_matches_live_and_codec_round_trip() -> None:
    """Bound views use the same DAG, order, algebra, and owner codecs."""

    @req(requirements=("demo>=1",), source="root")
    def root() -> None:
        helper()

    @req(tags=("helper",), source="helper")
    def helper() -> None:
        return None

    calls = (KernelCall(StaticDependenciesKernel(), None),)
    live = probe(
        root, (*calls, KernelCall(EnvironmentRequirementsKernel(), None))
    )
    capture = capture_inspection(root)
    bound = EnvironmentRequirementsKernel._from_capture(capture)
    snapshot = probe(capture.target, (*calls, KernelCall(bound, None)))

    from dryml.environments.kernel import _view_to_data

    decoded = EnvironmentRequirementsKernel._from_data(
        capture.target, _view_to_data(bound._view)
    )
    transported = probe(capture.target, (*calls, KernelCall(decoded, None)))
    assert snapshot.require(EnvironmentRequirementsKernel) == live.require(
        EnvironmentRequirementsKernel
    )
    assert transported.require(
        EnvironmentRequirementsKernel
    ) == snapshot.require(EnvironmentRequirementsKernel)


def test_environment_snapshot_requires_a_bound_closed_matching_view() -> None:
    """Missing views and another snapshot's view fail through the scheduler."""

    def first() -> None:
        return None

    def second() -> None:
        return None

    first_capture = capture_inspection(first)
    second_capture = capture_inspection(second)
    calls = (KernelCall(StaticDependenciesKernel(), None),)
    missing = probe(
        first_capture.target,
        (*calls, KernelCall(EnvironmentRequirementsKernel(), None)),
    )
    empty = probe(
        first_capture.target,
        (
            *calls,
            KernelCall(
                EnvironmentRequirementsKernel._from_capture(first_capture),
                None,
            ),
        ),
    )
    wrong = probe(
        second_capture.target,
        (
            *calls,
            KernelCall(
                EnvironmentRequirementsKernel._from_capture(first_capture),
                None,
            ),
        ),
    )

    assert missing.outcomes[-1].status == "failed"
    assert wrong.outcomes[-1].status == "failed"
    assert not empty.require(EnvironmentRequirementsKernel).has_value


def test_environment_kernel_keeps_equal_carriers_and_dedupes_wraps() -> None:
    """Occurrence identity preserves equals and removes wraps copies."""

    def decorate(function):
        @functools.wraps(function)
        def wrapper() -> None:
            function()

        return wrapper

    @decorate
    @req(tags=("copied",), source="copied")
    def copied() -> None:
        return None

    @req(tags=("equal",), source="same")
    def first() -> None:
        return None

    @req(tags=("equal",), source="same")
    def second() -> None:
        return None

    def root() -> None:
        copied()
        first()
        second()

    result = probe(
        root,
        (
            KernelCall(StaticDependenciesKernel(), None),
            KernelCall(EnvironmentRequirementsKernel(), None),
        ),
    ).require(EnvironmentRequirementsKernel)

    assert result.value.tags == ("copied", "equal")
    assert result.value.details["sources"] == (
        "1: copied",
        "2: same",
        "3: same",
    )


def test_environment_kernel_conflict_and_skip_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conflicts remain results while failed dependencies skip both forms."""

    @req(python="<3", source="root")
    def root() -> None:
        helper()

    @req(python=">=3", source="helper")
    def helper() -> None:
        return None

    capture = capture_inspection(root)
    bound = EnvironmentRequirementsKernel._from_capture(capture)
    calls = (KernelCall(StaticDependenciesKernel(), None),)
    live = probe(
        root, (*calls, KernelCall(EnvironmentRequirementsKernel(), None))
    )
    snapshot = probe(capture.target, (*calls, KernelCall(bound, None)))
    assert (
        live.require(EnvironmentRequirementsKernel).report.issues[0].path
        == "python"
    )
    assert snapshot.require(EnvironmentRequirementsKernel) == live.require(
        EnvironmentRequirementsKernel
    )

    monkeypatch.setattr(
        StaticDependenciesKernel,
        "run",
        lambda *args: (_ for _ in ()).throw(RuntimeError()),
    )
    live_failed = probe(
        root, (*calls, KernelCall(EnvironmentRequirementsKernel(), None))
    )
    snapshot_failed = probe(capture.target, (*calls, KernelCall(bound, None)))
    assert tuple(outcome.status for outcome in live_failed.outcomes) == (
        "failed",
        "skipped",
    )
    assert tuple(outcome.status for outcome in snapshot_failed.outcomes) == (
        "failed",
        "skipped",
    )


def test_environment_kernel_counts_class_carrier_across_many_methods() -> None:
    """One class annotation remains one occurrence across 65 method owners."""

    from dryml.code.targets import normalize_target
    from dryml.environments.kernel import _declarations_for_targets

    def method(name):
        def implementation(self) -> None:
            return None

        implementation.__name__ = name
        return implementation

    namespace = {
        f"method_{index}": method(f"method_{index}") for index in range(65)
    }
    subject = type("ManyMethods", (), namespace)
    assert req(tags=("class",), source="class")(subject) is subject

    groups = _declarations_for_targets(
        tuple(
            normalize_target(getattr(subject(), f"method_{index}"))
            for index in range(65)
        )
    )
    carrier_ids = [identifier for group in groups for identifier, _ in group]
    assert len(carrier_ids) == 65
    assert len(set(carrier_ids)) == 1
