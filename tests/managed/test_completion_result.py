"""Managed invocation completion-result contract tests."""

from __future__ import annotations

import inspect
from typing import Any
import warnings

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Repo, StateRef
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedDeclarationError, managed_operation


_completion_handoffs = []


class CompletionValue(Pickleable):
    """Small managed receiver whose body can deliberately return an unwanted value."""

    def __init__(self, value=0, unexpected=False):
        self.value = value
        self.unexpected = unexpected

    @managed_operation(return_state_ref=True)
    def advance(self, amount, *, managed) -> None:
        """Change state and optionally exercise the discarded-body-result boundary."""

        self.value += amount
        if self.unexpected:
            return _UnexpectedResult()


class CompletionHookValue(Pickleable):
    """Managed receiver that records the private completed-publication handoff."""

    def __init__(self, *, fail_hook=False):
        """Initialize mutation state and optional handoff delivery failure."""

        self.value = 0
        self.fail_hook = fail_hook

    @managed_operation(return_state_ref=True)
    def advance(self, *, managed) -> None:
        """Complete one normal invocation for lifecycle-hook assertions."""

        self.value += 1

    @managed_operation(return_state_ref=True)
    def fail(self, *, managed) -> None:
        """Raise from the body before a final state can be published."""

        raise ValueError("body failed")

    @managed_operation(return_state_ref=True)
    def interrupt(self, *, managed) -> None:
        """Request interruption before final publication and completion."""

        managed.interrupt()

    def _managed_post_publication(self, final_state, state_repo, report) -> None:
        """Record the invocation-local completed state and selected publication authority.

        Args:
            final_state: The exact validated final StateRef for this invocation.
            state_repo: The borrowed selected Repo that published ``final_state``.
            report: The immutable selected save report proving snapshot placement.

        Raises:
            RuntimeError: When this fixture is configured to model delivery failure.
        """

        status = self.advance.status(state_repo=state_repo)
        _completion_handoffs.append((final_state, state_repo, report, status))
        if self.fail_hook:
            raise RuntimeError("post-completion handoff failed")


class _UnexpectedResult:
    """A body result whose representation must never be requested."""

    def __repr__(self) -> str:
        raise AssertionError("discarded managed results must not be represented")


def test_return_state_ref_requires_explicit_none_annotation_and_projects_signature():
    """Only an authored None result contract can select managed completion results."""

    def absent(self, *, managed):
        return None

    def incompatible(self, *, managed) -> int:
        return None

    def ambiguous(self, *, managed) -> Any:
        return None

    def postponed(self, *, managed) -> "None":
        return None

    with pytest.raises(ManagedDeclarationError, match="explicit None"):
        managed_operation(return_state_ref=True)(absent)
    with pytest.raises(ManagedDeclarationError, match="explicit None"):
        managed_operation(return_state_ref=True)(incompatible)
    with pytest.raises(ManagedDeclarationError, match="explicit None"):
        managed_operation(return_state_ref=True)(ambiguous)
    declaration = managed_operation(return_state_ref=True)(postponed)

    class Subject:
        operation = declaration

    assert declaration.author_signature.return_annotation == postponed.__annotations__["return"]
    assert inspect.signature(Subject().operation).return_annotation is StateRef


def test_return_state_ref_captures_the_completed_publication(tmp_path):
    """A flagged call returns its own final StateRef after completion association."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = CompletionValue(repo=repo)

    result = value.advance(3, managed=ManagedConfig(state_repo=repo))

    assert type(result) is StateRef
    assert result == value.advance.status(state_repo=repo).final_state_ref
    assert repo.load_state_ref(result, reuse_live="never").value == 3


def test_receiver_without_private_post_publication_hook_retains_completion_behavior(tmp_path):
    """Ordinary managed receivers need no lifecycle-hook declaration to complete."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = CompletionValue(repo=repo)

    result = value.advance(3, managed=ManagedConfig(state_repo=repo))

    assert type(result) is StateRef
    assert (value.value, value.advance.status(state_repo=repo).final_state_ref) == (3, result)


def test_return_state_ref_discards_unexpected_result_before_publication_when_warning_errors(tmp_path):
    """Warnings-as-errors stop a flagged result before a final success is published."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = CompletionValue(unexpected=True, repo=repo)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="discarded"):
            value.advance(3, managed=ManagedConfig(state_repo=repo))

    assert value.advance.status(state_repo=repo).state == "failed"


def test_return_state_ref_warns_once_and_never_materializes_the_body_result(tmp_path):
    """A normal warning policy discards a non-None body value and still completes."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = CompletionValue(unexpected=True, repo=repo)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        result = value.advance(3, managed=ManagedConfig(state_repo=repo))

    assert type(result) is StateRef
    assert [warning.category for warning in captured] == [RuntimeWarning]


def test_private_post_publication_hook_receives_completed_invocation_authority(tmp_path):
    """Each private handoff retains its exact ref after a later receipt replaces it."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = CompletionHookValue(repo=repo)
    _completion_handoffs.clear()

    first = value.advance(managed=ManagedConfig(state_repo=repo))
    second = value.advance(managed=ManagedConfig(state_repo=repo, rerun=True))

    assert len(_completion_handoffs) == 2
    final_state, selected_repo, report, status = _completion_handoffs[0]
    assert (final_state, _completion_handoffs[1][0]) == (first, second)
    assert first != second
    assert selected_repo is repo
    assert (status.state, status.final_state_ref) == ("completed", first)
    assert value.last_state_ref == second
    assert any(snapshot.state_ref == first for snapshot in report.snapshots)
    assert repo.load_state_ref(first, reuse_live="never").value == 1
    assert repo.load_state_ref(second, reuse_live="never").value == 2


def test_private_post_publication_hook_failure_preserves_completed_authority(tmp_path):
    """A failed post-completion delivery raises without recasting the operation."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = CompletionHookValue(fail_hook=True, repo=repo)
    _completion_handoffs.clear()

    with pytest.raises(RuntimeError, match="post-completion handoff failed"):
        value.advance(managed=ManagedConfig(state_repo=repo))

    final_state, _, _, status_at_handoff = _completion_handoffs[0]
    status = value.advance.status(state_repo=repo)
    assert (status_at_handoff.state, status_at_handoff.final_state_ref) == ("completed", final_state)
    assert (status.state, status.final_state_ref, status.failure_code) == ("completed", final_state, None)
