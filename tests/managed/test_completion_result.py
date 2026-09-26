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
