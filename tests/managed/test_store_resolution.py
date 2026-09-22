"""Focused U4 Store-selection and authority-only StateRef validation tests."""

from __future__ import annotations

import pytest

from dryml.core import Repo, Serializable
from dryml.core.session import config
from dryml.core.store.dir import DirStore
from dryml.managed.errors import ManagedStoreError
from dryml.managed.storage import resolve_stores, validate_state_ref


class ResolutionValue(Serializable):
    """Minimal stateful value used to publish an exact current receipt."""

    def __init__(self, value=1):
        self.value = value


def test_explicit_and_single_store_resolution_without_saved_state(tmp_path):
    """Explicit pairs and a single physical session Store resolve before a save."""

    state = DirStore(tmp_path / "state")
    control = DirStore(tmp_path / "control")
    obj = ResolutionValue()
    assert resolve_stores(obj, state_repo=state, control_store=control).control_store is control
    assert resolve_stores(obj, state_repo=state).control_store is state
    with config(repo=Repo((state,))):
        resolved = resolve_stores(obj)
    assert resolved.state_repo is not None
    assert resolved.control_store is state


def test_same_physical_root_handles_are_one_default_store(tmp_path):
    """Separate handles for one canonical root do not make discovery ambiguous."""

    first = DirStore(tmp_path / "shared")
    second = DirStore(tmp_path / "shared")
    with config(repo=Repo((first, second))):
        resolved = resolve_stores(ResolutionValue())
    assert resolved.state_repo.default_store is first


def test_multistore_uses_the_current_repo_without_last_state_selection(tmp_path):
    """Multi-Store discovery retains routing authority without receipt guessing."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo((first, second))
    obj = ResolutionValue(repo=repo)
    with config(repo=repo):
        assert resolve_stores(obj).state_repo is repo


def test_missing_session_and_incomplete_selected_state_fail_before_mutation(tmp_path):
    """No session and incomplete exact StateRef authority require explicit recovery."""

    store = DirStore(tmp_path / "store")
    with pytest.raises(ManagedStoreError, match="no current Repo"):
        resolve_stores(ResolutionValue())
    repo = Repo((store,))
    obj = ResolutionValue(repo=repo)
    state = repo.save_object(obj, deep_capture=True)
    state_path = store.get_snapshot_directory(state) / "state-ref.record"
    # Removing immutable test authority makes the retained live receipt unusable;
    # validation must not silently select an arbitrary Store or materialize it.
    __import__("os").unlink(state_path)
    with pytest.raises(Exception):
        validate_state_ref(repo, state)


def test_explicit_types_and_stale_session_context_are_not_coerced(tmp_path):
    """Resolver snapshots only exact DirStore inputs and current context authority."""

    store = DirStore(tmp_path / "store")
    with pytest.raises(ManagedStoreError):
        resolve_stores(ResolutionValue(), state_repo=object())
    with config(repo=Repo((store,))):
        assert resolve_stores(ResolutionValue()).state_repo.default_store is store
    with pytest.raises(ManagedStoreError):
        resolve_stores(ResolutionValue())
