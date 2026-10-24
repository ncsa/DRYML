"""Borrowed managed resources and fresh, hook-free SavedNode preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Definition, Repo, Serializable
from dryml.core.materialization import build_exact_state_load_plan
from dryml.core.node import require_saved
from dryml.core.repo import RepoLoadError
from dryml.core.session import config
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore
from dryml.managed.errors import ManagedStoreError
from dryml.managed.storage import resolve_stores


class ResolutionValue(Serializable):
    """Minimal stateful value used to publish an exact current receipt."""

    constructions = captures = restores = 0

    def __init__(self, value=1):
        type(self).constructions += 1
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Capture one integer in dest_dir with the selected codec; propagate IO errors."""
        type(self).captures += 1
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Restore the integer from src_dir; propagate malformed data and IO errors."""
        type(self).restores += 1
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


def test_explicit_and_single_store_resolution_without_saved_state(tmp_path):
    """Explicit pairs and a single physical session Store resolve before a save."""

    state = DirStore(tmp_path / "state")
    control = DirStore(tmp_path / "control")
    obj = ResolutionValue()
    for selected_control in (control, None):
        resolved = resolve_stores(obj, state_repo=state, control_store=selected_control)
        try:
            assert resolved.state_repo.stores == [state]
            assert resolved.control_store is (state if selected_control is None else control)
        finally:
            resolved.close()
    with config(repo=Repo((state,))):
        resolved = resolve_stores(obj)
    assert resolved.state_repo is not None
    assert resolved.control_store is state
    resolved.close()


def test_same_physical_root_handles_are_one_default_store(tmp_path):
    """Separate handles for one canonical root do not make discovery ambiguous."""

    first = DirStore(tmp_path / "shared")
    second = DirStore(tmp_path / "shared")
    with config(repo=Repo((first, second))):
        resolved = resolve_stores(ResolutionValue())
    assert resolved.state_repo.default_store is first
    assert resolved.control_store is first
    resolved.close()


def test_multistore_uses_the_current_repo_without_last_state_selection(tmp_path):
    """Multi-Store discovery retains routing authority without receipt guessing."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo((first, second))
    obj = ResolutionValue(repo=repo)
    with config(repo=repo):
        resolved = resolve_stores(obj)
        assert resolved.state_repo is repo
        assert resolved.control_store is first
        resolved.close()


def test_missing_session_and_incomplete_selected_state_fail_before_mutation(tmp_path):
    """Absent resources and missing saved facets fail before any state hooks."""

    store = DirStore(tmp_path / "store")
    with pytest.raises(ManagedStoreError, match="explicit_state_repo_required"):
        resolve_stores(ResolutionValue())
    repo = Repo((store,))
    obj = ResolutionValue(repo=repo)
    state = repo.save_object(obj, deep_capture=True)
    counts = ResolutionValue.constructions, ResolutionValue.captures, ResolutionValue.restores
    incomplete = state.with_realization(states={"$": None})
    with pytest.raises(ValueError, match="Saved completeness"):
        build_exact_state_load_plan(repo, incomplete)
    assert (ResolutionValue.constructions, ResolutionValue.captures, ResolutionValue.restores) == counts
    assert Path(store.state_data_path(state.state.key), "value").read_text() == "1"


@pytest.mark.parametrize("fault", ("missing_manifest", "malformed_manifest", "missing_payload", "corrupt_payload"))
def test_saved_preflight_revalidates_current_authority_without_hooks(tmp_path, fault):
    """A prior successful preflight never authorizes missing or corrupt full-key data."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = ResolutionValue(repo=repo)
    state = repo.save_object(obj)
    counts = ResolutionValue.constructions, ResolutionValue.captures, ResolutionValue.restores
    assert require_saved(state) is state
    plan = build_exact_state_load_plan(repo, state)
    assert plan.state_ref is state
    action, = plan.actions
    assert action.state.key == state.state.key and action.store is store
    data = Path(store.state_data_path(state.state.key))
    target = data.parent / "manifest.record" if fault.endswith("manifest") else data / "value"
    if fault.startswith("missing"):
        target.unlink()
    else:
        target.write_bytes(b"malformed selected bytes")
    with pytest.raises(RepoLoadError, match="Selected state preflight is incomplete"):
        build_exact_state_load_plan(repo, state)
    assert (ResolutionValue.constructions, ResolutionValue.captures, ResolutionValue.restores) == counts
    assert obj.value == 1


@pytest.mark.parametrize("kind", ("non_node", "soft", "missing_identity"))
def test_invalid_saved_metadata_fails_before_source_reads(tmp_path, monkeypatch, kind):
    """Saved completeness precedes source selection, construction and restoration."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(ResolutionValue(repo=repo))
    invalid = {"non_node": object(), "soft": Definition(ResolutionValue),
               "missing_identity": state.with_realization(identities={"$": None})}[kind]
    counts = ResolutionValue.constructions, ResolutionValue.captures, ResolutionValue.restores
    monkeypatch.setattr(store, "read_states", lambda *a, **k: pytest.fail("invalid metadata read state"))
    with pytest.raises((TypeError, ValueError), match="Saved completeness"):
        build_exact_state_load_plan(repo, invalid)
    assert (ResolutionValue.constructions, ResolutionValue.captures, ResolutionValue.restores) == counts


def test_explicit_types_and_stale_session_context_are_not_coerced(tmp_path):
    """Invalid resources are not coerced, and expired ambient selection is not reused."""

    store = DirStore(tmp_path / "store")
    with pytest.raises(ManagedStoreError, match="invalid_state_repo"):
        resolve_stores(ResolutionValue(), state_repo=object())
    with pytest.raises(ManagedStoreError, match="invalid_control_store"):
        resolve_stores(ResolutionValue(), state_repo=store, control_store=object())
    with config(repo=Repo((store,))):
        resolved = resolve_stores(ResolutionValue())
        assert resolved.state_repo.default_store is store
        resolved.close()
    with pytest.raises(ManagedStoreError, match="explicit_state_repo_required"):
        resolve_stores(ResolutionValue())


@pytest.mark.parametrize("explicit_control", (False, True))
def test_buffered_control_is_rejected_without_commit_or_borrowed_close(tmp_path, monkeypatch, explicit_control):
    """Neither an implicit nor explicit archive can supply durable managed control."""
    buffered = ZipStore(tmp_path / "buffered.zip")
    calls = []
    try:
        with monkeypatch.context() as patch:
            patch.setattr(buffered, "commit", lambda: calls.append("commit"))
            patch.setattr(buffered, "close", lambda: calls.append("close"))
            with pytest.raises(ManagedStoreError, match="invalid_control_store"):
                resolve_stores(state_repo=buffered, control_store=buffered if explicit_control else None)
            assert calls == []
            assert not Path(buffered.archive_path).exists()
    finally:
        buffered.close()


@pytest.mark.parametrize("shorthand", (False, True))
def test_resource_release_closes_only_owned_wrapper_without_flush(tmp_path, monkeypatch, shorthand):
    """Repeated resolver close borrows caller resources and never commits state."""
    store, control = DirStore(tmp_path / "state"), DirStore(tmp_path / "control")
    repo = Repo(store)
    resolved = resolve_stores(state_repo=store if shorthand else repo, control_store=control)
    selected = resolved.state_repo
    calls = []
    original_close = Repo.close

    def close(wrapper, flush=True):
        calls.append((wrapper, flush))
        return original_close(wrapper, flush=flush)

    with monkeypatch.context() as patch:
        patch.setattr(Repo, "close", close)
        for borrowed in (store, control):
            patch.setattr(borrowed, "close", lambda: pytest.fail("closed borrowed Store"))
            patch.setattr(borrowed, "commit", lambda: pytest.fail("committed borrowed Store"))
        resolved.close()
        resolved.close()
        assert calls == ([(selected, False)] if shorthand else [])
        saved = repo.save_object(ResolutionValue(repo=repo))
        assert require_saved(saved) is saved
    repo.close(flush=False)


def test_failed_resolution_releases_owned_wrapper_without_closing_store(tmp_path, monkeypatch):
    """Invalid control selection releases the temporary wrapper on the error path."""
    store = DirStore(tmp_path / "state")
    calls = []
    original_close = Repo.close

    def close(wrapper, flush=True):
        calls.append((tuple(wrapper.stores), flush))
        return original_close(wrapper, flush=flush)

    with monkeypatch.context() as patch:
        patch.setattr(Repo, "close", close)
        patch.setattr(store, "close", lambda: pytest.fail("closed borrowed Store"))
        patch.setattr(store, "commit", lambda: pytest.fail("committed borrowed Store"))
        with pytest.raises(ManagedStoreError, match="invalid_control_store"):
            resolve_stores(state_repo=store, control_store=object())
        assert calls == [((store,), False)]
    repo = Repo(store)
    assert require_saved(repo.save_object(ResolutionValue(repo=repo))) is not None
    repo.close(flush=False)
