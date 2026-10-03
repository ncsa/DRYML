"""U7 completion-receipt characterization for Fold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dryml.artifacts import ArtifactRecoveryError, Fold, Value
from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath
from dryml.managed import ManagedConfig, ManagedRecoveryError, managed_operation

from .test_fold import CountingDataset, _fold


pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")


class NestedResultState(Pickleable):
    """Mutable child used to prove exact recovery validates descendant state."""

    def __init__(self, value):
        self.value = value


class NestedResultValue(Value):
    """Small completed Value with one materialized stateful child."""

    def __init__(self, child):
        self.child = child

    @property
    def ready(self):
        """Return whether this test Value has an installed result."""

        return self._value_is_present()

    @managed_operation(return_state_ref=True)
    def compute(self, *, managed) -> None:
        """Install the child's current value as this Artifact result."""

        self._install_value_payload({
            "format": "dryml.artifacts.value",
            "version": 1,
            "present": True,
            "result": self.child.value,
        })


def test_fold_compute_returns_its_exact_completed_state_ref(tmp_path):
    """The managed completion receipt is the exact saved ready Fold state."""

    store = DirStore(tmp_path / "store")
    fold = _fold(CountingDataset([np.ones((1, 2), dtype=np.float32)]))

    receipt = fold.compute(managed=ManagedConfig(state_repo=store))
    restored = Repo(DirStore(store.base_dir)).load_state_ref(receipt, reuse_live="never")

    assert receipt == fold.last_state_ref
    assert restored.ready
    assert restored.value() == 2.0


def test_fold_completed_recipe_query_and_reopen_recovery_skip_source_iteration(tmp_path, monkeypatch):
    """Completed recipe recovery selects one receipt before loading its Fold state."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    source = CountingDataset([np.ones((1, 2), dtype=np.float32)])
    fold = _fold(source)
    initial = repo.save_object(fold, deep_capture=True)
    receipt = fold.compute(managed=ManagedConfig(state_repo=repo))

    assert Fold.find_completed_state_ref(fold, repo=repo) == receipt
    assert Fold.load_completed(fold, repo=repo) is fold

    CountingDataset.reset()
    reopened = Repo(DirStore.open_existing(store.base_dir))
    monkeypatch.setattr(Fold, "_load_source", lambda *_: pytest.fail("reuse loaded Fold input"))
    recovered = Fold.recover(initial, repo=reopened, reuse_live="never")

    assert recovered.last_state_ref == receipt
    assert recovered.ready
    assert recovered.value() == 2.0
    assert CountingDataset.iterations == 0


def test_fold_recovery_returns_same_initial_operation_after_failure(tmp_path):
    """An initial receipt restores the same managed receiver when no result exists."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    source = CountingDataset([np.ones((1, 2), dtype=np.float32)], fail_after=0)
    fold = _fold(source)
    initial = repo.save_object(fold, deep_capture=True)

    assert Fold.find_completed_state_ref(fold, repo=repo) is None

    with pytest.raises(OSError, match="source failure"):
        fold.compute(managed=ManagedConfig(state_repo=repo))

    reopened = Repo(DirStore.open_existing(store.base_dir))
    recovered = Fold.recover(initial, repo=reopened, reuse_live="never")

    assert recovered.last_state_ref == initial
    assert not recovered.ready
    assert recovered.compute.status(state_repo=reopened).state == "failed"


def test_fold_compute_loads_a_saved_source_fresh_without_reusing_live_cache(tmp_path, monkeypatch):
    """Fold isolation loads an input StateRef without borrowing its live cache entry."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    source = CountingDataset([np.ones((1, 2), dtype=np.float32)])
    source_state = repo.save_object(source, deep_capture=True)
    fold = _fold(source_state)
    calls = []
    original = Repo.load_state_ref

    def tracked(self, state_ref, **kwargs):
        if state_ref == source_state:
            calls.append(kwargs)
        return original(self, state_ref, **kwargs)

    monkeypatch.setattr(Repo, "load_state_ref", tracked)
    fold.compute(managed=ManagedConfig(state_repo=repo))

    assert calls == [{"reuse_live": "never", "cache": "none"}]
    assert source.last_state_ref == source_state


def test_fold_recipe_query_rejects_multiple_completed_states_before_loading_candidates(tmp_path, monkeypatch):
    """Identical recipes with separate completion receipts remain explicitly ambiguous."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = _fold(CountingDataset([np.ones((1, 2), dtype=np.float32)]))
    second = _fold(CountingDataset([np.ones((1, 2), dtype=np.float32)]))

    first.compute(managed=ManagedConfig(state_repo=repo))
    second.compute(managed=ManagedConfig(state_repo=repo))

    monkeypatch.setattr(repo, "load_state_ref", lambda *_args, **_kwargs: pytest.fail("ambiguous query loaded a candidate"))
    with pytest.raises(ArtifactRecoveryError, match="multiple distinct completed"):
        Fold.load_completed(first, repo=repo)


def test_fold_recipe_query_rejects_corrupt_completed_authority(tmp_path):
    """A managed receipt with corrupted result bytes is never reused as complete."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    fold = _fold(CountingDataset([np.ones((1, 2), dtype=np.float32)]))
    receipt = fold.compute(managed=ManagedConfig(state_repo=repo))
    snapshot = Path(store.get_snapshot_directory(receipt))
    payload, = snapshot.glob(f"local-state/*/{receipt.states[GraphPath()]}/data/value.pkl")
    payload.write_bytes(b"corrupt")

    with pytest.raises(ManagedRecoveryError, match="exact StateRef closure"):
        Fold.load_completed(fold, repo=repo)


def test_fold_completed_recovery_rejects_active_live_ownership_for_all_policies(tmp_path, monkeypatch):
    """Every recovery policy rejects duplicate construction across an active owner."""

    repo = Repo(DirStore(tmp_path / "store"))
    fold = _fold(CountingDataset([np.ones((1, 2), dtype=np.float32)]))
    initial = repo.save_object(fold, deep_capture=True)
    fold.compute(managed=ManagedConfig(state_repo=repo))
    monkeypatch.setattr(
        repo, "load_state_ref",
        lambda *_args, **_kwargs: pytest.fail("ownership conflict loaded a duplicate"),
    )

    with repo.reserve_state_graph(fold):
        for policy in ("matching", "greedy", "never"):
            with pytest.raises(ArtifactRecoveryError, match="actively owned"):
                Fold.load_completed(fold, repo=repo, reuse_live=policy)
            with pytest.raises(ArtifactRecoveryError, match="actively owned"):
                Fold.recover(initial, repo=repo, reuse_live=policy)


def test_fold_completed_recovery_rejects_uncached_active_identity(tmp_path):
    """ObjectId ownership remains visible when the live graph is not cached."""

    store = DirStore(tmp_path / "store")
    fold = _fold(CountingDataset([np.ones((1, 2), dtype=np.float32)]))
    receipt = fold.compute(managed=ManagedConfig(state_repo=store))
    reopened = Repo(DirStore.open_existing(store.base_dir))
    uncached = reopened.load_state_ref(receipt, reuse_live="never", cache="none")

    with reopened.reserve_state_graph(uncached):
        for policy in ("matching", "greedy", "never"):
            with pytest.raises(ArtifactRecoveryError, match="actively owned"):
                Fold.load_completed(fold.definition, repo=reopened, reuse_live=policy)


def test_completed_reuse_validates_every_materialized_descendant_state(tmp_path):
    """A stale root receipt cannot hide an independently advanced child state."""

    repo = Repo(DirStore(tmp_path / "store"))
    artifact = NestedResultValue(NestedResultState(1), repo=repo)
    receipt = artifact.compute(managed=ManagedConfig(state_repo=repo))
    artifact.child.value = 2
    repo.save_object(artifact.child)

    recovered = NestedResultValue.load_completed(artifact, repo=repo)

    assert recovered is not artifact
    assert recovered.last_state_ref == receipt
    assert recovered.child.value == 1
    assert recovered.value() == 1
