"""Public artifact-surface coverage."""

from __future__ import annotations

import pytest

import dryml.artifacts as artifacts
from dryml.artifacts import Artifact, CachedDataset
from dryml.data import Dataset


pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


def test_scalar_surface_is_retired_and_cached_dataset_is_concrete():
    """The concrete cache remains the public Artifact and Dataset implementation."""

    assert not {"Accuracy", "Scalar", "ScalarAgg", "ScalarAvg"} & set(artifacts.__all__)
    assert issubclass(CachedDataset, Artifact)
    assert issubclass(CachedDataset, Dataset)
    with pytest.raises(TypeError):
        CachedDataset()


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_completed_artifact_history_and_replay_survive_explicit_state_cleanup(tmp_path, monkeypatch):
    """Completion metadata is historical; loading deleted bytes never recomputes."""
    from dryml.core import Repo
    from dryml.core.repo import RepoLoadError
    from dryml.core.store.dir import DirStore
    from dryml.managed import ManagedConfig
    from .test_fold_resume import _fold, ResumeDataset

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    fold = _fold()
    result = fold.compute(managed=ManagedConfig(state_repo=repo))
    calls = ResumeDataset.iterations
    operation = fold.compute
    from dryml.managed.control import ManagedControlStore
    ManagedControlStore(repo.default_store, repo).cleanup_checkpoints(result.execution.operation_id)
    assert operation.status(state_repo=repo).state == "completed"
    assert artifacts.Fold.find_completed_state_ref(fold, repo=repo).graph_equal(result.final_state_ref)
    monkeypatch.setattr(artifacts.Fold, "_load_source", lambda *_: pytest.fail("history recomputed"))
    replay = operation(managed=ManagedConfig(state_repo=repo))
    assert replay.value is None
    assert replay.execution.run_id == result.execution.run_id
    assert replay.execution.invocation_id != result.execution.invocation_id
    assert replay.final_state_ref.graph_equal(result.final_state_ref)
    assert ResumeDataset.iterations == calls
    with pytest.raises(RepoLoadError):
        artifacts.Fold.load_completed(fold, repo=repo, reuse_live="never")
