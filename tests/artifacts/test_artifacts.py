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
