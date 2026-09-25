"""U4 public artifact-surface retirement coverage."""

from __future__ import annotations

import inspect

import pytest

import dryml.artifacts as artifacts


pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


def test_scalar_surface_is_retired_and_cached_dataset_is_an_abstract_placeholder():
    """U4 removes scalar artifacts and leaves caching for its later implementation unit."""

    assert not {"Accuracy", "Scalar", "ScalarAgg", "ScalarAvg"} & set(artifacts.__all__)
    assert inspect.isabstract(artifacts.CachedDataset)
    with pytest.raises(TypeError):
        artifacts.CachedDataset()
