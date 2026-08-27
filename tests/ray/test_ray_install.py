"""Mandatory smoke coverage for the packaged Ray integration."""

import sys

import pytest


def test_ray_and_dryml_ray_import_with_valid_exports():
    """Import installed Ray and verify DRYML's adapter surface stays passive."""

    import dryml.ray

    assert "ray" not in sys.modules

    ray = pytest.importorskip("ray")
    import dryml.ray.tune

    assert ray is not None
    assert dryml.ray.__all__ == ["tune"]
    assert dryml.ray.tune is not None
