"""Mandatory smoke coverage for the packaged Ray integration."""

import subprocess
import sys
import textwrap

import pytest


def test_ray_and_dryml_ray_import_with_valid_exports():
    """Verify a fresh process imports DRYML's passive Ray adapter before Ray itself."""
    ray = pytest.importorskip("ray")
    assert "ray" in sys.modules
    program = textwrap.dedent(
        """
        import sys

        import dryml.ray

        assert "ray" not in sys.modules
        import ray
        import dryml.ray.tune

        assert ray is not None
        assert dryml.ray.__all__ == ["tune"]
        assert dryml.ray.tune is not None
        """
    )
    completed = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert ray is not None
