"""Keep experimental JAX dependencies outside ordinary model imports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_generic_and_jax_facade_imports_defer_training_dependencies() -> None:
    """Import public facades without loading JAX, Flax, or Optax."""

    environment = dict(os.environ)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        source_path + os.pathsep + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
import dryml
import dryml.jax
import dryml.models
print(json.dumps(sorted(
    name for name in ('jax', 'jaxlib', 'flax', 'optax') if name in sys.modules
)))
""",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_jax_model_facade_import_defers_native_frameworks() -> None:
    """Importing the experimental model API alone does not import JAX/Flax/Optax."""

    environment = dict(os.environ)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; import dryml.models.jax; print(json.dumps(sorted("
            "name for name in ('jax', 'jaxlib', 'flax', 'optax') if name in sys.modules)))",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_selected_jax_training_stack_exposes_nnx_and_optax() -> None:
    """Require the qualification environment to expose the selected JAX stack.

    The ordinary test environment intentionally need not install experimental
    training dependencies. The dedicated Python 3.12 qualification job installs
    them and therefore turns this skip into an import and CPU-runtime check.
    """

    jax = pytest.importorskip("jax")
    nnx = pytest.importorskip("flax.nnx")
    optax = pytest.importorskip("optax")

    assert jax.default_backend() == "cpu"
    assert all(device.platform == "cpu" for device in jax.devices())
    assert isinstance(nnx.Linear(1, 1, rngs=nnx.Rngs(0)), nnx.Module)
    assert optax.adam(1e-3) is not None
