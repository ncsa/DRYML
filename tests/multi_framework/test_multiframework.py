import pytest
import sys

from dryml.core.backend import discover_backend

jax = pytest.importorskip("jax")
if not hasattr(jax, "devices"):
    sys.modules.pop("jax", None)
    pytest.skip("JAX is not installed.", allow_module_level=True)
jnp = pytest.importorskip("jax.numpy")
tf = pytest.importorskip("tensorflow")
torch = pytest.importorskip("torch")
if not hasattr(torch, "Tensor"):
    sys.modules.pop("torch", None)
    pytest.skip("PyTorch is not installed.", allow_module_level=True)

import dryml.jax
import dryml.torch
import dryml.tf
import numpy as np

from dryml.core.backend import discover_backend

def test_all_backend_detectors():
    assert discover_backend(jnp.array(1)) == "jax"
    assert discover_backend(jnp.float32(1.5)) == "jax"
    assert discover_backend(np.uint8(1)) == "numpy"
    assert discover_backend(np.float64(1.5)) == "numpy"
    assert discover_backend(torch.tensor(1, dtype=torch.uint8)) == "torch"
    assert discover_backend(torch.tensor(1.5, dtype=torch.float64)) == "torch"
    assert discover_backend(tf.constant(1, dtype=tf.uint8)) == "tf"
    assert discover_backend(tf.constant(1.5, dtype=tf.float64)) == "tf"
