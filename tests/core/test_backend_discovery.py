import sys
import types

import numpy as np

from dryml.core.backend import Backend, discover_backends
from dryml.jax.backend import is_jax_available
from dryml.tf.backend import is_tf_available
from dryml.torch.backend import is_torch_available


def test_discover_backends_for_single_numpy_arg():
    x = np.zeros((2, 3), dtype=np.float32)

    assert discover_backends(x) == {Backend.numpy}


def test_discover_backends_for_nested_numpy_args():
    x = {"a": np.zeros((2,), dtype=np.float32), "b": (np.ones((3,), dtype=np.float32),)}

    assert discover_backends(x) == {Backend.numpy}


def test_discover_backends_ignores_non_backend_metadata():
    x = {"a": np.zeros((2,), dtype=np.float32), "label": 1}

    assert discover_backends(x) == {Backend.numpy}


def test_backend_availability_ignores_spec_less_framework_modules(monkeypatch):
    for name in ("jax", "torch", "tensorflow"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    assert not is_jax_available()
    assert not is_tf_available()
    assert not is_torch_available()
