import numpy as np

from dryml.core2.backend import Backend, discover_backends


def test_discover_backends_for_single_numpy_arg():
    x = np.zeros((2, 3), dtype=np.float32)

    assert discover_backends(x) == {Backend.numpy}


def test_discover_backends_for_nested_numpy_args():
    x = {"a": np.zeros((2,), dtype=np.float32), "b": (np.ones((3,), dtype=np.float32),)}

    assert discover_backends(x) == {Backend.numpy}


def test_discover_backends_ignores_non_backend_metadata():
    x = {"a": np.zeros((2,), dtype=np.float32), "label": 1}

    assert discover_backends(x) == {Backend.numpy}
