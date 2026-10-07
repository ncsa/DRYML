"""Adapter-focused JAX training-input tests; JAX training itself is separate work."""

from __future__ import annotations

import numpy as np
import pytest

from dryml.data import ArrayDataset, Batch
from dryml.methods import Method
from dryml.models.utils import TrainingPreparation


class PreparationOwner(Method):
    """Minimal Method owner for inspecting the JAX handoff preparation path."""

    def __call__(self, value):
        """Return the unused test value unchanged."""

        return value


def test_jax_native_iterator_yields_prepared_authored_batches():
    """The JAX bridge yields native batches and closes an exhausted traversal."""
    jax = pytest.importorskip("jax")
    import dryml.jax
    from dryml.data.native import PreparedDataset
    from dryml.jax.training_data import iter_training_batches

    data = Batch(ArrayDataset((
        np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32),
    )), 2)
    preparation = TrainingPreparation.from_specs(
        PreparationOwner(), data.spec[0], data.spec[1], "jax"
    )

    values = list(iter_training_batches(PreparedDataset(data), preparation))

    assert [tuple(x.shape) for x, _ in values] == [(2, 1), (1, 1)]
    assert all(isinstance(value, jax.Array) for pair in values for value in pair)
    assert [np.asarray(y)[:, 0].tolist() for _, y in values] == [[0.0, 2.0], [4.0]]
