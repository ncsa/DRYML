"""Stable source types and values used by CachedDataset compatibility fixtures."""

from __future__ import annotations

import numpy as np

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Dataset


REJECT_CONSTRUCTION = False
CONSTRUCTION_COUNT = 0


class FixtureDataset(Dataset):
    """Provide deterministic heterogeneous yields for compatibility generation.

    Args:
        kind: One of ``"fidelity"``, ``"empty"``, or ``"polynomial"``.

    The rejection switch lets reader tests prove that restoring a completed cache
    does not materialize its retained source definition.
    """

    def __init__(self, kind: str) -> None:
        """Retain detached fixture values unless reader tests forbid construction."""

        global CONSTRUCTION_COUNT
        if REJECT_CONSTRUCTION:
            raise AssertionError("completed fixture restore materialized its source")
        CONSTRUCTION_COUNT += 1
        if kind == "fidelity":
            values, spec = fidelity_values(), fidelity_spec()
        elif kind == "empty":
            values, spec = (), fidelity_spec()
        elif kind == "polynomial":
            values, spec = polynomial_values(), polynomial_spec()
        else:
            raise ValueError("unknown CachedDataset fixture source kind")
        self.kind = kind
        self.values = tuple(values)
        super().__init__(spec)

    def __iter__(self):
        """Return a fresh iterator over the retained deterministic values."""

        return iter(self.values)

    def __len__(self) -> Cardinality:
        """Return the exact finite fixture cardinality."""

        return Cardinality.finite(len(self.values))


def fidelity_spec():
    """Return the nested dynamic SpecTree shared by all codec fixtures."""

    return {
        "bool": TensorSpec("bool", shape=(Dynamic,), backend="numpy"),
        2: (
            TensorSpec("int64", shape=(), backend="numpy"),
            TensorSpec("uint64", shape=(Dynamic,), backend="numpy"),
            TensorSpec("float64", shape=(Dynamic,), backend="numpy"),
        ),
        False: [
            TensorSpec("bfloat16", shape=(Dynamic,), backend="numpy"),
            TensorSpec("complex128", shape=(Dynamic,), backend="numpy"),
            TensorSpec("string", shape=(Dynamic,), backend="numpy"),
        ],
    }


def fidelity_values():
    """Return deterministic values covering the representative dense dtype surface."""

    import ml_dtypes

    return (
        {
            "bool": np.asarray([True, False], dtype=np.bool_),
            2: (
                np.asarray(-7, dtype=np.int64),
                np.asarray([0, np.iinfo(np.uint64).max], dtype=np.uint64),
                np.asarray([np.nan, np.inf, -0.0], dtype=np.float64),
            ),
            False: [
                np.asarray([1.5], dtype=ml_dtypes.bfloat16),
                np.asarray([1 + 2j, -3 + 0.5j], dtype=np.complex128),
                np.asarray(["", "e\u0301", "a\x00b"], dtype=np.str_),
            ],
        },
        {
            "bool": np.asarray([], dtype=np.bool_),
            2: (
                np.asarray(9, dtype=np.int64),
                np.asarray([3], dtype=np.uint64),
                np.asarray([], dtype=np.float64),
            ),
            False: [
                np.asarray([], dtype=ml_dtypes.bfloat16),
                np.asarray([], dtype=np.complex128),
                np.asarray(["caf\u00e9"], dtype=np.str_),
            ],
        },
    )


def polynomial_spec():
    """Return the element spec for the deterministic polynomial-style sample set."""

    return {
        "x": TensorSpec("float64", shape=(2,), backend="numpy"),
        "y": TensorSpec("float64", shape=(), backend="numpy"),
    }


def polynomial_values():
    """Return fixed quadratic samples used for NumPy/Parquet equivalence."""

    points = (-2.0, -0.5, 0.0, 1.5, 3.0)
    return tuple(
        {
            "x": np.asarray([value, value * value], dtype=np.float64),
            "y": np.asarray(1.25 - 0.5 * value + 2.0 * value * value, dtype=np.float64),
        }
        for value in points
    )


__all__ = [
    "CONSTRUCTION_COUNT",
    "FixtureDataset",
    "REJECT_CONSTRUCTION",
    "fidelity_spec",
    "fidelity_values",
    "polynomial_spec",
    "polynomial_values",
]
