from __future__ import annotations

from typing import Any

import numpy as np

from dryml.core.utils.types import is_namedtuple
from dryml.core.tensor_spec import iter_specs


def split_for_spec(spec):
    """Return one native dense splitter for a uniform declared backend.

    The returned splitter preserves the existing tree order and cardinality. It
    falls back to :func:`default_split` for unknown, mixed, or unsupported specs.
    """

    backends = {tensor_spec.backend for tensor_spec in iter_specs(spec)}
    if len(backends) != 1:
        return None
    backend = next(iter(backends))
    if backend is None or backend.value == "numpy":
        return default_split
    if backend.value == "torch":
        import torch

        return lambda batch: _native_split(batch, torch.unbind)
    if backend.value == "tf":
        import tensorflow as tf

        return lambda batch: _native_split(batch, tf.unstack)
    if backend.value == "jax":
        import jax.numpy as jnp

        return lambda batch: _native_split(batch, jnp.unstack)
    return None


def _native_split(batch, unstack):
    """Split selected backend tensor leaves and rebuild the existing tree shape."""

    if isinstance(batch, dict):
        parts = {key: _native_split(value, unstack) for key, value in batch.items()}
        length = len(next(iter(parts.values()))) if parts else 0
        if any(len(value) != length for value in parts.values()):
            raise TypeError("All dict fields must have the same batch length for splitting.")
        return [{key: parts[key][index] for key in parts} for index in range(length)]
    if isinstance(batch, tuple):
        parts = [_native_split(value, unstack) for value in batch]
        length = len(parts[0]) if parts else 0
        if any(len(value) != length for value in parts):
            raise TypeError("All tuple fields must have the same batch length for splitting.")
        return [tuple(value[index] for value in parts) for index in range(length)]
    if isinstance(batch, list):
        parts = [_native_split(value, unstack) for value in batch]
        length = len(parts[0]) if parts else 0
        if any(len(value) != length for value in parts):
            raise TypeError("All list fields must have the same batch length for splitting.")
        return [[value[index] for value in parts] for index in range(length)]
    return list(unstack(batch, dim=0) if getattr(unstack, "__module__", "").startswith("torch") else unstack(batch, axis=0))


def _leaf_batch_len(x: Any) -> int:
    if hasattr(x, "shape"):
        shape = tuple(x.shape)
        if len(shape) == 0:
            raise ValueError("Cannot split a rank-0 leaf as a batch.")
        return int(shape[0])

    if isinstance(x, (list, tuple)):
        return len(x)

    raise TypeError(f"Cannot determine batch length for {type(x).__name__}.")


def _leaf_index(x: Any, i: int) -> Any:
    return x[i]


def default_split(batch: Any) -> list[Any]:
    if isinstance(batch, dict):
        if len(batch) == 0:
            return []
        split_fields = {k: default_split(v) for k, v in batch.items()}
        n = len(next(iter(split_fields.values())))
        if any(len(v) != n for v in split_fields.values()):
            raise TypeError("All dict fields must have the same batch length for splitting.")
        return [
            {k: split_fields[k][i] for k in split_fields}
            for i in range(n)
        ]

    if is_namedtuple(batch):
        split_parts = [default_split(v) for v in batch]
        n = len(split_parts[0]) if split_parts else 0
        if any(len(v) != n for v in split_parts):
            raise TypeError("All namedtuple fields must have the same batch length for splitting.")
        return [
            type(batch)(*(split_parts[j][i] for j in range(len(split_parts))))
            for i in range(n)
        ]

    if isinstance(batch, tuple):
        split_parts = [default_split(v) for v in batch]
        n = len(split_parts[0]) if split_parts else 0
        if any(len(v) != n for v in split_parts):
            raise TypeError("All tuple fields must have the same batch length for splitting.")
        return [
            tuple(split_parts[j][i] for j in range(len(split_parts)))
            for i in range(n)
        ]

    if isinstance(batch, list):
        split_parts = [default_split(v) for v in batch]
        n = len(split_parts[0]) if split_parts else 0
        if any(len(v) != n for v in split_parts):
            raise TypeError("All list fields must have the same batch length for splitting.")
        return [
            [split_parts[j][i] for j in range(len(split_parts))]
            for i in range(n)
        ]

    n = _leaf_batch_len(batch)
    return [_leaf_index(batch, i) for i in range(n)]
