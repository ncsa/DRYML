from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dryml.core.utils.types import is_namedtuple
from dryml.core.tensor_spec import iter_specs


def collate_for_spec(spec):
    """Return one native dense collator for a uniform declared backend.

    Args:
        spec: Declared element specification tree for one Batch source.

    Returns:
        A selected collator, or ``None`` when the generic Python fallback is the
        only supported path.

    Side Effects:
        Imports only the declared TensorFlow, Torch, or JAX endpoint when selecting
        its stack operation; no optional framework is imported for NumPy or fallback
        selection.
    """

    backends = {tensor_spec.backend for tensor_spec in iter_specs(spec)}
    if len(backends) != 1:
        return None
    backend = next(iter(backends))
    if backend is None:
        return None
    if backend.value == "numpy":
        return lambda items: _native_collate(items, np.stack, axis=0)
    if backend.value == "torch":
        import torch

        return lambda items: _native_collate(items, torch.stack, axis=0)
    if backend.value == "tf":
        import tensorflow as tf

        return lambda items: _native_collate(items, tf.stack, axis=0)
    if backend.value == "jax":
        import jax.numpy as jnp

        return lambda items: _native_collate(items, jnp.stack, axis=0)
    return None


def _native_collate(items, stack, *, axis):
    """Apply one selected backend stack leafwise while retaining tree validation."""

    first = items[0]
    if isinstance(first, dict):
        if any(not isinstance(item, dict) or item.keys() != first.keys() for item in items[1:]):
            raise TypeError("All dict items must have identical keys for collation.")
        return {key: _native_collate([item[key] for item in items], stack, axis=axis) for key in first}
    if is_namedtuple(first):
        if any(not is_namedtuple(item) or type(item) is not type(first) for item in items[1:]):
            raise TypeError("All namedtuple items must have identical types for collation.")
        return type(first)(*(_native_collate([item[index] for item in items], stack, axis=axis) for index in range(len(first))))
    if isinstance(first, tuple):
        if any(not isinstance(item, tuple) or len(item) != len(first) for item in items[1:]):
            raise TypeError("All tuple items must have identical lengths for collation.")
        return tuple(_native_collate([item[index] for item in items], stack, axis=axis) for index in range(len(first)))
    if isinstance(first, list):
        if any(not isinstance(item, list) or len(item) != len(first) for item in items[1:]):
            raise TypeError("All list items must have identical lengths for collation.")
        return [_native_collate([item[index] for item in items], stack, axis=axis) for index in range(len(first))]
    return stack(items, dim=axis) if getattr(stack, "__module__", "").startswith("torch") else stack(items, axis=axis)


def default_collate(items: list[Any]) -> Any:
    if len(items) == 0:
        raise ValueError("Cannot collate an empty item list.")

    first = items[0]

    if isinstance(first, dict):
        keys = first.keys()
        for item in items[1:]:
            if not isinstance(item, dict) or item.keys() != keys:
                raise TypeError("All dict items must have identical keys for collation.")
        return {
            k: default_collate([item[k] for item in items])
            for k in keys
        }

    if is_namedtuple(first):
        return type(first)(*[
            default_collate([item[i] for item in items])
            for i in range(len(first))
        ])

    if isinstance(first, tuple):
        for item in items[1:]:
            if not isinstance(item, tuple) or len(item) != len(first):
                raise TypeError("All tuple items must have identical lengths for collation.")
        return tuple(
            default_collate([item[i] for item in items])
            for i in range(len(first))
        )

    if isinstance(first, list):
        for item in items[1:]:
            if not isinstance(item, list) or len(item) != len(first):
                raise TypeError("All list items must have identical lengths for collation.")
        return [
            default_collate([item[i] for item in items])
            for i in range(len(first))
        ]

    # torch tensor
    try:
        import torch
        if isinstance(first, torch.Tensor):
            return torch.stack(items, dim=0)
    except Exception:
        pass

    # tf tensor
    try:
        import tensorflow as tf
        if tf.is_tensor(first):
            return tf.stack(items, axis=0)
    except Exception:
        pass

    # numpy or scalar fallback
    try:
        return np.stack(items, axis=0)
    except Exception as e:
        raise TypeError(
            f"Don't know how to collate leaf type {type(first).__name__}."
        ) from e
