"""Dataset collation for native tensors and explicitly opaque Python leaves."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dryml.core.utils.types import is_namedtuple
from dryml.core.tensor_spec import Layout, TensorSpec, batch_spec_tree, iter_specs


def collate_for_spec(spec, *, ragged=False):
    """Select native tensor collation or spec-directed opaque-object batching.

    Args:
        spec: Declared element specification tree for one Batch source.
        ragged: Explicitly partition unequal dense samples. Ragged source leaves
            always gain an outer partition. NumPy and TensorFlow are supported.

    Returns:
        A selected collator, or ``None`` for the existing generic fallback.
        Python object leaves become lists of the original values without content
        inspection or coercion. Mixed trees retain their declared structure and
        independently select numeric-leaf collators.

    Side Effects:
        Imports only the declared TensorFlow, Torch, or JAX endpoint when selecting
        its stack operation; no optional framework is imported for NumPy or fallback
        selection.
    """

    leaves = tuple(iter_specs(spec))
    if ragged or any(leaf.is_python_object or leaf.layout is Layout.RAGGED for leaf in leaves):
        return _object_tree_collator(spec, ragged=ragged)
    backends = {tensor_spec.backend for tensor_spec in leaves}
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


def _object_tree_collator(spec, *, ragged=False):
    """Compile a tree collator whose opaque leaves never enter array detection."""
    if isinstance(spec, TensorSpec):
        if not spec.is_python_object and (ragged or spec.layout is Layout.RAGGED):
            output = batch_spec_tree(spec, ragged=ragged)
            if output.layout is Layout.RAGGED:
                return _ragged_collator(spec, output)
        return list if spec.is_python_object else collate_for_spec(spec) or default_collate
    children = ({key: _object_tree_collator(child, ragged=ragged) for key, child in spec.items()}
                if isinstance(spec, dict) else tuple(_object_tree_collator(child, ragged=ragged) for child in spec))

    def collate(items):
        if not items:
            raise ValueError("Cannot collate an empty item list.")
        if isinstance(spec, dict):
            if any(not isinstance(item, dict) or tuple(item) != tuple(spec) for item in items):
                raise TypeError("Batch values must match their declared mapping structure.")
            return {key: child([item[key] for item in items]) for key, child in children.items()}
        kind = tuple if isinstance(spec, tuple) else list
        if any(not isinstance(item, kind) or len(item) != len(spec) for item in items):
            raise TypeError("Batch values must match their declared sequence structure.")
        if is_namedtuple(items[0]):
            if any(type(item) is not type(items[0]) for item in items):
                raise TypeError("Namedtuple Batch values must have identical types.")
            return type(items[0])(*(child([item[index] for item in items]) for index, child in enumerate(children)))
        return kind(child([item[index] for item in items]) for index, child in enumerate(children))

    return collate


def _ragged_collator(spec, output_spec):
    """Validate declared sample facts before packing native row partitions."""
    from dryml.methods.signature import runtime_node_for_constraint, satisfies, spec_node

    constraint = spec_node(spec)
    if spec.backend is None:
        raise TypeError("Ragged collation requires a declared backend.")
    spec.backend.module
    if spec.backend.value == "numpy":
        from dryml.numpy.ragged import RaggedTensor
        def pack(items):
            dtype = (items[0].row_splits_dtype if isinstance(items[0], RaggedTensor) else "int64")
            if output_spec.row_splits_dtype is not None:
                dtype = output_spec.row_splits_dtype.np()
            if any(isinstance(item, RaggedTensor) and item.row_splits_dtype != np.dtype(dtype) for item in items):
                raise TypeError("Ragged Batch input partition dtypes must agree.")
            return RaggedTensor.stack(items, ragged_rank=output_spec.ragged_rank,
                                      row_splits_dtype=dtype)
    elif spec.backend.value == "tf":
        import tensorflow as tf
        def pack(items):
            dtype = (items[0].row_splits.dtype if isinstance(items[0], tf.RaggedTensor) else tf.int64)
            if output_spec.row_splits_dtype is not None:
                dtype = output_spec.row_splits_dtype.tf()
            if any(isinstance(item, tf.RaggedTensor) and item.row_splits.dtype != dtype for item in items):
                raise TypeError("Ragged Batch input partition dtypes must agree.")
            rank = output_spec.ragged_rank - 1
            parts = [item if isinstance(item, tf.RaggedTensor) or rank == 0
                     else tf.RaggedTensor.from_tensor(item, ragged_rank=rank, row_splits_dtype=dtype)
                     for item in items]
            lengths = tf.stack([tf.cast(item.nrows() if isinstance(item, tf.RaggedTensor)
                                       else tf.shape(item)[0], dtype) for item in parts])
            values = tf.concat(parts, axis=0)
            if spec.shape is not None and type(spec.shape[0]) is int:
                return tf.RaggedTensor.from_uniform_row_length(
                    values, tf.constant(spec.shape[0], dtype), nrows=tf.constant(len(items), dtype), validate=True)
            return tf.RaggedTensor.from_row_lengths(values, lengths, validate=True)
    else:
        raise TypeError("Native ragged collation supports only NumPy and TensorFlow.")

    def collate(items):
        if not all(satisfies(constraint, runtime_node_for_constraint(item, constraint)) for item in items):
            raise ValueError("Ragged Batch inputs do not satisfy their declared sample spec.")
        return pack(items)
    return collate


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
