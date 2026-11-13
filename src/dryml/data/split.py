"""Dataset splitting that respects tensor axes and opaque object batch boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from dryml.core.utils.types import is_namedtuple
from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec, iter_specs, _python_object_batch_size


def split_for_spec(spec):
    """Select native splitting or spec-directed opaque-object unbatching.

    The returned splitter preserves the existing tree order and cardinality. It
    falls back to :func:`default_split` for unknown, mixed, or unsupported specs.
    Explicitly batched Python object leaves split only their outer list/tuple;
    their contents and classes remain uninspected. Mixed object/tensor trees are
    split by declaration. NumPy and TensorFlow ragged values split their outer row
    partition, preserving empty rows and deeper partitions. Unbatched object or
    ragged leaves raise ValueError; unsupported ragged backends raise TypeError.
    Runtime ragged values are validated against the retained leaf spec.
    """

    leaves = tuple(iter_specs(spec))
    if any(leaf.is_python_object or leaf.layout is Layout.RAGGED for leaf in leaves):
        return _object_tree_splitter(spec)
    backends = {tensor_spec.backend for tensor_spec in leaves}
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


def _object_tree_splitter(spec):
    """Compile tree splitting while preserving each opaque example by identity."""
    if isinstance(spec, TensorSpec):
        if spec.layout is Layout.RAGGED:
            if not spec.batched:
                raise ValueError("Unbatch requires explicitly batched ragged specs.")
            if spec.backend is None or spec.backend.value not in {"numpy", "tf"}:
                raise TypeError("Native ragged splitting supports only NumPy and TensorFlow.")
            spec.backend.module
            def split_ragged(batch):
                from dryml.methods.signature import runtime_node_for_constraint, satisfies, spec_node
                constraint = spec_node(spec)
                if not satisfies(constraint, runtime_node_for_constraint(batch, constraint)):
                    raise ValueError("Ragged Unbatch input differs from its declared spec.")
                size = len(batch) if spec.backend.value == "numpy" else int(batch.nrows())
                return [batch[i] for i in range(size)]
            return split_ragged
        if not spec.is_python_object:
            return split_for_spec(spec) or default_split
        if not spec.batched:
            raise ValueError("Unbatch requires explicitly batched Python object specs.")
        def split_objects(batch):
            size = _python_object_batch_size(batch)
            if spec.batch is not Dynamic and size != spec.batch:
                raise ValueError("Opaque Python object batch length differs from its declaration.")
            return list(batch)
        return split_objects
    children = ({key: _object_tree_splitter(child) for key, child in spec.items()}
                if isinstance(spec, dict) else tuple(_object_tree_splitter(child) for child in spec))

    def split(batch):
        if isinstance(spec, dict):
            if not isinstance(batch, dict) or tuple(batch) != tuple(spec):
                raise TypeError("Batch value must match its declared mapping structure.")
            parts = {key: child(batch[key]) for key, child in children.items()}
            lengths = {len(value) for value in parts.values()}
        else:
            kind = tuple if isinstance(spec, tuple) else list
            if not isinstance(batch, kind) or len(batch) != len(spec):
                raise TypeError("Batch value must match its declared sequence structure.")
            parts = [child(value) for child, value in zip(children, batch)]
            lengths = {len(value) for value in parts}
        if len(lengths) > 1:
            raise TypeError("Declared batch fields have different lengths.")
        size = next(iter(lengths), 0)
        if isinstance(spec, dict):
            return [{key: value[index] for key, value in parts.items()} for index in range(size)]
        if is_namedtuple(batch):
            return [type(batch)(*(value[index] for value in parts)) for index in range(size)]
        return [kind(value[index] for value in parts) for index in range(size)]

    return split


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
