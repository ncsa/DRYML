from __future__ import annotations

from typing import Any

import numpy as np

from dryml.core2.utils.types import is_namedtuple


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
