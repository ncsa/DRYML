from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dryml.core.utils.types import is_namedtuple


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
        from dryml.runtime import import_configured_framework
        torch = import_configured_framework("torch")
        if isinstance(first, torch.Tensor):
            return torch.stack(items, dim=0)
    except Exception:
        pass

    # tf tensor
    try:
        from dryml.runtime import import_configured_framework
        tf = import_configured_framework("tensorflow")
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
