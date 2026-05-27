from __future__ import annotations

from typing import Any

from dryml.core2.cardinality import Cardinality
from dryml.data.dataset import Dataset


def _pack_tree(args: tuple[Any, ...]) -> Any:
    if len(args) == 1:
        return args[0]
    return args


def _validate_tree_key(key):
    if not isinstance(key, (str, int)):
        raise TypeError(f"Pack dict keys must be str or int, got {type(key).__name__}.")


def _map_dataset_tree(tree, fn):
    if isinstance(tree, Dataset):
        return fn(tree)
    if isinstance(tree, dict):
        for key in tree:
            _validate_tree_key(key)
        return {k: _map_dataset_tree(v, fn) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_map_dataset_tree(v, fn) for v in tree)
    if isinstance(tree, list):
        return [_map_dataset_tree(v, fn) for v in tree]
    raise TypeError(f"Pack expects Dataset leaves, got {type(tree).__name__}.")


def _iter_dataset_leaves(tree):
    if isinstance(tree, Dataset):
        yield tree
        return
    if isinstance(tree, dict):
        for key, v in tree.items():
            _validate_tree_key(key)
            yield from _iter_dataset_leaves(v)
        return
    if isinstance(tree, (tuple, list)):
        for v in tree:
            yield from _iter_dataset_leaves(v)
        return
    raise TypeError(f"Pack expects Dataset leaves, got {type(tree).__name__}.")


def _min_cardinality(cardinalities):
    finite_values = []
    saw_infinite = False
    for cardinality in cardinalities:
        if isinstance(cardinality, Cardinality):
            if cardinality.is_unknown:
                return Cardinality.UNKNOWN
            if cardinality.is_infinite:
                saw_infinite = True
            else:
                finite_values.append(cardinality.require_finite())
        else:
            finite_values.append(int(cardinality))

    if finite_values:
        return Cardinality.finite(min(finite_values))
    if saw_infinite:
        return Cardinality.INFINITE
    return Cardinality.UNKNOWN


class Pack(Dataset):
    """Combine one or more datasets into matching element trees."""

    def __init__(self, *sources):
        self.sources = _pack_tree(sources)
        if not tuple(_iter_dataset_leaves(self.sources)):
            raise ValueError("Pack requires at least one Dataset leaf.")
        super().__init__(spec=_map_dataset_tree(self.sources, lambda ds: ds.spec))

    def __iter__(self):
        iterators = [iter(ds) for ds in _iter_dataset_leaves(self.sources)]

        while True:
            values = []
            for it in iterators:
                try:
                    values.append(next(it))
                except StopIteration:
                    return

            value_iter = iter(values)

            def build(tree):
                if isinstance(tree, Dataset):
                    return next(value_iter)
                if isinstance(tree, dict):
                    for key in tree:
                        _validate_tree_key(key)
                    return {k: build(v) for k, v in tree.items()}
                if isinstance(tree, tuple):
                    return tuple(build(v) for v in tree)
                if isinstance(tree, list):
                    return [build(v) for v in tree]
                raise TypeError(f"Pack expects Dataset leaves, got {type(tree).__name__}.")

            yield build(self.sources)

    def __len__(self) -> Cardinality:
        return _min_cardinality(ds.__len__() for ds in _iter_dataset_leaves(self.sources))


class Zip(Pack):
    """Compatibility alias for Pack(ds1, ds2, ...)."""
