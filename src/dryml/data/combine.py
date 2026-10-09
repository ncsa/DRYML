from __future__ import annotations

from typing import Any

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import merge_spec_trees
from dryml.data.dataset import Dataset


def _pack_tree(args: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> Any:
    kwargs = kwargs or {}
    if args and kwargs:
        raise ValueError("Zip accepts positional sources or keyword sources, not both.")
    if kwargs:
        return dict(kwargs)
    if len(args) == 1:
        return args[0]
    return args


def _validate_tree_key(key):
    if not isinstance(key, (str, int)):
        raise TypeError(f"Zip dict keys must be str or int, got {type(key).__name__}.")


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
    raise TypeError(f"Zip expects Dataset leaves, got {type(tree).__name__}.")


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
    raise TypeError(f"Zip expects Dataset leaves, got {type(tree).__name__}.")


def _min_cardinality(cardinalities):
    cardinalities = tuple(cardinalities)
    if any(cardinality.is_finite and cardinality.require_finite() == 0 for cardinality in cardinalities):
        return Cardinality.finite(0)
    finite_values = []
    saw_infinite = False
    for cardinality in cardinalities:
        if cardinality.is_unknown:
            return Cardinality.UNKNOWN
        if cardinality.is_infinite:
            saw_infinite = True
        else:
            finite_values.append(cardinality.require_finite())

    if finite_values:
        return Cardinality.finite(min(finite_values))
    if saw_infinite:
        return Cardinality.INFINITE
    return Cardinality.UNKNOWN


def _sum_cardinality(cardinalities):
    total = 0
    saw_unknown = False
    for cardinality in cardinalities:
        if cardinality.is_infinite:
            return Cardinality.INFINITE
        if cardinality.is_unknown:
            saw_unknown = True
        else:
            total += cardinality.require_finite()

    if saw_unknown:
        return Cardinality.UNKNOWN
    return Cardinality.finite(total)


class Zip(Dataset):
    """Combine one or more datasets in parallel into matching element trees."""

    _stream_operator = "zip"

    def __init__(self, *sources, **named_sources):
        self.sources = _pack_tree(sources, named_sources)
        if not tuple(_iter_dataset_leaves(self.sources)):
            raise ValueError("Zip requires at least one Dataset leaf.")
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
                raise TypeError(f"Zip expects Dataset leaves, got {type(tree).__name__}.")

            yield build(self.sources)

    def __len__(self) -> Cardinality:
        return _min_cardinality(
            ds.yield_cardinality() for ds in _iter_dataset_leaves(self.sources)
        )

    def example_cardinality(self) -> Cardinality:
        """Return a total only when every zipped branch is provably aligned.

        Returns:
            A shared source example cardinality when every branch has the same
            known yield and example total; zero when any branch is known empty;
            otherwise ``Cardinality.UNKNOWN``.

        Side Effects:
            None. This inspects declarations and never opens source iterators.
        """

        output = self.yield_cardinality()
        if output.is_finite and output.require_finite() == 0:
            return Cardinality.finite(0)
        sources = tuple(_iter_dataset_leaves(self.sources))
        yields = tuple(ds.yield_cardinality() for ds in sources)
        examples = tuple(ds.example_cardinality() for ds in sources)
        if any(value.is_unknown for value in yields) or any(value.is_unknown for value in examples):
            return Cardinality.UNKNOWN
        if len(set(yields)) != 1 or len(set(examples)) != 1:
            return Cardinality.UNKNOWN
        return examples[0]


class Chain(Dataset):
    """Yield all elements from each source dataset in sequence."""

    _stream_operator = "chain"

    def __init__(self, *sources: Dataset):
        if not sources:
            raise ValueError("Chain requires at least one Dataset.")
        for source in sources:
            if not isinstance(source, Dataset):
                raise TypeError(f"Chain expects Dataset sources, got {type(source).__name__}.")
        self.sources = tuple(sources)
        super().__init__(spec=merge_spec_trees([source.spec for source in self.sources]))

    def __iter__(self):
        for source in self.sources:
            yield from source

    def __len__(self) -> Cardinality:
        return _sum_cardinality(source.yield_cardinality() for source in self.sources)

    def example_cardinality(self) -> Cardinality:
        """Sum source totals only when their batch representations are compatible.

        Returns:
            The sum of all known source example totals for uniformly batched or
            uniformly unbatched sources; otherwise ``Cardinality.UNKNOWN``.

        Side Effects:
            None. This inspects declarations and never opens source iterators.
        """

        examples = tuple(source.example_cardinality() for source in self.sources)
        if any(value.is_unknown for value in examples):
            return Cardinality.UNKNOWN
        modes = {_example_mode(source) for source in self.sources}
        if None in modes or len(modes) != 1:
            return Cardinality.UNKNOWN
        return _sum_cardinality(examples)


def _example_mode(dataset: Dataset) -> str | None:
    """Classify one complete TensorSpec tree's top-level example representation."""

    try:
        from dryml.data.dataset import _spec_tensor_leaves

        leaves = _spec_tensor_leaves(dataset.spec, strict=True)
    except (TypeError, ValueError):
        return None
    if not leaves:
        return None
    batched = {leaf.batched for leaf in leaves}
    return "batched" if batched == {True} else "unbatched" if batched == {False} else None


__all__ = ["Chain", "Zip"]
