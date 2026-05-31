from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable
import itertools
import numpy as np

from dryml.core2.tensor_spec import (
    SpecTree,
    TensorSpec,
    SpecHint,
    as_tensor_spec,
    detect_spec_tree,
    unbatch_spec_tree,
)
from dryml.core2.cardinality import Cardinality
from dryml.core2.utils.recurse import first_leaf, iter_leaves, map_leaves
from .dataset import Dataset


# ----------------------------------------------------------------------
# Source datasets
# ----------------------------------------------------------------------

class SourceDataset(Dataset):
    pass


def _tree_index(x: Any, i: int) -> Any:
    return map_leaves(x, lambda leaf: leaf[i])


def _fresh_iterable(factory: Callable[..., Any], *args, **kwargs) -> Iterable[Any]:
    candidate = factory(*args, **kwargs)
    if not hasattr(candidate, "__iter__") and callable(candidate):
        candidate = candidate()
    if not hasattr(candidate, "__iter__"):
        raise TypeError(
            f"generator_fn returned {type(candidate).__name__}, which is not iterable."
        )
    return candidate


class GeneratorDataset(SourceDataset):
    """
    Dataset backed by a callable that returns a fresh iterator/generator
    each time it is invoked.
    """

    def __init__(
        self,
        gen_factory: Callable[[], Iterable[Any]],
        *factory_args,
        cardinality: Cardinality =Cardinality.UNKNOWN,
        spec: SpecTree|str|int|dict[str,str|int]|None=None,
        **factory_kwargs
    ):
        super().__init__()

        if not callable(gen_factory):
            raise TypeError("generator_fn must be callable and return a fresh iterator.")

        self.gen_factory = gen_factory
        self.factory_args = factory_args
        self.factory_kwargs = factory_kwargs
        self.cardinality = cardinality
        if spec is None:
            self._spec = None
        elif isinstance(spec, SpecHint):
            self._spec = detect_spec_tree(iter(self), spec)
        else:
            leaf = first_leaf(spec)
            if isinstance(leaf, TensorSpec):
                self._spec = spec
            else:
                self._spec = detect_spec_tree(iter(self), SpecHint.build(spec))


    def __iter__(self) -> Iterator[Any]:
        it = _fresh_iterable(
            self.gen_factory,
            *self.factory_args,
            **self.factory_kwargs)
        if self.cardinality.is_finite:
            yield from itertools.islice(it, int(self.cardinality))
        else:
            yield from it

    def __len__(self) -> Cardinality:
        return self.cardinality


class ArrayDataset(SourceDataset):
    """
    Dataset backed by one or more aligned stacked arrays.

    Examples
    --------
    arrays = {
        "cart": np.zeros((100, 3), dtype=np.float32),
        "sph": np.zeros((100, 2), dtype=np.float32),
    }

    ds = ArrayDataset(arrays)
    x0 = ds.peek()
    # x0["cart"].shape == (3,)
    # x0["sph"].shape == (2,)
    """

    def __init__(
        self,
        arrays: Any,
        *,
        spec: SpecTree | None = None,
        batched = True,
        validate_lengths: bool = True,
    ):
        if validate_lengths:
            lengths = list(map(len,iter_leaves(arrays)))
            if not lengths:
                raise ValueError("ArrayDataset requires at least one leaf.")
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"All ArrayDataset leaves must agree on leading length, got {lengths}."
                )
            self._length = lengths[0]
        else:
            self._length = len(next(iter_leaves(arrays)))

        self.arrays = arrays

        if spec is None:
            spec = as_tensor_spec(arrays, batched=batched)
            if batched:
                spec = unbatch_spec_tree(spec)

        super().__init__(spec=spec)

    def __iter__(self) -> Iterator[Any]:
        for i in range(self._length):
            yield _tree_index(self.arrays, i)

    def __len__(self) -> int:
        return self._length

    def peek(self) -> Any:
        if self._length == 0:
            raise ValueError("Cannot peek an empty dataset.")
        return _tree_index(self.arrays, 0)


class NpyFileDataset(SourceDataset):
    """Dataset whose elements are loaded from sorted ``.npy`` files."""

    def __init__(
        self,
        root: str | Path,
        *,
        pattern: str = "*.npy",
        spec: SpecTree | None = None,
        batched: bool = False,
        allow_pickle: bool = False,
    ):
        self.root = Path(root)
        self.pattern = pattern
        self.allow_pickle = allow_pickle
        self.files = tuple(sorted(self.root.glob(pattern)))

        if spec is None:
            if not self.files:
                raise ValueError("NpyFileDataset requires spec when no files match.")
            spec = as_tensor_spec(
                np.load(self.files[0], allow_pickle=allow_pickle),
                batched=batched,
            )

        super().__init__(spec=spec)

    def __iter__(self):
        for path in self.files:
            yield np.load(path, allow_pickle=self.allow_pickle)

    def __len__(self) -> Cardinality:
        return Cardinality.finite(len(self.files))


class TFDSAdapter(SourceDataset):
    """
    Adapter for tf.data.Dataset.

    Notes
    -----
    - If `as_numpy=True`, iteration uses `dataset.as_numpy_iterator()`.
    - If `spec` is not provided, it is derived from `dataset.element_spec`.
    - Batch semantics are ambiguous in flat TensorFlow specs, so if the
      TF dataset yields batches and you want that reflected in DRYML specs,
      pass `assume_batched=True` or provide `spec` explicitly.
    """

    def __init__(
        self,
        name,
        *,
        split: list[str]|str|None=None,
        batch_size: int|None=None,
        as_supervised: bool=False,
        as_numpy: bool = False,
        assume_batched: bool | None = None,
        spec: SpecTree | None = None,
    ):
        import tensorflow_datasets as tfds

        self.dataset = tfds.load(
            name,
            split=split,
            batch_size=batch_size,
            as_supervised=as_supervised)
        self.as_numpy = as_numpy
        self.assume_batched = (batch_size is not None) if assume_batched is None else assume_batched

        if spec is None:
            if as_numpy:
                import dryml.numpy

                try:
                    first = next(self.dataset.as_numpy_iterator())
                except StopIteration as e:
                    raise ValueError("TFDSAdapter requires spec when the TFDS split is empty.") from e
                spec = as_tensor_spec(first, batched=self.assume_batched)
            else:
                import dryml.tf

                spec = as_tensor_spec(
                    self.dataset.element_spec,
                    batched=self.assume_batched,
                )

        super().__init__(spec=spec)

    def __iter__(self) -> Iterator[Any]:
        if self.as_numpy:
            yield from self.dataset.as_numpy_iterator()
        else:
            yield from self.dataset

    def __len__(self) -> Cardinality:
        card = self.dataset.cardinality()
        card_val = int(card.numpy())

        if card_val == -2:
            return Cardinality.UNKNOWN
        if card_val == -1:
            return Cardinality.INFINITE

        return Cardinality.finite(card_val)


class TorchDatasetAdapter(SourceDataset):
    """
    Adapter for torch.utils.data.Dataset and IterableDataset.

    Notes
    -----
    - For map-style datasets, iteration uses dataset[i].
    - For iterable datasets, iteration delegates to iter(dataset).
    - If `spec` is omitted and `infer_spec=True`, spec is inferred from `peek()`.
      For iterable datasets this assumes the dataset is safely re-iterable.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        spec: SpecTree | None = None,
        infer_spec: bool = False,
    ):
        try:
            import torch.utils.data as tud  # type: ignore
        except Exception as e:
            raise ImportError("PyTorch is required for TorchDatasetAdapter.") from e

        if not isinstance(dataset, (tud.Dataset, tud.IterableDataset)):
            raise TypeError(
                "dataset must be a torch.utils.data.Dataset or IterableDataset, "
                f"got {type(dataset).__name__}."
            )

        self.dataset = dataset

        if spec is None and infer_spec:
            spec = as_tensor_spec(self.peek())

        super().__init__(spec=spec)

    def __iter__(self) -> Iterator[Any]:
        try:
            import torch.utils.data as tud  # type: ignore
        except Exception as e:
            raise ImportError("PyTorch is required for TorchDatasetAdapter.") from e

        if isinstance(self.dataset, tud.IterableDataset):
            yield from iter(self.dataset)
            return

        for i in range(len(self.dataset)):
            yield self.dataset[i]

    def __len__(self) -> Cardinality:
        if hasattr(self.dataset, "__len__"):
            return Cardinality.finite(int(len(self.dataset)))
        return super().__len__()

    def peek(self) -> Any:
        try:
            import torch.utils.data as tud  # type: ignore
        except Exception as e:
            raise ImportError("PyTorch is required for TorchDatasetAdapter.") from e

        if isinstance(self.dataset, tud.IterableDataset):
            return super().peek()

        n = len(self.dataset)
        if n == 0:
            raise ValueError("Cannot peek an empty dataset.")
        return self.dataset[0]
