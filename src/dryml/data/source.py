from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable
import itertools
import numpy as np

from dryml.core.tensor_spec import (
    SpecTree,
    TensorSpec,
    SpecHint,
    as_tensor_spec,
    detect_spec_tree,
    unbatch_spec_tree,
)
from dryml.core.cardinality import Cardinality
from dryml.core.utils.recurse import first_leaf, iter_leaves, map_leaves
from .dataset import Dataset, DatasetCursor, DatasetExhaustedError, _normalize_cardinality


# ----------------------------------------------------------------------
# Source datasets
# ----------------------------------------------------------------------

class SourceDataset(Dataset):
    pass


class _IndexedDatasetCursor(DatasetCursor):
    """Cursor that advances a private index without fetching discarded values."""

    def __init__(self, length: int, item_at: Callable[[int], Any]) -> None:
        super().__init__(iter(()))
        self._length = length
        self._item_at = item_at

    def __next__(self) -> Any:
        """Fetch the current indexed value and advance this cursor's position."""

        if self._closed or self._position >= self._length:
            self.close()
            raise StopIteration
        value = self._item_at(self._position)
        self._position += 1
        return value

    def skip(self, n: int) -> None:
        """Advance the private index exactly without reading skipped values."""

        self._validate_skip_count(n)
        actual = min(n, self._length - self._position)
        self._position += actual
        if actual != n:
            self.close()
            raise DatasetExhaustedError(n, actual)


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
    """Dataset backed by a callable that returns a fresh iterable on each open.

    Args:
        gen_factory: Callable producing a fresh iterable. A ``seed_aware`` factory
            additionally accepts a ``seed`` keyword argument.
        *factory_args: Positional arguments retained for ``gen_factory``.
        cardinality: Declared yield cardinality.
        example_count: Optional explicit total for opaque or dynamic batches.
        spec: Optional yielded-element specification or inference hint.
        seed: Base seed for the supported seed-aware source protocol.
        seed_aware: Enable the versioned per-epoch seed protocol.
        epoch_seed_version: Version of the stable epoch seed derivation protocol.
        **factory_kwargs: Keyword arguments retained for ``gen_factory``.

    ``iterator_for_epoch(epoch)`` opens a selected seed-aware epoch directly;
    ordinary iteration is epoch zero. Explicit example totals are metadata only.
    """

    def __init__(
        self,
        gen_factory: Callable[[], Iterable[Any]],
        *factory_args,
        cardinality: Cardinality = Cardinality.UNKNOWN,
        example_count: int | Cardinality | None = None,
        spec: SpecTree|str|int|dict[str,str|int]|None=None,
        seed: int | None = None,
        seed_aware: bool = False,
        epoch_seed_version: int = 1,
        **factory_kwargs
    ):
        super().__init__()

        if not callable(gen_factory):
            raise TypeError("generator_fn must be callable and return a fresh iterator.")

        self.gen_factory = gen_factory
        self.factory_args = factory_args
        self.factory_kwargs = factory_kwargs
        self.cardinality = _normalize_cardinality(cardinality)
        self._declared_example_count = (
            None if example_count is None else _normalize_cardinality(example_count)
        )
        if type(seed_aware) is not bool:
            raise TypeError("seed_aware must be an exact bool.")
        if seed is not None and type(seed) is not int:
            raise TypeError("seed must be an exact int or None.")
        if type(epoch_seed_version) is not int or epoch_seed_version <= 0:
            raise ValueError("epoch_seed_version must be a positive exact int.")
        if seed_aware and seed is None:
            raise ValueError("seed-aware GeneratorDataset requires a base seed.")
        self.seed = seed
        self.seed_aware = seed_aware
        self.epoch_seed_version = epoch_seed_version
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
        yield from self.iterator_for_epoch(0)

    def iterator_for_epoch(self, epoch: int) -> Iterator[Any]:
        """Open one selected reproducible epoch without traversing prior epochs.

        Args:
            epoch: Exact nonnegative logical epoch index.

        Returns:
            A fresh iterator bounded by finite declared yield cardinality.

        Raises:
            TypeError: If ``epoch`` is not an exact integer.
            ValueError: If ``epoch`` is negative.

        Side Effects:
            Calls the factory once. Seed-aware factories receive a deterministic,
            versioned ``seed`` keyword; opaque factories retain prior invocation.
        """

        if type(epoch) is not int:
            raise TypeError("epoch must be a nonnegative exact int.")
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        kwargs = dict(self.factory_kwargs)
        if self.seed_aware:
            kwargs["seed"] = _epoch_seed(self.seed, epoch, self.epoch_seed_version)
        values = _fresh_iterable(self.gen_factory, *self.factory_args, **kwargs)
        if self.cardinality.is_finite:
            values = itertools.islice(values, int(self.cardinality))

        observed_examples = 0
        validate_observed = (
            self._declared_example_count is not None
            and not self._declared_example_count.is_unknown
        )
        for value in values:
            if validate_observed:
                try:
                    observed_examples += self.examples_in(value)
                except TypeError:
                    validate_observed = False
            yield value
        if validate_observed:
            observed = Cardinality.finite(observed_examples)
            if observed != self._declared_example_count:
                raise ValueError(
                    "GeneratorDataset observed example count conflicts with its declaration."
                )

    def __len__(self) -> Cardinality:
        return self.cardinality

    def example_cardinality(self) -> Cardinality:
        """Return an explicit opaque-source total or ordinary spec-derived facts."""

        derived = super().example_cardinality()
        declared = self._declared_example_count
        if declared is None or declared.is_unknown:
            return derived
        if not derived.is_unknown and derived != declared:
            raise ValueError(
                "GeneratorDataset example_count conflicts with its spec and yield cardinality."
            )
        return declared


def _epoch_seed(base_seed: int, epoch: int, version: int) -> int:
    """Derive one stable nonnegative seed from a declared protocol version and epoch."""

    import hashlib

    payload = f"dryml-generator-epoch-v{version}:{base_seed}:{epoch}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


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

    def iterator(self) -> DatasetCursor[Any]:
        """Create an indexed cursor that does not materialize skipped array rows."""

        return _IndexedDatasetCursor(self._length, lambda index: _tree_index(self.arrays, index))

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

    def iterator(self) -> DatasetCursor[Any]:
        """Create an indexed cursor that loads only files whose values are read."""

        return _IndexedDatasetCursor(
            len(self.files),
            lambda index: np.load(self.files[index], allow_pickle=self.allow_pickle),
        )

    def __len__(self) -> Cardinality:
        return Cardinality.finite(len(self.files))


class TFDSAdapter(SourceDataset):
    """Adapt one TFDS split to DRYML's re-iterable Dataset contract.

    Args:
        name: TFDS builder name accepted by :func:`tensorflow_datasets.load`.
        split: Optional TFDS split expression.
        batch_size: Optional TFDS batch size.
        as_supervised: Request TFDS supervised ``(input, target)`` elements.
        as_numpy: Yield NumPy values rather than native TensorFlow tensors.
        assume_batched: Whether yielded elements are batched for spec inference;
            defaults to whether ``batch_size`` is supplied.
        spec: Optional explicit DRYML element specification.
        data_dir: Optional local TFDS data root as a string path. ``None`` uses
            TFDS's product default data directory.
        download: Exact bool passed to TFDS. It defaults to ``True`` for product
            compatibility; ``False`` requires a prepared local split and never
            permits this adapter to request a download.
        download_config: Optional TFDS ``DownloadConfig`` passed only when a
            download/prepare operation is requested by TFDS.

    Returns:
        A Dataset whose iteration delegates to the loaded TFDS dataset. In NumPy
        mode, iteration uses ``as_numpy_iterator``; otherwise it yields native
        TensorFlow values.

    Raises:
        TypeError: If ``download`` is not an exact bool.
        ImportError: If TensorFlow Datasets or the selected native spec backend is
            unavailable.
        Exception: Propagates TFDS loading, local filesystem, download, and split
            validation failures. In particular, ``download=False`` fails for a
            missing selected local dataset rather than fetching it.

    Side Effects:
        Imports TensorFlow Datasets and calls ``tfds.load``. With the default
        ``download=True``, TFDS may access the network and create/update its data
        directory; explicit ``data_dir`` confines that TFDS-managed filesystem
        work to the supplied root. ``download_config`` is forwarded unchanged to
        TFDS and can further control that preparation behavior.

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
        data_dir: str | None = None,
        download: bool = True,
        download_config: object | None = None,
    ):
        if type(download) is not bool:
            raise TypeError("TFDSAdapter download must be an exact bool.")
        import tensorflow_datasets as tfds

        load_kwargs = {
            "split": split,
            "batch_size": batch_size,
            "as_supervised": as_supervised,
            "data_dir": data_dir,
            "download": download,
        }
        if download_config is not None:
            load_kwargs["download_and_prepare_kwargs"] = {
                "download_config": download_config,
            }
        self.dataset = tfds.load(
            name,
            **load_kwargs)
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
