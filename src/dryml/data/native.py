"""Frozen source acquisition adapters and prepared native-training Dataset cursors.

Preparation detaches small controls and topology without acquiring sources or
cloning payload bytes. Each cursor owns independent live progress and resources;
ordinary pulls neither extract continuation state nor encode it for persistence.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, TYPE_CHECKING

from dryml.core.tensor_spec import SpecTree

from dryml.methods.stream import DatasetExhaustedError, _CursorLifecycle

if TYPE_CHECKING:
    from .dataset import Dataset, DatasetCursor
    from dryml.methods.stream import StreamGraphCursor


@dataclass(frozen=True, slots=True)
class _ConfigTree:
    """Detached container topology, with payload leaves retained by reference."""

    kind: type
    items: tuple

    def unpack(self, memo=None):
        memo = {} if memo is None else memo
        if id(self) in memo:
            return memo[id(self)]
        if self.kind is dict:
            result = {key: _unpack_config(value, memo) for key, value in self.items}
        else:
            result = self.kind(_unpack_config(value, memo) for value in self.items)
        memo[id(self)] = result
        return result


def _unpack_config(value, memo=None):
    return value.unpack(memo) if isinstance(value, _ConfigTree) else value


def _freeze_config(value, *, small=False, active=None, memo=None):
    """Freeze plain small configuration containers, never copy payload leaves.

    Large factory argument containers are payload bindings, not small controls.
    The bounded preflight avoids traversing or cloning a source-sized list just
    to discover that distinction. Array-source topology is always detached.
    """
    if type(value) not in (dict, list, tuple):
        return value
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if small and not _small_config(value, [256], set()):
        return value
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError("Prepared source configuration contains a container cycle.")
    active.add(id(value))
    try:
        items = (tuple((key, _freeze_config(child, active=active, memo=memo)) for key, child in value.items())
                 if type(value) is dict else tuple(_freeze_config(child, active=active, memo=memo) for child in value))
        frozen = _ConfigTree(type(value), items)
        memo[id(value)] = frozen
        return frozen
    finally:
        active.remove(id(value))


def _small_config(value, budget, active):
    budget[0] -= 1
    if budget[0] < 0:
        return False
    if type(value) not in (dict, list, tuple):
        return True
    if id(value) in active:
        raise ValueError("Prepared source configuration contains a container cycle.")
    if len(value) > budget[0]:
        return False
    active.add(id(value))
    try:
        return all(_small_config(child, budget, active) for child in
                   (value.values() if type(value) is dict else value))
    finally:
        active.remove(id(value))


@dataclass(frozen=True, slots=True)
class _SourceBinding:
    """One inert source declaration and its frozen acquisition callable."""

    kind: str
    binding: object
    open: object


@dataclass(frozen=True, slots=True)
class _GeneratorBinding:
    """Frozen factory controls consumed by the existing owned generator cursor."""

    gen_factory: object
    args: tuple
    kwargs: tuple
    cardinality: object
    _declared_example_count: object
    _spec: object
    examples_in: object
    seed: int | None
    seed_aware: bool
    epoch_seed_version: int

    @property
    def factory_args(self):
        return tuple(_unpack_config(value) for value in self.args)

    @property
    def factory_kwargs(self):
        return {key: _unpack_config(value) for key, value in self.kwargs}

    def open(self, epoch):
        from .source import _GeneratorCursor

        memo = {}
        invocation = replace(self,
            args=tuple(_unpack_config(value, memo) for value in self.args),
            kwargs=tuple((key, _unpack_config(value, memo)) for key, value in self.kwargs),
        )
        return _GeneratorCursor(invocation, epoch)


def _prepare_source(dataset, spec):
    """Bind concrete built-in source controls without acquisition or payload copies.

    Unknown/custom source implementations retain their trusted compatibility
    opener; this binding does not certify their configuration or recovery.
    """
    from .dataset import Dataset, DatasetCursor
    from .source import (ArrayDataset, GeneratorDataset, NpyFileDataset,
                         TFDSAdapter, TorchDatasetAdapter, _IndexedDatasetCursor, _tree_index)
    from dryml.methods.signature import spec_node

    if type(dataset) is GeneratorDataset:
        memo = {}
        binding = _GeneratorBinding(
            dataset.gen_factory,
            tuple(_freeze_config(value, small=True, memo=memo) for value in dataset.factory_args),
            tuple((key, _freeze_config(value, small=True, memo=memo)) for key, value in dataset.factory_kwargs.items()),
            dataset.cardinality, dataset._declared_example_count, spec_node(spec),
            partial(Dataset._examples_in_spec, spec),
            dataset.seed, dataset.seed_aware, dataset.epoch_seed_version,
        )
        return _SourceBinding("generator", binding, binding.open)
    if type(dataset) is ArrayDataset:
        tree, length = _freeze_config(dataset.arrays), dataset._length

        def open_array(epoch):
            arrays = _unpack_config(tree)
            return _IndexedDatasetCursor(length, partial(_tree_index, arrays))

        return _SourceBinding("array", (tree, length), open_array)
    if type(dataset) is NpyFileDataset:
        import numpy as np

        files, allow_pickle = tuple(dataset.files), dataset.allow_pickle
        return _SourceBinding("npy", (files, allow_pickle), lambda epoch: _IndexedDatasetCursor(
            len(files), lambda index: np.load(files[index], allow_pickle=allow_pickle)))
    if type(dataset) is TFDSAdapter:
        native, as_numpy = dataset.dataset, dataset.as_numpy
        return _SourceBinding("tfds", (native, as_numpy), lambda epoch: DatasetCursor(
            native.as_numpy_iterator() if as_numpy else iter(native)))
    if type(dataset) is TorchDatasetAdapter:
        native = dataset.dataset

        def open_torch(epoch):
            import torch.utils.data as tud

            return DatasetCursor(iter(native) if isinstance(native, tud.IterableDataset)
                                 else (native[index] for index in range(len(native))))

        return _SourceBinding("torch", native, open_torch)
    opener = getattr(dataset, "iterator_for_epoch", None)
    if not callable(opener):
        iterator = dataset.iterator
        opener = lambda epoch: iterator()
    return _SourceBinding("custom", dataset, opener)


class PreparedDataset:
    """Plan one Dataset pipeline and reopen isolated traversals for native consumers.

    Args:
        dataset: Dataset pipeline whose declared stream graph is prepared once.

    Attributes:
        dataset: Inspection reference to the source Dataset recipe; qualified
            execution does not reread its controls.
        spec: Captured output SpecTree from the prepared plan (or eager fallback
            declaration), independent of later shared Dataset metadata changes.
        execution_level: ``"stream"`` when the qualified stream graph was
            prepared, or ``"eager"`` when an unqualified operator retains its
            documented ordinary Dataset iteration fallback.

    A prepared graph freezes operator controls, source acquisition declarations,
    small factory configuration containers, topology, specs and selections
    without opening a source. Payload bindings are retained, not cloned. Each
    :meth:`iterator` call returns an independent closeable cursor, so a consumer
    can stop, resume from a separately opened cursor, or traverse concurrently
    without sharing source position. This carrier owns no traversal and does not
    update training progress, checkpoint state, or exposure.
    Preparation and fallback never publish or reset child Method state; failed
    planning cannot leave partial child selections requiring cleanup.

    Raises:
        ImplementationSelectionError: If a qualified declared Method cannot be
            selected. Other inference/compilation and backend errors propagate.
        TypeError, ValueError: For invalid specs or unsupported source controls,
            including cycles in small factory configuration containers.

    Custom/unqualified execution retains its established eager semantics and
    does not gain a frozen-configuration or continuation guarantee.
    """

    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset
        self.graph = dataset.method_graph()
        try:
            self.graph.learn()
        except NotImplementedError as error:
            if not str(error).startswith("Dataset graph planning does not support"):
                raise
            self.execution_level = "eager"
        else:
            self.execution_level = "stream"
        from dryml.methods.signature import spec_from_node

        if self.execution_level == "stream":
            root = self.graph._stream_plan._prepared.root
            self._signature = root.signature
            self._cardinality = root.cardinality
            count_spec = spec_from_node(self._signature)
        else:
            self._eager_spec = dataset.spec
            self._cardinality = None
            count_spec = self._eager_spec
        self._count_examples = partial(dataset._examples_in_spec, count_spec)

    @property
    def spec(self) -> SpecTree:
        """Return captured output metadata, detached for qualified stream plans.

        Qualified mutable list/dict structure is detached on each read; immutable
        TensorSpec leaves are shared. Eager fallback retains its original declared
        spec. Reading performs no selection or source acquisition.
        """
        from dryml.methods.signature import spec_from_node

        if self.execution_level == "eager":
            return self._eager_spec
        return spec_from_node(self._signature)

    def examples_in(self, value: Any) -> int:
        """Validate/count one value against captured output metadata.

        Args:
            value: Runtime value with the prepared output structure and batching.

        Returns:
            Positive example count using Dataset's established validation rules.

        Raises:
            TypeError, ValueError: If structure, batches, or counts are invalid.

        Side Effects:
            Reads captured specs only; no selection, inference, or source access.
        """
        return self._count_examples(value)

    def _require_declared_selection(self) -> None:
        """Reject Maps needing cursor-local discovery, including eager wrappers."""
        from dryml.methods import Method

        for node in self.graph.method_nodes:
            if (node.method_type is not None and issubclass(node.method_type, Method)
                    and node.selected is None):
                raise TypeError("Native training requires complete declared Method inputs before source acquisition.")
        if self.execution_level == "eager":
            from dryml.core.utils.recurse import iter_leaves
            from dryml.data.dataset import Dataset, Map
            from dryml.methods import ImplementationSelectionError

            visited = set()

            def admit(dataset):
                if id(dataset) in visited:
                    return
                visited.add(id(dataset))
                if isinstance(dataset, Map):
                    try:
                        dataset.method.find_implementation(dataset.src.spec)
                    except ImplementationSelectionError as error:
                        raise TypeError("Native training requires complete declared Method inputs before source acquisition.") from error
                fields = vars(dataset)
                for source in iter_leaves((fields.get("src"), fields.get("sources", ()))):
                    if isinstance(source, Dataset):
                        admit(source)

            admit(self.dataset)

    def iterator(self, *, epoch: int = 0) -> DatasetCursor[Any] | StreamGraphCursor:
        """Open one independent closeable traversal at the prepared execution level.

        Args:
            epoch: Logical Dataset epoch. A nonzero epoch selects a supporting
                seed-aware source directly without traversing earlier epochs.

        Returns:
            A graph cursor for qualified pipelines or the Dataset's ordinary
            cursor for the documented eager fallback.

        Raises:
            ValueError: If ``epoch`` is not a nonnegative exact integer.
            Source-specific acquisition or iteration errors from the underlying
            Dataset pipeline.

        Side Effects:
            Qualified cursor creation acquires nothing; its first advancement may
            acquire source resources. Eager fallback may acquire on creation.
            The reusable plan holds source bindings but no live cursor position
            or acquired traversal resources.
        """

        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative exact int.")
        if self.execution_level == "stream":
            return self.graph.iterator(epoch=epoch)
        opener = getattr(self.dataset, "iterator_for_epoch", None)
        if epoch and callable(opener):
            return opener(epoch)
        return self.dataset.iterator()

    def training_batches(self, preparation, *, epoch: int = 0) -> "NativeTrainingCursor":
        """Open one closeable native-training batch cursor.

        Args:
            preparation: A once-planned object exposing ``prepare(x, y)`` for the
                native consumer's retained x/y conversion edges.
            epoch: Logical Dataset epoch for a supporting seed-aware source.

        Returns:
            A cursor yielding prepared ``(x, y)`` batches in authored Dataset
            order.

        Raises:
            TypeError: If ``preparation`` lacks the retained ``prepare`` operation.
                Also raised when a Map still needs input discovery.

        Side Effects:
            Opens no cursor until the returned cursor is advanced. Reading values
            never updates DRYML training progress or checkpoints.
        """

        if not callable(getattr(preparation, "prepare", None)):
            raise TypeError("Native training data requires a prepared x/y handoff.")
        return NativeTrainingCursor(self, preparation, epoch=epoch)


class NativeTrainingCursor(_CursorLifecycle, Iterator[tuple[Any, Any]]):
    """Closeable prepared x/y batch cursor whose reads have no progress side effects.

    Args:
        dataset: Prepared Dataset that opens this cursor's independent traversal.
        preparation: Retained x/y handoff carrier applied exactly once per yield.
        epoch: Exact nonnegative logical epoch forwarded to the Dataset traversal.

    A finite declared Dataset that exhausts early raises
    :class:`DatasetExhaustedError`; ordinary exhaustion closes the source. Closing
    is idempotent and forwards to the owned Dataset/graph cursor. Adapter reads do
    not advance accepted-update, exposure, or checkpoint state. ``observed_eof``
    becomes true only after count validation succeeds, even when EOF cleanup
    fails. Use as a context manager around updates to retain consumer failures
    over cleanup errors; non-pull StopIteration is a causal RuntimeError.
    Unresolved Map selections raise TypeError during cursor construction,
    before source acquisition, even when output declarations name a native backend.
    """

    def __init__(self, dataset: PreparedDataset, preparation, *, epoch: int = 0) -> None:
        super().__init__()
        dataset._require_declared_selection()
        self._dataset = dataset
        self._preparation = preparation
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative exact int.")
        self._epoch = epoch
        self._cursor = None
        cardinality = dataset._cardinality
        if cardinality is None:
            cardinality = self._call(dataset.dataset.yield_cardinality)
        self._expected = cardinality.require_finite() if cardinality.is_finite else None
        self._yielded = 0

    def __iter__(self) -> "NativeTrainingCursor":
        """Return this native batch cursor as its own iterator."""

        return self

    def _read(self) -> tuple[Any, Any]:
        if self._cursor is None:
            self._cursor = self._own(self._call(self._dataset.iterator, epoch=self._epoch))
        try:
            value = self._pull(self._cursor)
        except StopIteration as error:
            if self._expected is not None and self._yielded != self._expected:
                raise DatasetExhaustedError(self._expected, self._yielded) from error
            raise
        result = self._call(self._prepare, value)
        self._yielded += 1
        return result

    def _prepare(self, value):
        x, y = value
        self._dataset.examples_in((x, y))
        return self._preparation.prepare(x, y)

    def skip(self, n: int) -> None:
        """Skip raw authored batches without executing their conversion edges.

        Args:
            n: Exact nonnegative number of completed batches to discard.

        Raises:
            TypeError: If ``n`` is not an exact integer.
            ValueError: If ``n`` is negative.
            DatasetExhaustedError: If fewer than ``n`` batches remain.

        Side Effects:
            Advances only this cursor's source position. Discarded batches do not
            execute preparation and do not update training progress.
        """

        self._validate_skip_count(n)
        if n == 0:
            return
        if self._closed:
            raise DatasetExhaustedError(n, 0)
        try:
            if self._cursor is None:
                self._cursor = self._own(self._call(self._dataset.iterator, epoch=self._epoch))
            self._call(self._cursor.skip, n)
        except BaseException as error:
            if isinstance(error, DatasetExhaustedError):
                self._yielded += error.yielded
            self._close_after_failure(error)
            raise
        self._yielded += n


__all__ = ["NativeTrainingCursor", "PreparedDataset"]
