"""Dependency-light prepared Dataset cursors for native training integrations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from dryml.methods.stream import DatasetExhaustedError, _CursorLifecycle


class PreparedDataset:
    """Plan one Dataset pipeline and reopen isolated traversals for native consumers.

    Args:
        dataset: Dataset pipeline whose declared stream graph is prepared once.

    Attributes:
        dataset: The unchanged source Dataset pipeline.
        spec: Captured output SpecTree from the prepared plan (or eager fallback
            declaration), independent of later shared Dataset metadata changes.
        execution_level: ``"stream"`` when the qualified stream graph was
            prepared, or ``"eager"`` when an unqualified operator retains its
            documented ordinary Dataset iteration fallback.

    A prepared graph is constructed without opening a source. Each
    :meth:`iterator` call returns an independent closeable cursor, so a consumer
    can stop, resume from a separately opened cursor, or traverse concurrently
    without sharing source position. This carrier owns no traversal and does not
    update training progress, checkpoint state, or exposure.
    Preparation and fallback never publish or reset child Method state; failed
    planning cannot leave partial child selections requiring cleanup.
    """

    def __init__(self, dataset) -> None:
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
        if self.execution_level == "stream":
            from dryml.methods.signature import spec_from_node

            self.spec = spec_from_node(self.graph.nodes[-1].outputs[0].spec)
        else:
            self.spec = dataset.spec

    def examples_in(self, value) -> int:
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
        return self.dataset._examples_in_spec(self.spec, value)

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

    def iterator(self, *, epoch: int = 0):
        """Open one independent closeable traversal at the prepared execution level.

        Args:
            epoch: Logical Dataset epoch. A nonzero epoch selects a supporting
                seed-aware source directly without traversing earlier epochs.

        Returns:
            A graph cursor for qualified pipelines or the Dataset's ordinary
            cursor for the documented eager fallback.

        Raises:
            Source-specific acquisition or iteration errors from the underlying
            Dataset pipeline.

        Side Effects:
            May acquire resources for this one traversal. The prepared plan itself
            remains reusable and holds neither cursor position nor source handles.
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
