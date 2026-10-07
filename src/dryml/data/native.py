"""Dependency-light prepared Dataset cursors for native training integrations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from dryml.data.dataset import DatasetExhaustedError


class PreparedDataset:
    """Plan one Dataset pipeline and reopen isolated traversals for native consumers.

    Args:
        dataset: Dataset pipeline whose declared stream graph is prepared once.

    Attributes:
        dataset: The unchanged source Dataset pipeline.
        execution_level: ``"stream"`` when the qualified stream graph was
            prepared, or ``"eager"`` when an unqualified operator retains its
            documented ordinary Dataset iteration fallback.

    A prepared graph is constructed without opening a source. Each
    :meth:`iterator` call returns an independent closeable cursor, so a consumer
    can stop, resume from a separately opened cursor, or traverse concurrently
    without sharing source position. This carrier owns no traversal and does not
    update training progress, checkpoint state, or exposure.
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.graph = dataset.method_graph()
        try:
            self.graph.learn()
        except NotImplementedError as error:
            if not str(error).startswith("Dataset graph planning does not support"):
                raise
            # A partially built graph can retain selected child Methods. Reset it
            # before preserving ordinary eager Dataset behavior for this invocation.
            self.graph.eager()
            self.execution_level = "eager"
        else:
            self.execution_level = "stream"

    def iterator(self):
        """Open one independent closeable traversal at the prepared execution level.

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

        if self.execution_level == "stream":
            return self.graph.iterator()
        return self.dataset.iterator()

    def training_batches(self, preparation) -> "NativeTrainingCursor":
        """Open one closeable native-training batch cursor.

        Args:
            preparation: A once-planned object exposing ``prepare(x, y)`` for the
                native consumer's retained x/y conversion edges.

        Returns:
            A cursor yielding prepared ``(x, y)`` batches in authored Dataset
            order.

        Raises:
            TypeError: If ``preparation`` lacks the retained ``prepare`` operation.

        Side Effects:
            Opens no cursor until the returned cursor is advanced. Reading values
            never updates DRYML training progress or checkpoints.
        """

        if not callable(getattr(preparation, "prepare", None)):
            raise TypeError("Native training data requires a prepared x/y handoff.")
        return NativeTrainingCursor(self, preparation)


class NativeTrainingCursor(Iterator[tuple[Any, Any]]):
    """Closeable prepared x/y batch cursor whose reads have no progress side effects.

    Args:
        dataset: Prepared Dataset that opens this cursor's independent traversal.
        preparation: Retained x/y handoff carrier applied exactly once per yield.

    A finite declared Dataset that exhausts early raises
    :class:`DatasetExhaustedError`; ordinary exhaustion closes the source. Closing
    is idempotent and forwards to the owned Dataset/graph cursor. Adapter reads do
    not advance accepted-update, exposure, or checkpoint state.
    """

    def __init__(self, dataset: PreparedDataset, preparation) -> None:
        self._dataset = dataset
        self._preparation = preparation
        self._cursor = None
        cardinality = dataset.dataset.yield_cardinality()
        self._expected = cardinality.require_finite() if cardinality.is_finite else None
        self._yielded = 0
        self._closed = False

    def __iter__(self) -> "NativeTrainingCursor":
        """Return this native batch cursor as its own iterator."""

        return self

    def __next__(self) -> tuple[Any, Any]:
        """Return one prepared batch and close the owned traversal on failure/end."""

        if self._closed:
            raise StopIteration
        if self._cursor is None:
            self._cursor = self._dataset.iterator()
        try:
            value = next(self._cursor)
        except StopIteration as error:
            self.close()
            if self._expected is not None and self._yielded != self._expected:
                raise DatasetExhaustedError(self._expected, self._yielded) from error
            raise
        except BaseException as error:
            self._close_after_failure(error)
            raise
        try:
            x, y = value
            self._dataset.dataset.examples_in((x, y))
            result = self._preparation.prepare(x, y)
        except BaseException as error:
            self._close_after_failure(error)
            raise
        self._yielded += 1
        return result

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

        if self._closed:
            if type(n) is not int:
                raise TypeError("skip count must be a nonnegative exact int.")
            if n < 0:
                raise ValueError("skip count must be non-negative.")
            if n:
                raise DatasetExhaustedError(n, 0)
            return
        if self._cursor is None:
            self._cursor = self._dataset.iterator()
        try:
            self._cursor.skip(n)
        except BaseException as error:
            self._close_after_failure(error)
            raise
        self._yielded += n

    def close(self) -> None:
        """Close the owned Dataset/graph traversal once."""

        if self._closed:
            return
        self._closed = True
        if self._cursor is not None:
            self._cursor.close()

    def _close_after_failure(self, primary: BaseException) -> None:
        """Close after an iteration failure without masking the primary error."""

        if self._closed:
            return
        try:
            self.close()
        except BaseException as cleanup:
            add_note = getattr(primary, "add_note", None)
            if add_note is not None:
                add_note(f"Native training cursor cleanup also failed: {type(cleanup).__name__}: {cleanup}")


__all__ = ["NativeTrainingCursor", "PreparedDataset"]
