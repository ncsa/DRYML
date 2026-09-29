from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import Generic, TypeVar

from dryml.core import Object
from dryml.core.backend import discover_backends
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import SpecTree
from dryml.methods import ImplementationSelectionError


T = TypeVar("T")
_PENDING = object()


class DatasetExhaustedError(ValueError):
    """Report that a cursor could not consume a requested number of values.

    Args:
        requested: Nonnegative number of values the operation was required to
            consume.
        yielded: Number of values actually consumed before exhaustion.

    Attributes:
        requested: The requested number of values.
        yielded: The number successfully consumed before exhaustion.
    """

    def __init__(self, requested: int, yielded: int) -> None:
        self.requested = requested
        self.yielded = yielded
        super().__init__(
            f"Requested {requested} dataset elements, but the source yielded only {yielded}."
        )


class DatasetCursor(Iterator[T], Generic[T]):
    """Own one closeable Dataset traversal with a consumed-yield position.

    Args:
        iterator: Fresh iterator owned by this cursor.

    ``position`` advances after each source value is consumed, including values
    discarded by :meth:`skip`. Closing is idempotent and forwards to an owned
    iterator's ``close`` method when it has one. A closed cursor is exhausted.
    """

    def __init__(self, iterator: Iterator[T]) -> None:
        self._iterator = iterator
        self._position = 0
        self._closed = False

    def __iter__(self) -> DatasetCursor[T]:
        """Return this cursor as its own iterator."""

        return self

    def __next__(self) -> T:
        """Return the next value and advance position, closing at exhaustion."""

        if self._closed:
            raise StopIteration
        try:
            value = next(self._iterator)
        except StopIteration:
            self.close()
            raise
        self._position += 1
        return value

    @property
    def position(self) -> int:
        """Return the number of source yields this cursor has consumed."""

        return self._position

    @staticmethod
    def _validate_skip_count(n: int) -> None:
        """Reject skip counts that cannot express an exact nonnegative yield count."""

        if type(n) is not int:
            raise TypeError("skip count must be a nonnegative exact int.")
        if n < 0:
            raise ValueError("skip count must be non-negative.")

    def skip(self, n: int) -> None:
        """Consume exactly ``n`` values or report partial advancement.

        Args:
            n: Nonnegative exact integer number of yields to discard.

        Raises:
            TypeError: If ``n`` is not an exact integer.
            ValueError: If ``n`` is negative.
            DatasetExhaustedError: If fewer than ``n`` values remain. Its counts
                describe this call's requested and actual advancement.
        """

        self._validate_skip_count(n)
        advanced = 0
        while advanced < n:
            try:
                next(self)
            except StopIteration as error:
                raise DatasetExhaustedError(n, advanced) from error
            advanced += 1

    def close(self) -> None:
        """Close the owned iterator once; subsequent reads are exhausted."""

        if self._closed:
            return
        self._closed = True
        close = getattr(self._iterator, "close", None)
        if close is not None:
            close()


class Dataset(Object, Generic[T]):
    """Abstract re-iterable source of values with an optional element spec.

    Subclasses must implement :meth:`__iter__`; cardinality remains optional.
    ``spec`` describes one yielded element when known, and ``peek`` obtains one
    value from a fresh iterator without changing persistent dataset state.
    """

    def __init__(self, spec: SpecTree | None = None):
        super().__init__()
        self._spec = spec

    _stream_operator = "source"

    @property
    def spec(self) -> SpecTree:
        if self._spec is None:
            raise ValueError(f"{type(self).__name__} has no known spec.")
        return self._spec

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Return a fresh iterator over this dataset's elements.

        Returns:
            An iterator yielding values compatible with this Dataset's element
            specification when one is available.

        Raises:
            Implementations may raise source-specific access or decoding errors.
        """

    def iterator(self) -> DatasetCursor[T]:
        """Create an independent closeable cursor over a fresh iterator.

        Returns:
            A cursor starting at position zero and owning this traversal's
            underlying iterator.

        Side Effects:
            May acquire source-local iteration resources. The Dataset itself does
            not retain cursor position or resource ownership.
        """

        return DatasetCursor(iter(self))

    def method_graph(self):
        """Return an inert local stream graph for this Dataset pipeline.

        Returns:
            A :class:`dryml.methods.MethodGraph` whose :meth:`learn` prepares the
            qualified stream subset and whose :meth:`iterator` opens a fresh graph
            cursor.

        Raises:
            NotImplementedError: During planning if this pipeline includes an
                operator outside the qualified stream subset.

        Side Effects:
            Constructing the graph opens no source and retains no cursor state.
        """

        from dryml.methods import MethodGraph
        from dryml.methods.stream import StreamPlan

        return MethodGraph(self, stream_plan=StreamPlan(self))

    def peek(self) -> T:
        """
        Return one element from the dataset without mutating long-term dataset
        state, assuming the dataset is re-iterable.
        """
        it = self.iterator()
        try:
            return next(it)
        except StopIteration as e:
            raise ValueError("Cannot peek an empty dataset.") from e
        finally:
            it.close()

    def __len__(self) -> Cardinality:
        """
        Override in subclasses when cardinality is known.
        """
        raise NotImplementedError("Subclasses must implement their lengths")


class _MapCursor(DatasetCursor):
    """Cursor that owns Map selection and optional source-level skip delegation."""

    def __init__(self, dataset: Map) -> None:
        super().__init__(iter(()))
        self._dataset = dataset
        self._source_cursor: DatasetCursor | None = None
        self._implementation = None
        self._pending = _PENDING
        self._resolved = False

    def _resolve(self) -> None:
        """Perform Map's existing selection path without invoking a selected target."""

        if self._resolved:
            return
        try:
            implementation = self._dataset.method.find_implementation(
                input_spec=self._dataset.src.spec,
            )
        except ImplementationSelectionError as error:
            if (
                error.reason != "unknown_traits"
                or error.unknown_traits != ("backend",)
            ):
                raise
            self._source_cursor = self._dataset.src.iterator()
            try:
                first = next(self._source_cursor)
            except StopIteration:
                self._resolved = True
                return
            backends = discover_backends(first)
            if len(backends) != 1:
                if len(backends) > 1:
                    raise ImplementationSelectionError("conflict")
                raise error
            implementation = self._dataset.method.find_implementation(
                input_spec=self._dataset.src.spec,
                backend=next(iter(backends)),
            )
            self._pending = first
        else:
            self._source_cursor = self._dataset.src.iterator()
        self._implementation = implementation
        self._resolved = True

    def __next__(self):
        """Transform one source value using Map's previously resolved local call."""

        if self._closed:
            raise StopIteration
        self._resolve()
        if self._implementation is None:
            self.close()
            raise StopIteration
        if self._pending is _PENDING:
            assert self._source_cursor is not None
            try:
                item = next(self._source_cursor)
            except StopIteration:
                self.close()
                raise
        else:
            item = self._pending
            self._pending = _PENDING
        self._position += 1
        return self._implementation(item)

    def skip(self, n: int) -> None:
        """Skip mapped values, delegating only after safe selection and declaration."""

        self._validate_skip_count(n)
        if n == 0:
            return
        self._resolve()
        if not self._dataset.method.iteration_independent:
            advanced = 0
            while advanced < n:
                try:
                    next(self)
                except StopIteration as error:
                    raise DatasetExhaustedError(n, advanced) from error
                advanced += 1
            return
        if self._implementation is None:
            self.close()
            raise DatasetExhaustedError(n, 0)
        advanced = 0
        if self._pending is not _PENDING:
            self._pending = _PENDING
            self._position += 1
            advanced = 1
        remaining = n - advanced
        if remaining == 0:
            return
        assert self._source_cursor is not None
        try:
            self._source_cursor.skip(remaining)
        except DatasetExhaustedError as error:
            self._position += error.yielded
            self.close()
            raise DatasetExhaustedError(n, advanced + error.yielded) from error
        self._position += remaining

    def close(self) -> None:
        """Close the Map traversal and its owned source cursor."""

        if self._closed:
            return
        if self._source_cursor is not None:
            self._source_cursor.close()
        super().close()


class Map(Dataset):
    """Dataset node that applies one selected Method callable per source element.

    A complete source specification selects the callable before source
    consumption. When backend alone is unknown, the iterator contributes one
    value at most to complete that constraint. Other selection failures propagate
    without source consumption.
    """

    _stream_operator = "map"

    def __init__(self, src: Dataset, *methods):
        if not methods:
            raise ValueError("Map requires at least one Method.")

        if len(methods) == 1:
            method = methods[0]
        else:
            from dryml.data.methods import Pipe
            method = Pipe(*methods)

        self.src = src
        self.method = method
        super().__init__(spec=method.infer_output_spec(src.spec))

    def __iter__(self) -> Iterator:
        """Return an independent closeable Map cursor for ordinary iteration."""

        return self.iterator()

    def iterator(self) -> DatasetCursor:
        """Create a Map cursor with conservative capability-driven skipping.

        Returns:
            An independent cursor that preserves normal selection and transforms
            discarded values unless this Method explicitly declares iteration
            independence after selection resolves.
        """

        return _MapCursor(self)

    def __len__(self) -> Cardinality:
        return self.src.__len__()


class StreamDataset(Dataset):
    """Compose one bounded custom :class:`StreamNode` with Dataset sources.

    Args:
        *sources: Ordered Dataset inputs borrowed by the node implementation.

    Attributes:
        node: Class-level explicit deterministic stream declaration with bounded
            buffering.

    Iteration uses the Dataset MethodGraph path, so each traversal opens fresh
    source occurrences and the graph cursor owns all acquired resources.  The
    node's output spec is inferred without opening or consuming a source. Invalid
    declarations, source count, or source types raise ``TypeError``/``ValueError``
    before iteration.
    """

    _stream_operator = "custom"
    node = None

    def __init_subclass__(cls, **kwargs):
        """Mark each author subclass as an explicit custom stream participant."""

        super().__init_subclass__(**kwargs)
        cls._stream_operator = "custom"

    def __init__(self, *sources: Dataset):
        from dryml.methods.stream import StreamNode

        node = type(self).node
        if not isinstance(node, StreamNode):
            raise TypeError("StreamDataset subclasses must declare a StreamNode class attribute.")
        if len(sources) != len(node.inputs):
            raise ValueError("StreamDataset source count must match StreamNode inputs.")
        if not all(isinstance(source, Dataset) for source in sources):
            raise TypeError("StreamDataset sources must be Dataset instances.")
        self.node = node
        self.sources = tuple(sources)
        super().__init__(spec=node.output_spec(tuple(source.spec for source in self.sources)))

    def __iter__(self):
        """Return a fresh graph cursor for ordinary local iteration."""

        graph = self.method_graph()
        graph.learn()
        return graph.iterator()

    def iterator(self):
        """Return a fresh closeable graph cursor for this custom stream node."""

        return iter(self)

    def __len__(self) -> Cardinality:
        """Return the node-declared output cardinality without source traversal."""

        return self.node.output_cardinality(tuple(source.__len__() for source in self.sources))
