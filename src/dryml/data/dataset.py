from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from numbers import Integral
from typing import Generic, TypeVar

from dryml.core import Object
from dryml.core.backend import discover_backends
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, SpecTree, TensorSpec
from dryml.methods import ImplementationSelectionError


T = TypeVar("T")
_PENDING = object()


def _normalize_cardinality(value: int | Cardinality) -> Cardinality:
    """Normalize a legacy Dataset length declaration without coercing malformed values."""

    if isinstance(value, Cardinality):
        if value.is_finite:
            value = value.require_finite()
        else:
            return value
    if type(value) is not int:
        raise TypeError("Dataset cardinality must be a Cardinality or exact nonnegative int.")
    if value < 0:
        raise ValueError("Dataset cardinality must be non-negative.")
    return Cardinality.finite(value)


def _spec_tensor_leaves(spec, *, strict: bool = False) -> list[TensorSpec]:
    """Return TensorSpec leaves while distinguishing generic from malformed trees."""

    if isinstance(spec, TensorSpec):
        return [spec]
    if isinstance(spec, dict):
        leaves = []
        for child in spec.values():
            leaves.extend(_spec_tensor_leaves(child, strict=strict))
        return leaves
    if isinstance(spec, (tuple, list)):
        leaves = []
        for child in spec:
            leaves.extend(_spec_tensor_leaves(child, strict=strict))
        return leaves
    if strict:
        raise TypeError("Dataset example counting requires a TensorSpec tree.")
    return []


def _batch_length(value, path: tuple[object, ...]) -> int:
    """Return one runtime leading dimension without importing a tensor backend."""

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            if len(shape) == 0:
                raise ValueError(f"Batched value at {path} is rank-0.")
            length = shape[0]
        except TypeError as error:
            raise TypeError(f"Batched value at {path} has no readable shape.") from error
    else:
        try:
            length = len(value)
        except TypeError as error:
            raise TypeError(
                f"Batched value at {path} has no leading example dimension."
            ) from error
    if isinstance(length, bool) or not isinstance(length, Integral):
        raise TypeError(f"Batched value at {path} has a non-integral leading dimension.")
    length = int(length)
    if length < 0:
        raise ValueError(f"Batched value at {path} has a negative leading dimension.")
    return length


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

    def yield_cardinality(self) -> Cardinality:
        """Return the declared number of values yielded by this Dataset.

        Returns:
            A finite, unknown, or infinite :class:`~dryml.core.cardinality.Cardinality`.
            Legacy exact integer ``__len__`` declarations are normalized to finite
            cardinality.

        Raises:
            TypeError: If a legacy declaration is not a Cardinality or exact integer.
            ValueError: If a finite legacy declaration is negative.

        Side Effects:
            None. This method never opens, peeks, or scans a Dataset iterator.
        """

        try:
            declared = self.__len__()
        except NotImplementedError:
            return Cardinality.UNKNOWN
        return _normalize_cardinality(declared)

    def _range_yield_cardinality(self, start: int, stop: int | None = None) -> Cardinality:
        """Return exact cardinality for a yield range when declarations prove it.

        This private seam accepts exact nonnegative yield offsets. Finite sources
        clip range endpoints to their declared extent; unknown sources remain
        unknown unless the requested range is empty. Consumers requiring strict
        exhaustion, such as :class:`Take`, must not use finite-source clipping to
        weaken their requested-yield contract.
        """

        if type(start) is not int:
            raise TypeError("range start must be an exact nonnegative int.")
        if start < 0:
            raise ValueError("range start must be a nonnegative exact int.")
        if stop is not None and type(stop) is not int:
            raise TypeError("range stop must be an exact nonnegative int or None.")
        if stop is not None and stop < 0:
            raise ValueError("range stop must be a nonnegative exact int or None.")
        if stop is not None and stop < start:
            raise ValueError("range stop must not precede range start.")

        cardinality = self.yield_cardinality()
        if stop is not None and stop == start:
            return Cardinality.finite(0)
        if cardinality.is_finite:
            total = cardinality.require_finite()
            clipped_start = min(start, total)
            clipped_stop = total if stop is None else min(stop, total)
            return Cardinality.finite(max(0, clipped_stop - clipped_start))
        if cardinality.is_infinite:
            if stop is None:
                return Cardinality.INFINITE
            return Cardinality.finite(stop - start)
        return Cardinality.UNKNOWN

    def example_cardinality(self) -> Cardinality:
        """Return a provable total number of examples represented by Dataset yields.

        Returns:
            A finite, unknown, or infinite Cardinality. Unbatched TensorSpec trees
            have one example per yield. Uniform fixed batch declarations multiply
            yield cardinality; dynamic batch declarations remain unknown. Zero
            yields always prove zero examples.

        Raises:
            TypeError: If a TensorSpec tree mixes with unsupported declarations.
            ValueError: If declared fixed batch dimensions conflict or are zero for
            a Dataset with a nonzero declared yield count.

        Side Effects:
            None. This method reads only declared cardinality and spec metadata;
            it never opens, peeks, or scans a Dataset iterator.
        """

        return self._range_example_cardinality(0)

    def _range_example_cardinality(
        self, start: int, stop: int | None = None,
    ) -> Cardinality:
        """Return a proved example total for a yield range without iteration.

        Args:
            start: Exact nonnegative first yield offset.
            stop: Optional exact nonnegative exclusive yield offset.

        Returns:
            The exact example cardinality when TensorSpec metadata gives every
            selected yield a uniform example count; otherwise ``UNKNOWN``.

        Raises:
            TypeError: If range endpoints or a countable spec declaration are malformed.
            ValueError: If endpoints are negative/reversed or batches disagree.

        Side Effects:
            None. This private proof seam only reads declared metadata.
        """

        yields = self._range_yield_cardinality(start, stop)
        if yields.is_finite and yields.require_finite() == 0:
            return Cardinality.finite(0)

        spec = self._spec if hasattr(self, "_spec") else self.spec
        if spec is None:
            return Cardinality.UNKNOWN

        specs = _spec_tensor_leaves(spec)
        if not specs:
            return Cardinality.UNKNOWN
        _spec_tensor_leaves(spec, strict=True)

        batched = [spec.batch for spec in specs if spec.batched]
        if not batched:
            return yields
        if len(batched) != len(specs):
            raise ValueError("Dataset TensorSpecs must be uniformly batched or unbatched for counting.")

        fixed = {batch for batch in batched if batch is not Dynamic}
        if len(fixed) > 1:
            raise ValueError("Dataset fixed batch declarations disagree.")
        if Dynamic in batched:
            return Cardinality.UNKNOWN

        batch_size = next(iter(fixed))
        assert type(batch_size) is int
        if batch_size == 0:
            raise ValueError("A nonempty Dataset cannot declare zero-example batches.")
        if yields.is_unknown:
            return Cardinality.UNKNOWN
        if yields.is_infinite:
            return Cardinality.INFINITE
        return Cardinality.finite(yields.require_finite() * batch_size)

    def examples_in(self, value: T) -> int:
        """Validate and return the exact examples represented by one yielded value.

        Args:
            value: A runtime value matching this Dataset's declared TensorSpec tree.

        Returns:
            The positive number of examples in a uniformly batched value, or ``1``
            for a coherent unbatched TensorSpec tree.

        Raises:
            TypeError: If the declaration/value structures are unsupported or differ.
            ValueError: If batch declarations, leading dimensions, or fixed batch
            sizes disagree; if a batch is empty; or if a batched value is rank zero.

        Side Effects:
            Inspects only ``value`` and declared specs. It never opens a Dataset
            cursor, runs a model, or imports an optional tensor framework.
        """

        def count(spec, runtime, path: tuple[object, ...]) -> list[int | None]:
            if isinstance(spec, TensorSpec):
                if not spec.batched:
                    return [None]
                observed = _batch_length(runtime, path)
                if observed == 0:
                    raise ValueError(f"Batched value at {path} has zero examples.")
                if spec.batch is not Dynamic and observed != spec.batch:
                    raise ValueError(
                        f"Batched value at {path} has {observed} examples, but its "
                        f"declared batch size is {spec.batch}."
                    )
                return [observed]
            if isinstance(spec, dict):
                if not isinstance(runtime, dict):
                    raise TypeError(f"Value at {path} must be a dict.")
                if tuple(runtime.keys()) != tuple(spec.keys()):
                    raise ValueError(f"Value at {path} has keys that differ from its spec.")
                return [item for key in spec for item in count(spec[key], runtime[key], path + (key,))]
            if isinstance(spec, tuple):
                if not isinstance(runtime, tuple) or len(runtime) != len(spec):
                    raise TypeError(f"Value at {path} must be a tuple matching its spec.")
                return [item for index in range(len(spec)) for item in count(spec[index], runtime[index], path + (index,))]
            if isinstance(spec, list):
                if not isinstance(runtime, list) or len(runtime) != len(spec):
                    raise TypeError(f"Value at {path} must be a list matching its spec.")
                return [item for index in range(len(spec)) for item in count(spec[index], runtime[index], path + (index,))]
            raise TypeError("Dataset example counting requires a TensorSpec tree.")

        counts = count(self.spec, value, ())
        if not counts:
            raise TypeError("Dataset example counting requires at least one TensorSpec leaf.")
        batched = [count for count in counts if count is not None]
        if not batched:
            return 1
        if len(batched) != len(counts):
            raise ValueError("Dataset TensorSpecs must be uniformly batched or unbatched for counting.")
        if len(set(batched)) != 1:
            raise ValueError("Batched TensorSpec leaves disagree on their example count.")
        return batched[0]

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
        result = self._implementation(item)
        if getattr(self._dataset, "preserves_examples", False):
            source_examples = self._dataset.src.examples_in(item)
            if self._dataset.examples_in(result) != source_examples:
                raise ValueError("Map declared preserves_examples but changed an example count.")
        self._position += 1
        return result

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

    def __init__(self, src: Dataset, *methods, preserves_examples: bool = False):
        """Construct a one-to-one Dataset transform.

        Args:
            src: Source Dataset supplying one input value per Method invocation.
            *methods: One Method, or sequential Methods composed into a Pipe.
            preserves_examples: Declare and validate that every output has the
                same runtime example count as its input. Defaults to ``False``.

        Raises:
            TypeError: If ``preserves_examples`` is not an exact bool.
            ValueError: If no Method is supplied.

        Side Effects:
            Construction does not open a source; iteration selects Methods lazily.
        """

        if not methods:
            raise ValueError("Map requires at least one Method.")
        if type(preserves_examples) is not bool:
            raise TypeError("preserves_examples must be an exact bool.")

        if len(methods) == 1:
            method = methods[0]
        else:
            from dryml.data.methods import Pipe
            method = Pipe(*methods)

        self.src = src
        self.method = method
        self.preserves_examples = preserves_examples
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
        return self.src.yield_cardinality()

    def _range_example_cardinality(
        self, start: int, stop: int | None = None,
    ) -> Cardinality:
        """Propagate range facts only for an explicit preserving Map declaration."""

        yields = self._range_yield_cardinality(start, stop)
        if yields.is_finite and yields.require_finite() == 0:
            return Cardinality.finite(0)
        # Older saved Map definitions have no field and deliberately default to
        # the conservative non-preserving behavior.
        if not getattr(self, "preserves_examples", False):
            return Cardinality.UNKNOWN
        return self.src._range_example_cardinality(start, stop)


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

        return self.node.output_cardinality(
            tuple(source.yield_cardinality() for source in self.sources)
        )
