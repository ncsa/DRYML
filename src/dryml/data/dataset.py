"""Re-iterable Dataset APIs and owned cursors with independent EOF evidence."""

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
from dryml.methods.stream import DatasetExhaustedError, _CursorLifecycle


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


class DatasetCursor(_CursorLifecycle, Iterator[T], Generic[T]):
    """Own one closeable Dataset traversal with a consumed-yield position.

    Args:
        iterator: Fresh iterator owned by this cursor.

    ``position`` advances after each source value is consumed, including values
    discarded by :meth:`skip`. Closing is idempotent and forwards to an owned
    iterator's ``close`` method when it has one. A closed cursor stops yielding,
    but only ``observed_eof`` proves successful root-output termination. Use a
    context manager to preserve consumer failures when cleanup also fails.
    Read failures close the source; cleanup failures propagate without erasing
    genuine EOF. Non-pull StopIteration becomes RuntimeError with its cause.
    """

    def __init__(self, iterator: Iterator[T]) -> None:
        super().__init__()
        self._iterator = self._own(iterator)
        self._position = 0

    @classmethod
    def _from_generator(cls, factory):
        """Give an ordinary operator generator explicit ownership of child sources."""

        cursor = cls(iter(()))
        cursor._iterator = cursor._own(cursor._call(factory, cursor))
        return cursor

    def __iter__(self) -> DatasetCursor[T]:
        """Return this cursor as its own iterator."""

        return self

    def _read(self) -> T:
        value = self._pull(self._iterator)
        self._position += 1
        return value

    @property
    def position(self) -> int:
        """Return the number of source yields this cursor has consumed."""

        return self._position


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

        Raises:
            BaseException: Source acquisition failure. Unexpected acquisition
                StopIteration is translated to RuntimeError with its cause.

        Side Effects:
            May acquire source-local iteration resources. The Dataset itself does
            not retain cursor position or resource ownership.
        """

        return DatasetCursor(_CursorLifecycle._call(iter, self))

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

    def prepare(self):
        """Plan this Dataset once for native training-data delivery.

        Returns:
            A dependency-light :class:`dryml.data.native.PreparedDataset` that
            reports whether qualified stream execution or eager fallback applies.

        Raises:
            ImplementationSelectionError: If a qualified Method cannot select a
            declared local implementation. Unqualified Dataset operators instead
            retain documented eager iteration.

        Side Effects:
            Records qualified local stream selections without opening or consuming
            a source. Each later prepared cursor owns an independent traversal.
        """

        from dryml.data.native import PreparedDataset

        return PreparedDataset(self)

    def peek(self) -> T:
        """Return one fresh element, releasing its traversal before returning.

        Raises:
            ValueError: If the Dataset is empty.
            BaseException: Source or cleanup failure; a source failure remains
                primary when cleanup also fails.
        """
        with self.iterator() as it:
            try:
                return next(it)
            except StopIteration as e:
                raise ValueError("Cannot peek an empty dataset.") from e

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
            implementation = self._call(self._dataset.method.find_implementation,
                input_spec=self._dataset.src.spec,
            )
        except ImplementationSelectionError as error:
            if (
                error.reason != "unknown_traits"
                or error.unknown_traits != ("backend",)
            ):
                raise
            self._source_cursor = self._own(self._call(self._dataset.src.iterator))
            try:
                first = self._pull(self._source_cursor)
            except StopIteration:
                self._resolved = True
                return
            backends = self._call(discover_backends, first)
            if len(backends) != 1:
                if len(backends) > 1:
                    raise ImplementationSelectionError("conflict")
                raise error
            implementation = self._call(self._dataset.method.find_implementation,
                input_spec=self._dataset.src.spec,
                backend=next(iter(backends)),
            )
            self._pending = first
        else:
            self._source_cursor = self._own(self._call(self._dataset.src.iterator))
        self._implementation = self._call(implementation.prepared_invoker)
        self._resolved = True

    def _read(self):
        self._call(self._resolve)
        if self._implementation is None:
            raise StopIteration
        if self._pending is _PENDING:
            assert self._source_cursor is not None
            item = self._pull(self._source_cursor)
        else:
            item = self._pending
            self._pending = _PENDING
        result = self._call(self._implementation, item)
        if getattr(self._dataset, "preserves_examples", False):
            source_examples = self._call(self._dataset.src.examples_in, item)
            if self._call(self._dataset.examples_in, result) != source_examples:
                raise ValueError("Map declared preserves_examples but changed an example count.")
        self._position += 1
        return result

    def _skip(self, n: int) -> None:
        """Delegate only iteration-independent skips within the shared boundary."""

        self._resolve()
        if not self._dataset.method.iteration_independent:
            return super()._skip(n)
        if self._implementation is None:
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
            self._call(self._source_cursor.skip, remaining)
        except DatasetExhaustedError as error:
            self._position += error.yielded
            raise DatasetExhaustedError(n, advanced + error.yielded) from error
        self._position += remaining


class Map(Dataset):
    """Dataset node that applies one selected Method callable per source element.

    Each cursor retains a prepared invoker once after selection; advancing values
    never recompiles the selected signature or changes child Method state.

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
