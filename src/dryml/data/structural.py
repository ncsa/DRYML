"""Structural Dataset operators with explicit child-traversal ownership."""

from __future__ import annotations

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, SpecTree, batch_spec_tree, unbatch_spec_tree
from dryml.data.collate import collate_for_spec, default_collate
from dryml.data.dataset import Dataset, DatasetCursor, DatasetExhaustedError
from dryml.data.split import default_split, split_for_spec


class Batch(Dataset):
    """Group source yields into owned, bounded batches.

    Args:
        src: Dataset supplying unbatched elements.
        batch_size: Positive number of elements per batch.
        drop_remainder: Discard the final short batch when true.

    Iteration returns a lazy closeable cursor. Source/collation failures close
    children while retaining the primary error; non-pull StopIteration is failure.
    Construction rejects nonpositive sizes with ValueError and opens no source.
    """

    _stream_operator = "batch"
    def __init__(self, src: Dataset, batch_size: int, *, drop_remainder: bool = False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.src = src
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        # A non-dropping final batch may be shorter than ``batch_size``. Preserve
        # a fixed size only when dropping that final partial batch is guaranteed.
        batch = batch_size if drop_remainder else Dynamic
        super().__init__(spec=batch_spec_tree(src.spec, batch=batch))

    def __iter__(self):
        """Return a lazy owned traversal, releasing children on every terminal path."""

        return DatasetCursor._from_generator(self._iterate)

    def _iterate(self, owner):
        it = owner._own(owner._call(self.src.iterator))
        first_batch = self._next_batch(it, owner)
        if not first_batch:
            return
        if self.drop_remainder and len(first_batch) < self.batch_size:
            return

        collate = owner._call(self._resolve_collate, first_batch)
        yield owner._call(collate, first_batch)

        while True:
            batch = self._next_batch(it, owner)
            if not batch:
                return
            if self.drop_remainder and len(batch) < self.batch_size:
                return
            yield owner._call(collate, batch)

    def __len__(self) -> Cardinality:
        src_cardinality = self.src.yield_cardinality()
        if src_cardinality.is_infinite:
            return Cardinality.INFINITE
        if src_cardinality.is_unknown:
            return Cardinality.UNKNOWN
        n = src_cardinality.require_finite()
        if self.drop_remainder:
            out = n // self.batch_size
        else:
            out = (n + self.batch_size - 1) // self.batch_size
        return Cardinality.finite(out)

    def _range_example_cardinality(
        self, start: int, stop: int | None = None,
    ) -> Cardinality:
        """Map batch-yield ranges to the exact contributing source-yield range."""

        outputs = self._range_yield_cardinality(start, stop)
        if outputs.is_finite and outputs.require_finite() == 0:
            return Cardinality.finite(0)
        if outputs.is_unknown:
            return Cardinality.UNKNOWN
        source_start = start * self.batch_size
        if self.drop_remainder:
            source_stop = None if outputs.is_infinite else source_start + outputs.require_finite() * self.batch_size
        else:
            source_stop = None if stop is None else stop * self.batch_size
        return self.src._range_example_cardinality(source_start, source_stop)

    def _next_batch(self, it, owner):
        batch = []
        for _ in range(self.batch_size):
            try:
                batch.append(owner._pull(it))
            except StopIteration:
                break
        return batch

    def _resolve_collate(self, first_batch):
        """Select one declared-backend collator, retaining generic fallback."""

        del first_batch
        return collate_for_spec(self.src.spec) or default_collate


class Unbatch(Dataset):
    """Split a batched Dataset into an owned traversal of individual elements.

    Args:
        src: Dataset with a batched element specification.

    Iteration returns a lazy closeable cursor owning its source and split
    iterators. Source/split/cleanup errors propagate, retaining any primary
    failure. Construction infers output specs without opening a traversal.
    """

    _stream_operator = "unbatch"
    def __init__(self, src: Dataset):
        self.src = src
        super().__init__(spec=unbatch_spec_tree(src.spec))

    def __iter__(self):
        """Return a lazy traversal owning both its source and split iterators."""

        return DatasetCursor._from_generator(self._iterate)

    def _iterate(self, owner):
        split = owner._call(self._resolve_split)
        source = owner._own(owner._call(self.src.iterator))
        while True:
            try:
                batch = owner._pull(source)
            except StopIteration:
                return
            values = owner._own(owner._call(iter, owner._call(split, batch)))
            while True:
                try:
                    value = owner._pull(values)
                except StopIteration:
                    break
                yield value
            owner._release(values)

    def __len__(self) -> Cardinality:
        return self.src.example_cardinality()

    def _range_example_cardinality(
        self, start: int, stop: int | None = None,
    ) -> Cardinality:
        """Return unbatched output facts from this node's proved output range."""

        return self._range_yield_cardinality(start, stop)

    def _resolve_split(self):
        """Return the selected split invoker once per qualified graph cursor."""

        return split_for_spec(self.src.spec) or default_split


class Take(Dataset):
    """Yield exactly a requested number of source values or report exhaustion.

    Each lazy cursor owns its child. Shortfall retains requested/yielded counts,
    remains primary over cleanup errors, and never certifies root EOF.
    """

    _stream_operator = "take"

    def __init__(
        self, src: Dataset, n: int, *, epoch: int = 0, fixed_prefix: bool = False,
    ):
        """Require a bounded source prefix, optionally selecting a logical epoch.

        Args:
            src: Source Dataset.
            n: Exact nonnegative number of source yields required per traversal.
            epoch: Exact nonnegative base epoch for a supporting seed-aware source.
            fixed_prefix: Keep the base epoch during repeated traversals instead of
                advancing logical epochs.

        Seed-aware selection never changes Take's strict yield count. Opaque
        sources retain ordinary iteration behavior.
        """

        if type(n) is not int:
            raise TypeError("n must be a nonnegative exact int.")
        if n < 0:
            raise ValueError("n must be non-negative.")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative exact int.")
        if type(fixed_prefix) is not bool:
            raise TypeError("fixed_prefix must be an exact bool.")
        self.src = src
        self.n = n
        self.epoch = epoch
        self.fixed_prefix = fixed_prefix
        super().__init__(spec=src.spec)

    def __iter__(self):
        """Return an independent strict Take cursor for ordinary iteration."""

        return self.iterator()

    def iterator(self) -> DatasetCursor:
        """Create a lazy closeable cursor that owns at most ``n`` source yields.

        Returns:
            A cursor reporting source exhaustion with this Take's requested and
            observed counts. ``Take(0)`` creates no source iterator.
        """

        return self.iterator_for_epoch(0)

    def iterator_for_epoch(self, logical_epoch: int) -> DatasetCursor:
        """Create one strict cursor for a logical repeat epoch without replay."""

        if type(logical_epoch) is not int or logical_epoch < 0:
            raise ValueError("logical_epoch must be a nonnegative exact int.")
        epoch = self.epoch if self.fixed_prefix else self.epoch + logical_epoch
        return _TakeCursor(self.src, self.n, epoch=epoch)

    def __len__(self) -> Cardinality:
        return Cardinality.finite(self.n)

    def example_cardinality(self) -> Cardinality:
        """Return proved examples in this strict requested yield prefix."""

        return self._range_example_cardinality(0, self.n)

    def _range_example_cardinality(
        self, start: int, stop: int | None = None,
    ) -> Cardinality:
        """Delegate bounded output ranges to corresponding source yield ranges."""

        own = self._range_yield_cardinality(start, stop)
        if own.is_finite and own.require_finite() == 0:
            return Cardinality.finite(0)
        return self.src._range_example_cardinality(start, self.n if stop is None else min(stop, self.n))


class _TakeCursor(DatasetCursor):
    """Lazy strict Take cursor that closes its source on every terminal path."""

    def __init__(self, src: Dataset, n: int, *, epoch: int) -> None:
        super().__init__(iter(()))
        self._src = src
        self._n = n
        self._epoch = epoch
        self._source_cursor: DatasetCursor | None = None

    def _read(self):
        if self._position >= self._n:
            raise StopIteration
        if self._source_cursor is None:
            opener = self._call(getattr, self._src, "iterator_for_epoch", None)
            self._source_cursor = self._own(
                self._call(opener, self._epoch) if callable(opener) else self._call(self._src.iterator)
            )
        try:
            value = self._pull(self._source_cursor)
        except (StopIteration, DatasetExhaustedError) as error:
            raise DatasetExhaustedError(self._n, self._position) from error
        self._position += 1
        return value


class Skip(Dataset):
    """Discard a forgiving prefix, then yield the remaining owned source values.

    Args:
        src: Dataset to traverse lazily.
        n: Exact nonnegative number of yields to discard.

    TypeError/ValueError reject malformed counts. Iteration returns a closeable
    cursor and propagates source/cleanup failures while retaining primary errors.
    A source ending within the discarded prefix produces empty output normally.
    """

    def __init__(self, src: Dataset, n: int):
        if type(n) is not int:
            raise TypeError("n must be a nonnegative exact int.")
        if n < 0:
            raise ValueError("n must be non-negative.")
        self.src = src
        self.n = n
        super().__init__(spec=src.spec)

    def __iter__(self):
        """Return an owned traversal; a short discarded prefix remains forgiving."""

        return DatasetCursor._from_generator(self._iterate)

    def _iterate(self, owner):
        it = owner._own(owner._call(self.src.iterator))
        for _ in range(self.n):
            try:
                owner._pull(it)
            except StopIteration:
                return
        while True:
            try:
                value = owner._pull(it)
            except StopIteration:
                return
            yield value

    def __len__(self) -> Cardinality:
        return self.src._range_yield_cardinality(self.n)

    def _range_example_cardinality(
        self, start: int, stop: int | None = None,
    ) -> Cardinality:
        """Delegate output ranges after this node's skipped source prefix."""

        own = self._range_yield_cardinality(start, stop)
        if own.is_finite and own.require_finite() == 0:
            return Cardinality.finite(0)
        return self.src._range_example_cardinality(
            self.n + start, None if stop is None else self.n + stop,
        )


class Repeat(Dataset):
    """Reopen an independently owned source traversal for each logical epoch.

    Args:
        src: Re-iterable Dataset, optionally supporting epoch selection.
        count: Nonnegative repetition count, or None for unbounded repetition.

    Iteration returns a lazy closeable cursor; empty unbounded sources terminate
    normally. Each completed epoch releases its resources before reopening the
    source. Source/cleanup failures propagate and preserve any primary failure.
    Negative counts raise ValueError without source acquisition.
    """

    def __init__(self, src: Dataset, count: int | None = None):
        if count is not None and count < 0:
            raise ValueError("count must be non-negative or None.")
        self.src = src
        self.count = count
        super().__init__(spec=src.spec)

    def __iter__(self):
        """Return a lazy cursor reopening and releasing one source per epoch."""

        return DatasetCursor._from_generator(self._iterate)

    def _iterate(self, owner):
        epoch = 0
        while self.count is None or epoch < self.count:
            opener = getattr(self.src, "iterator_for_epoch", None)
            iterator = owner._own(owner._call(opener, epoch) if callable(opener) else owner._call(self.src.iterator))
            produced = False
            while True:
                try:
                    value = owner._pull(iterator)
                except StopIteration:
                    break
                produced = True
                yield value
            epoch += 1
            if (self.count is None and not produced) or epoch == self.count:
                return
            owner._release(iterator)
            if owner._deferred_cleanup:
                raise owner._deferred_cleanup.pop(0)

    def __len__(self) -> Cardinality:
        src_cardinality = self.src.yield_cardinality()
        if self.count is None:
            if src_cardinality.is_finite and src_cardinality.require_finite() == 0:
                return Cardinality.finite(0)
            return Cardinality.UNKNOWN if src_cardinality.is_unknown else Cardinality.INFINITE
        if src_cardinality.is_unknown:
            return Cardinality.UNKNOWN
        if src_cardinality.is_infinite:
            return Cardinality.INFINITE if self.count > 0 else Cardinality.finite(0)
        return Cardinality.finite(src_cardinality.require_finite() * self.count)

    def example_cardinality(self) -> Cardinality:
        """Return multiplied source facts when Repeat's count proves them."""

        source = self.src.example_cardinality()
        if self.count == 0 or (source.is_finite and source.require_finite() == 0):
            return Cardinality.finite(0)
        if self.count is None:
            return Cardinality.UNKNOWN if source.is_unknown else Cardinality.INFINITE
        if source.is_unknown:
            return Cardinality.UNKNOWN
        if source.is_infinite:
            return Cardinality.INFINITE
        return Cardinality.finite(source.require_finite() * self.count)


class Shuffle(Dataset):
    """Shuffle yields through a bounded buffer while owning the source traversal.

    Args:
        src: Dataset traversed lazily in authored order before buffering.
        buffer_size: Positive maximum number of buffered yields.
        seed: Optional NumPy random-generator seed.

    Iteration returns a closeable cursor; early close/failure releases the source.
    Source/cleanup errors preserve the primary failure. Construction rejects
    nonpositive buffers with ValueError and acquires no traversal resources.
    """

    def __init__(self, src: Dataset, buffer_size: int, *, seed=None):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self.src = src
        self.buffer_size = buffer_size
        self.seed = seed
        super().__init__(spec=src.spec)

    def __iter__(self):
        """Return a bounded-buffer traversal owning its source until termination."""

        return DatasetCursor._from_generator(self._iterate)

    def _iterate(self, owner):
        import numpy as np

        rng = np.random.default_rng(seed=self.seed)
        it = owner._own(owner._call(self.src.iterator))
        buffer = []

        for _ in range(self.buffer_size):
            try:
                buffer.append(owner._pull(it))
            except StopIteration:
                break

        while buffer:
            idx = int(rng.integers(0, len(buffer))) if len(buffer) > 1 else 0
            yield buffer.pop(idx)
            try:
                buffer.append(owner._pull(it))
            except StopIteration:
                pass

    def __len__(self) -> Cardinality:
        return self.src.yield_cardinality()

    def example_cardinality(self) -> Cardinality:
        """Preserve full membership facts while withholding shuffled prefix facts."""

        return self.src.example_cardinality()

    def _range_example_cardinality(
        self, start: int, stop: int | None = None,
    ) -> Cardinality:
        """Return a fact only for whole shuffled membership or an empty range."""

        yields = self._range_yield_cardinality(start, stop)
        if yields.is_finite and yields.require_finite() == 0:
            return Cardinality.finite(0)
        if start == 0 and stop is None:
            return self.src.example_cardinality()
        return Cardinality.UNKNOWN


__all__ = ["Batch", "Repeat", "Shuffle", "Skip", "Take", "Unbatch"]
