from __future__ import annotations

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, SpecTree, batch_spec_tree, unbatch_spec_tree
from dryml.data.collate import default_collate
from dryml.data.dataset import Dataset, DatasetCursor, DatasetExhaustedError
from dryml.data.split import default_split


def _as_cardinality(value):
    if isinstance(value, Cardinality):
        return value
    return Cardinality.finite(int(value))


class Batch(Dataset):
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
        it = iter(self.src)
        first_batch = self._next_batch(it)
        if not first_batch:
            return
        if self.drop_remainder and len(first_batch) < self.batch_size:
            return

        collate = self._resolve_collate(first_batch)
        yield collate(first_batch)

        while True:
            batch = self._next_batch(it)
            if not batch:
                return
            if self.drop_remainder and len(batch) < self.batch_size:
                return
            yield collate(batch)

    def __len__(self) -> Cardinality:
        src_cardinality = _as_cardinality(self.src.__len__())
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

    def _next_batch(self, it):
        batch = []
        for _ in range(self.batch_size):
            try:
                batch.append(next(it))
            except StopIteration:
                break
        return batch

    def _resolve_collate(self, first_batch):
        return default_collate


class Unbatch(Dataset):
    def __init__(self, src: Dataset):
        self.src = src
        super().__init__(spec=unbatch_spec_tree(src.spec))

    def __iter__(self):
        for batch in self.src:
            yield from default_split(batch)

    def __len__(self) -> Cardinality:
        return Cardinality.UNKNOWN


class Take(Dataset):
    """Yield exactly a requested number of source values or report exhaustion."""

    def __init__(self, src: Dataset, n: int):
        if type(n) is not int:
            raise TypeError("n must be a nonnegative exact int.")
        if n < 0:
            raise ValueError("n must be non-negative.")
        self.src = src
        self.n = n
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

        return _TakeCursor(self.src, self.n)

    def __len__(self) -> Cardinality:
        return Cardinality.finite(self.n)


class _TakeCursor(DatasetCursor):
    """Lazy strict Take cursor that closes its source on every terminal path."""

    def __init__(self, src: Dataset, n: int) -> None:
        super().__init__(iter(()))
        self._src = src
        self._n = n
        self._source_cursor: DatasetCursor | None = None

    def __next__(self):
        """Return one required source value or raise Take's counted exhaustion error."""

        if self._closed or self._position >= self._n:
            self.close()
            raise StopIteration
        if self._source_cursor is None:
            self._source_cursor = self._src.iterator()
        try:
            value = next(self._source_cursor)
        except (StopIteration, DatasetExhaustedError) as error:
            self.close()
            raise DatasetExhaustedError(self._n, self._position) from error
        except BaseException:
            self.close()
            raise
        self._position += 1
        return value

    def close(self) -> None:
        """Close this cursor and any lazily acquired source cursor once."""

        if self._closed:
            return
        if self._source_cursor is not None:
            self._source_cursor.close()
        super().close()


class Skip(Dataset):
    def __init__(self, src: Dataset, n: int):
        if n < 0:
            raise ValueError("n must be non-negative.")
        self.src = src
        self.n = n
        super().__init__(spec=src.spec)

    def __iter__(self):
        it = iter(self.src)
        for _ in range(self.n):
            try:
                next(it)
            except StopIteration:
                return
        yield from it

    def __len__(self) -> Cardinality:
        src_cardinality = _as_cardinality(self.src.__len__())
        if src_cardinality.is_unknown:
            return Cardinality.UNKNOWN
        if src_cardinality.is_infinite:
            return Cardinality.INFINITE
        return Cardinality.finite(max(0, src_cardinality.require_finite() - self.n))


class Repeat(Dataset):
    def __init__(self, src: Dataset, count: int | None = None):
        if count is not None and count < 0:
            raise ValueError("count must be non-negative or None.")
        self.src = src
        self.count = count
        super().__init__(spec=src.spec)

    def __iter__(self):
        if self.count is None:
            while True:
                yield from self.src
        else:
            for _ in range(self.count):
                yield from self.src

    def __len__(self) -> Cardinality:
        if self.count is None:
            return Cardinality.INFINITE
        src_cardinality = _as_cardinality(self.src.__len__())
        if src_cardinality.is_unknown:
            return Cardinality.UNKNOWN
        if src_cardinality.is_infinite:
            return Cardinality.INFINITE if self.count > 0 else Cardinality.finite(0)
        return Cardinality.finite(src_cardinality.require_finite() * self.count)


class Shuffle(Dataset):
    def __init__(self, src: Dataset, buffer_size: int, *, seed=None):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self.src = src
        self.buffer_size = buffer_size
        self.seed = seed
        super().__init__(spec=src.spec)

    def __iter__(self):
        import numpy as np

        rng = np.random.default_rng(seed=self.seed)
        it = iter(self.src)
        buffer = []

        for _ in range(self.buffer_size):
            try:
                buffer.append(next(it))
            except StopIteration:
                break

        while buffer:
            idx = int(rng.integers(0, len(buffer))) if len(buffer) > 1 else 0
            yield buffer.pop(idx)
            try:
                buffer.append(next(it))
            except StopIteration:
                pass

    def __len__(self) -> Cardinality:
        return _as_cardinality(self.src.__len__())


__all__ = ["Batch", "Repeat", "Shuffle", "Skip", "Take", "Unbatch"]
