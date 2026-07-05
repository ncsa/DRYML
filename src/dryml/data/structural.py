from __future__ import annotations

import itertools

from dryml.core2.cardinality import Cardinality
from dryml.core2.tensor_spec import SpecTree, batch_from_spec_tree, batch_spec_tree, unbatch_spec_tree
from dryml.data.collate import default_collate
from dryml.data.dataset import Dataset
from dryml.data.split import default_split


def _as_cardinality(value):
    if isinstance(value, Cardinality):
        return value
    return Cardinality.finite(int(value))


def _dataset_cardinality(dataset: Dataset) -> Cardinality:
    try:
        return _as_cardinality(dataset.__len__())
    except NotImplementedError:
        return Cardinality.UNKNOWN


class Batch(Dataset):
    def __init__(self, src: Dataset, batch_size: int, *, drop_remainder: bool = False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.src = src
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        super().__init__(spec=batch_spec_tree(src.spec, batch=batch_size))

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
        src_cardinality = _dataset_cardinality(self.src)
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
        if isinstance(self.src, Batch):
            source_cardinality = _dataset_cardinality(self.src.src)
            if source_cardinality.is_unknown:
                return Cardinality.UNKNOWN
            if source_cardinality.is_infinite:
                return Cardinality.INFINITE

            n = source_cardinality.require_finite()
            if self.src.drop_remainder:
                n = (n // self.src.batch_size) * self.src.batch_size
            return Cardinality.finite(n)

        src_cardinality = _dataset_cardinality(self.src)
        if src_cardinality.is_infinite:
            batch = batch_from_spec_tree(self.src.spec)
            if isinstance(batch, int) and batch > 0:
                return Cardinality.INFINITE
        return Cardinality.UNKNOWN


class Take(Dataset):
    def __init__(self, src: Dataset, n: int):
        if n < 0:
            raise ValueError("n must be non-negative.")
        self.src = src
        self.n = n
        super().__init__(spec=src.spec)

    def __iter__(self):
        yield from itertools.islice(iter(self.src), self.n)

    def __len__(self) -> Cardinality:
        src_cardinality = _dataset_cardinality(self.src)
        if src_cardinality.is_unknown:
            return Cardinality.finite(self.n)
        if src_cardinality.is_infinite:
            return Cardinality.finite(self.n)
        return Cardinality.finite(min(self.n, src_cardinality.require_finite()))


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
        src_cardinality = _dataset_cardinality(self.src)
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
        src_cardinality = _dataset_cardinality(self.src)
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
        return _dataset_cardinality(self.src)


__all__ = ["Batch", "Repeat", "Shuffle", "Skip", "Take", "Unbatch"]
