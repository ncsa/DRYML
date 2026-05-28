from __future__ import annotations

from dryml.core2.cardinality import Cardinality
import itertools

from dryml.core2.tensor_spec import SpecTree, batch_spec_tree, unbatch_spec_tree
from dryml.data.collate import default_collate
from dryml.data.split import default_split
from dryml.data.dataset import StructuralDataset

from .base import StructuralTransform


class BatchTransform(StructuralTransform):
    def __init__(self, batch_size: int, *, drop_remainder: bool = False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return batch_spec_tree(input_spec, batch=self.batch_size)

    def infer_cardinality(self, src_cardinality):
        if isinstance(src_cardinality, Cardinality):
            if src_cardinality.is_infinite:
                return Cardinality.INFINITE
            if src_cardinality.is_unknown:
                return Cardinality.UNKNOWN
            n = src_cardinality.require_finite()
        else:
            n = int(src_cardinality)

        if self.drop_remainder:
            out = n // self.batch_size
        else:
            out = (n + self.batch_size - 1) // self.batch_size
        return Cardinality.finite(out)

    def iter_dataset(self, src):
        it = iter(src)
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


class Batch(StructuralDataset):
    def __init__(self, src, batch_size: int, *, drop_remainder: bool = False):
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        super().__init__(src, BatchTransform(batch_size, drop_remainder=drop_remainder))


class UnbatchTransform(StructuralTransform):
    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return unbatch_spec_tree(input_spec)

    def infer_cardinality(self, src_cardinality):
        return Cardinality.UNKNOWN

    def iter_dataset(self, src):
        for batch in src:
            yield from default_split(batch)


class Unbatch(StructuralDataset):
    def __init__(self, src):
        super().__init__(src, UnbatchTransform())


class TakeTransform(StructuralTransform):
    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n must be non-negative.")
        self.n = n

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return input_spec

    def infer_cardinality(self, src_cardinality):
        if isinstance(src_cardinality, Cardinality):
            if src_cardinality.is_unknown:
                return Cardinality.finite(self.n)
            if src_cardinality.is_infinite:
                return Cardinality.finite(self.n)
            return Cardinality.finite(min(self.n, src_cardinality.require_finite()))
        return Cardinality.finite(min(self.n, int(src_cardinality)))

    def iter_dataset(self, src):
        yield from itertools.islice(iter(src), self.n)


class Take(StructuralDataset):
    def __init__(self, src, n: int):
        self.n = n
        super().__init__(src, TakeTransform(n))


class SkipTransform(StructuralTransform):
    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n must be non-negative.")
        self.n = n

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return input_spec

    def infer_cardinality(self, src_cardinality):
        if isinstance(src_cardinality, Cardinality):
            if src_cardinality.is_unknown:
                return Cardinality.UNKNOWN
            if src_cardinality.is_infinite:
                return Cardinality.INFINITE
            return Cardinality.finite(max(0, src_cardinality.require_finite() - self.n))
        return Cardinality.finite(max(0, int(src_cardinality) - self.n))

    def iter_dataset(self, src):
        it = iter(src)
        for _ in range(self.n):
            try:
                next(it)
            except StopIteration:
                return
        yield from it


class Skip(StructuralDataset):
    def __init__(self, src, n: int):
        self.n = n
        super().__init__(src, SkipTransform(n))


class RepeatTransform(StructuralTransform):
    def __init__(self, count: int | None = None):
        if count is not None and count < 0:
            raise ValueError("count must be non-negative or None.")
        self.count = count

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return input_spec

    def infer_cardinality(self, src_cardinality):
        if self.count is None:
            return Cardinality.INFINITE
        if isinstance(src_cardinality, Cardinality):
            if src_cardinality.is_unknown:
                return Cardinality.UNKNOWN
            if src_cardinality.is_infinite:
                return Cardinality.INFINITE if self.count > 0 else Cardinality.finite(0)
            return Cardinality.finite(src_cardinality.require_finite() * self.count)
        return Cardinality.finite(int(src_cardinality) * self.count)

    def iter_dataset(self, src):
        if self.count is None:
            while True:
                yield from src
        else:
            for _ in range(self.count):
                yield from src


class Repeat(StructuralDataset):
    def __init__(self, src, count: int | None = None):
        self.count = count
        super().__init__(src, RepeatTransform(count))


class ShuffleTransform(StructuralTransform):
    def __init__(self, buffer_size: int, *, seed=None):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self.buffer_size = buffer_size
        self.seed = seed

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return input_spec

    def infer_cardinality(self, src_cardinality):
        return src_cardinality

    def iter_dataset(self, src):
        import numpy as np

        rng = np.random.default_rng(seed=self.seed)
        it = iter(src)
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


class Shuffle(StructuralDataset):
    def __init__(self, src, buffer_size: int, *, seed=None):
        self.buffer_size = buffer_size
        self.seed = seed
        super().__init__(src, ShuffleTransform(buffer_size, seed=seed))
