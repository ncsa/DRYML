from __future__ import annotations

from dryml.core2.cardinality import Cardinality
from dryml.core2.tensor_spec import SpecTree, batch_spec_tree
from dryml.data.collate import default_collate
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
        return Cardinality.finite(out) if out > 0 else Cardinality.UNKNOWN

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
