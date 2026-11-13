"""Owned NumPy ragged values with validated, nested row partitions.

This reference representation is an ordinary tensor value, not a Serializable
Store container. It imports no optional tensor framework and performs no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from operator import index
from typing import Sequence

import numpy as np


def _splits(values, dtype):
    """Check overflow before narrowing newly computed partition offsets."""
    array = np.asarray(values)
    if array.size and (array.min() < 0 or array.max() > np.iinfo(dtype).max):
        raise ValueError("Row partition offsets exceed the requested integer dtype.")
    return array.astype(dtype)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class RaggedTensor:
    """A numeric packed-values tensor with one or more row-split arrays.

    Args:
        flat_values: NumPy array of rank at least one. Its leading dimension is
            the innermost partition's population; trailing axes are dense.
        nested_row_splits: Nonempty outer-to-inner sequence of one-dimensional
            NumPy int32 or int64 arrays, all using the same dtype. Each starts at
            zero, is nondecreasing, and ends at the next level's row population.

    Input buffers are copied and exposed read-only. Duplicate splits represent
    empty rows. ``shape`` reports the outer population, proved uniform partition
    widths (None otherwise), and dense trailing axes. ``ragged_rank`` counts
    partitions in the physical tensor, independently of logical batch metadata.

    Raises:
        TypeError: For non-array, object-dtype, or unsupported partition inputs.
        ValueError: For scalar values, absent partitions or inconsistent offsets.

    Side Effects:
        Allocates owned buffers. No implicit dense conversion, padding or I/O.
    """

    _flat_values: np.ndarray
    _partitions: tuple[np.ndarray, ...]
    _shape: tuple[int | None, ...]

    def __init__(self, flat_values: np.ndarray, nested_row_splits: Sequence[np.ndarray]):
        if not isinstance(flat_values, np.ndarray) or flat_values.dtype.hasobject:
            raise TypeError("Ragged flat_values must be a non-object NumPy array.")
        if flat_values.ndim < 1:
            raise ValueError("Ragged flat_values must have a leading population axis.")
        parts = tuple(nested_row_splits)
        if not parts:
            raise ValueError("Ragged tensors require at least one row partition.")
        for part in parts:
            if not isinstance(part, np.ndarray) or part.dtype not in (np.dtype("int32"), np.dtype("int64")):
                raise TypeError("Row splits must be NumPy int32 or int64 arrays.")
        values = np.array(flat_values, copy=True)
        parts = tuple(np.array(part, copy=True) for part in parts)
        for part in parts:
            if part.ndim != 1 or not len(part) or part[0] != 0 or np.any(part[1:] < part[:-1]):
                raise ValueError("Row splits must be nonempty, start at zero and be nondecreasing.")
            if part.dtype != parts[0].dtype:
                raise TypeError("Nested row splits must use one common integer dtype.")
        for level, part in enumerate(parts):
            population = len(parts[level + 1]) - 1 if level + 1 < len(parts) else len(values)
            if part[-1] != population:
                raise ValueError("Row partition endpoint disagrees with its child population.")
        partitions = parts
        for array in (values, *partitions):
            array.flags.writeable = False
        object.__setattr__(self, "_flat_values", values)
        object.__setattr__(self, "_partitions", partitions)
        widths = []
        for part in partitions:
            lengths = np.diff(part)
            widths.append(int(lengths[0]) if len(lengths) and np.all(lengths == lengths[0]) else None)
        object.__setattr__(self, "_shape", (len(partitions[0]) - 1, *widths, *values.shape[1:]))

    @property
    def flat_values(self) -> np.ndarray:
        """Return a read-only view of the packed values, including dense tail axes."""
        return self._flat_values.view()

    @property
    def nested_row_splits(self) -> tuple[np.ndarray, ...]:
        """Return read-only outer-to-inner partition views; offsets are not copied."""
        return tuple(part.view() for part in self._partitions)

    @property
    def dtype(self) -> np.dtype:
        """Return the packed values' NumPy dtype without inspecting content."""
        return self._flat_values.dtype

    @property
    def ragged_rank(self) -> int:
        """Return the positive number of physical row-partition levels."""
        return len(self._partitions)

    @property
    def row_splits_dtype(self) -> np.dtype:
        """Return the common NumPy int32 or int64 partition dtype."""
        return self._partitions[0].dtype

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Return cached physical shape, with widths proved from owned partitions."""
        return self._shape

    def __len__(self) -> int:
        """Return the outer row population, including empty rows."""
        return len(self._partitions[0]) - 1

    def __repr__(self) -> str:
        """Describe shape, dtype and partition rank without printing tensor content."""
        return f"RaggedTensor(shape={self.shape}, dtype={self.dtype}, ragged_rank={self.ragged_rank})"

    def __reduce__(self):
        """Reconstruct through validation/ownership on trusted pickle round trips.

        Returns:
            The class and constructor buffers. Loading copies and validates the
            buffers again, preserving ordinary read-only ownership. This is not
            a safe-deserialization boundary or a DRYML Store codec.
        """
        return type(self), (self._flat_values, self._partitions)

    def __array__(self, dtype=None, copy=None):
        """Reject NumPy coercion instead of implicitly densifying uniform rows.

        Args:
            dtype: Requested NumPy conversion dtype; never applied.
            copy: Requested NumPy copy policy; never applied.

        Raises:
            TypeError: Always. Use explicit row or flat_values access instead.
            This boundary reads no content and allocates no dense result.
        """
        raise TypeError("RaggedTensor requires explicit row or flat_values access; implicit dense conversion is unsupported.")

    def __getitem__(self, row: int) -> np.ndarray | RaggedTensor:
        """Extract one outer row, reducing partition rank by one.

        Args:
            row: Integer index, including negative indices. Slices are unsupported.

        Returns:
            A read-only dense view for rank one, or an owned RaggedTensor for
            deeper partitions. Empty rows preserve dense tail shape and dtype.

        Raises:
            TypeError: For non-integer indices. IndexError: For absent rows.
        """
        row = index(row)
        if row < 0:
            row += len(self)
        if row < 0 or row >= len(self):
            raise IndexError("Ragged row index is out of range.")
        start, stop = map(int, self._partitions[0][row:row + 2])
        if self.ragged_rank == 1:
            return self._flat_values[start:stop].view()
        parts = []
        for part in self._partitions[1:]:
            parts.append(part[start:stop + 1] - part[start])
            start, stop = int(part[start]), int(part[stop])
        return RaggedTensor(self._flat_values[start:stop], parts)

    def with_flat_values(self, values: np.ndarray) -> RaggedTensor:
        """Replace packed values while preserving row partitions.

        Args:
            values: NumPy array with the same leading population; dense trailing
                dimensions and dtype may change.

        Returns:
            An owned RaggedTensor containing copies of the new values/partitions.

        Raises:
            TypeError, ValueError: For invalid values or a changed population.
        """
        return RaggedTensor(values, self._partitions)

    def astype(self, dtype) -> RaggedTensor:
        """Return an owned dtype-converted value retaining all row partitions.

        Args:
            dtype: NumPy-compatible target dtype.

        Raises:
            TypeError, ValueError: For unsupported casts or object output dtype.
        """
        return self.with_flat_values(self._flat_values.astype(dtype))

    @classmethod
    def stack(cls, items: Sequence[np.ndarray | RaggedTensor], *, ragged_rank: int = 1,
              row_splits_dtype="int64") -> RaggedTensor:
        """Add an outer row partition to equally typed dense or ragged values.

        Args:
            items: Nonempty sequence. Dense values must admit ragged_rank-1
                uniform partitions; ragged values must already have that rank.
                All packed values must share dtype and dense trailing shape.
            ragged_rank: Positive partition rank of the result.
            row_splits_dtype: NumPy int32 or int64; overflowing offsets are errors.

        Returns:
            An owned RaggedTensor. Unequal lengths and empty rows are preserved.

        Raises:
            TypeError: For invalid ranks, dtypes or value types.
            ValueError: For empty input, incompatible ranks/tails or offset overflow.

        Side Effects:
            Allocates concatenated values and partitions, leaving inputs unchanged.
        """
        if type(ragged_rank) is not int or ragged_rank < 1:
            raise TypeError("ragged_rank must be a positive integer.")
        dtype = np.dtype(row_splits_dtype)
        if dtype not in (np.dtype("int32"), np.dtype("int64")):
            raise TypeError("Row splits require int32 or int64.")
        items = tuple(items)
        if not items:
            raise ValueError("Cannot stack an empty sequence without a dense-tail declaration.")
        values, partitions, lengths = [], [], []
        for item in items:
            if isinstance(item, cls):
                if item.ragged_rank != ragged_rank - 1:
                    raise ValueError("Stack input ragged rank disagrees with the result rank.")
                values.append(item._flat_values)
                partitions.append(item._partitions)
                lengths.append(len(item))
            elif isinstance(item, np.ndarray) and not item.dtype.hasobject:
                if item.ndim < ragged_rank:
                    raise ValueError("Dense input rank is too small for the requested partitions.")
                parts = []
                for axis in range(1, ragged_rank):
                    parts.append(_splits(np.arange(prod(item.shape[:axis]) + 1, dtype="int64") * item.shape[axis], dtype))
                values.append(item.reshape((prod(item.shape[:ragged_rank]), *item.shape[ragged_rank:])))
                partitions.append(tuple(parts))
                lengths.append(len(item))
            else:
                raise TypeError("Stack inputs must be non-object NumPy arrays or RaggedTensors.")
        if any(value.dtype != values[0].dtype or value.shape[1:] != values[0].shape[1:] for value in values):
            raise ValueError("Ragged stack requires equal dtype and dense trailing shape.")
        output_parts = [_splits(np.concatenate(([0], np.cumsum(lengths, dtype="int64"))), dtype)]
        for level in range(ragged_rank - 1):
            pieces, offset = [np.zeros(1, dtype="int64")], 0
            for parts in partitions:
                part = parts[level]
                pieces.append(part[1:].astype("int64") + offset)
                offset += int(part[-1])
            output_parts.append(_splits(np.concatenate(pieces), dtype))
        return cls(np.concatenate(values, axis=0), output_parts)
