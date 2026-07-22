"""Lightweight physical representations for managed Artifact products."""

from .numpy_sequence import (
    NUMPY_SEQUENCE_KIND,
    NUMPY_SEQUENCE_REPRESENTATION,
    NumpySequenceCorruptError,
    NumpySequenceIndex,
    NumpySequenceShard,
    iter_numpy_sequence,
    iter_numpy_sequence_partitions,
    read_numpy_sequence_index,
    write_numpy_sequence,
    write_numpy_sequence_stream,
)
from .parquet import (
    PARQUET_KIND,
    PARQUET_REPRESENTATION,
    ParquetCompatibilityError,
    ParquetCorruptError,
    ParquetIndex,
    ParquetPartition,
    ParquetUnavailableError,
    iter_parquet_sequence,
    numpy_to_parquet_adapter_registry,
    read_parquet_index,
    write_parquet_from_numpy_sequence,
    write_parquet_sequence,
)


__all__ = [
    "NUMPY_SEQUENCE_KIND",
    "NUMPY_SEQUENCE_REPRESENTATION",
    "NumpySequenceCorruptError",
    "NumpySequenceIndex",
    "NumpySequenceShard",
    "PARQUET_KIND",
    "PARQUET_REPRESENTATION",
    "ParquetCompatibilityError",
    "ParquetCorruptError",
    "ParquetIndex",
    "ParquetPartition",
    "ParquetUnavailableError",
    "iter_numpy_sequence",
    "iter_numpy_sequence_partitions",
    "iter_parquet_sequence",
    "numpy_to_parquet_adapter_registry",
    "read_numpy_sequence_index",
    "read_parquet_index",
    "write_numpy_sequence",
    "write_numpy_sequence_stream",
    "write_parquet_from_numpy_sequence",
    "write_parquet_sequence",
]
