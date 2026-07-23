"""Lazy partitioned Parquet representation for rectangular NumPy rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from dryml.records import (
    AdapterDescriptor,
    AdapterRegistry,
    AdapterUnsupportedError,
    RepresentationRequirement,
    RepresentationSpec,
)

from .numpy_sequence import iter_numpy_sequence_partitions, read_numpy_sequence_index


PARQUET_KIND = "dryml.parquet"
PARQUET_SCHEMA = "dryml.parquet_sequence.v1"
PARQUET_REPRESENTATION = RepresentationSpec.create(
    PARQUET_KIND,
    version="1",
    traits=("columnar", "table", "stream-readable"),
    storage_kinds=("product-dir",),
    payload={"index": "index.json", "partitions": "parquet"},
)


class ParquetCompatibilityError(TypeError):
    """Raised when sequence rows cannot be represented as a flat table."""


class ParquetCorruptError(RuntimeError):
    """Raised when Parquet metadata or partition bytes are inconsistent."""


class ParquetUnavailableError(AdapterUnsupportedError):
    """Raised when the optional PyArrow dependency is unavailable."""

    def __init__(self, message: str = "Parquet support requires the dryml[parquet] extra"):
        super().__init__(message, code="optional_dependency_missing")


@dataclass(frozen=True, slots=True)
class ParquetPartition:
    """One ordered Parquet row partition."""

    index: int
    path: str
    start: int
    stop: int
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ParquetCorruptError("partition index must be non-negative")
        _validate_path(self.path)
        if type(self.start) is not int or type(self.stop) is not int or self.start < 0 or self.stop <= self.start:
            raise ParquetCorruptError("partition row bounds are invalid")
        if type(self.size) is not int or self.size < 0:
            raise ParquetCorruptError("partition size must be non-negative")
        if not _is_digest(self.sha256):
            raise ParquetCorruptError("partition digest is invalid")

    def to_json(self) -> dict[str, Any]:
        """Return strict partition metadata."""

        return {
            "index": self.index,
            "path": self.path,
            "start": self.start,
            "stop": self.stop,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, value: Any) -> "ParquetPartition":
        """Decode one strict partition entry."""

        fields = {"index", "path", "start", "stop", "size", "sha256"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ParquetCorruptError("partition metadata fields are malformed")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ParquetIndex:
    """Compact schema and ordered-partition index for one table product."""

    count: int
    dtype: str
    shape: tuple[int, ...]
    columns: tuple[str, ...]
    partitions: tuple[ParquetPartition, ...] = ()

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ParquetCorruptError("Parquet row count must be non-negative")
        try:
            dtype = np.dtype(self.dtype)
        except Exception as exc:
            raise ParquetCorruptError("Parquet NumPy dtype is invalid") from exc
        if dtype.hasobject or dtype.kind not in "biufUS":
            raise ParquetCorruptError("Parquet NumPy dtype is unsupported")
        shape = tuple(self.shape)
        invalid_shape = any((
            len(shape) > 1,
            any(type(dim) is not int or dim < 0 for dim in shape),
            bool(shape and shape[0] == 0),
        ))
        if invalid_shape:
            raise ParquetCorruptError("Parquet row shape must be scalar or one-dimensional")
        expected_columns = _column_names(shape)
        if tuple(self.columns) != expected_columns:
            raise ParquetCorruptError("Parquet columns do not match row shape")
        partitions = tuple(
            item if isinstance(item, ParquetPartition) else ParquetPartition.from_json(item)
            for item in self.partitions
        )
        row = 0
        for index, partition in enumerate(partitions):
            if partition.index != index or partition.start != row:
                raise ParquetCorruptError("Parquet partitions are not contiguous and ordered")
            row = partition.stop
        if row != self.count:
            raise ParquetCorruptError("Parquet partition rows do not match row count")
        object.__setattr__(self, "dtype", dtype.str)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "partitions", partitions)

    def to_json(self) -> dict[str, Any]:
        """Return the versioned compact index document."""

        return {
            "schema": PARQUET_SCHEMA,
            "schema_version": 1,
            "count": self.count,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "columns": list(self.columns),
            "partitions": [item.to_json() for item in self.partitions],
        }

    def to_bytes(self) -> bytes:
        """Return deterministic UTF-8 JSON bytes."""

        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    @classmethod
    def from_json(cls, value: Any) -> "ParquetIndex":
        """Decode and validate a strict index document."""

        fields = {"schema", "schema_version", "count", "dtype", "shape", "columns", "partitions"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ParquetCorruptError("Parquet index fields are malformed")
        if value.get("schema") != PARQUET_SCHEMA or value.get("schema_version") != 1:
            raise ParquetCorruptError("Parquet index schema is unsupported")
        if not isinstance(value.get("shape"), list) or not isinstance(value.get("columns"), list) or not isinstance(value.get("partitions"), list):
            raise ParquetCorruptError("Parquet index arrays are malformed")
        return cls(
            count=value.get("count"),
            dtype=value.get("dtype"),
            shape=tuple(value["shape"]),
            columns=tuple(value["columns"]),
            partitions=tuple(ParquetPartition.from_json(item) for item in value["partitions"]),
        )


def write_parquet_sequence(
    elements: Iterable[Any],
    root: str | Path,
    *,
    partition_rows: int = 1024,
) -> ParquetIndex:
    """Write scalar or one-dimensional rows to bounded Parquet partitions."""

    if type(partition_rows) is not int or partition_rows < 1:
        raise ValueError("partition_rows must be a positive integer")
    _load_pyarrow()
    product_root = Path(root)
    product_root.mkdir(parents=True, exist_ok=True)
    dtype = None
    shape = None
    partitions: list[ParquetPartition] = []
    buffer: list[np.ndarray] = []
    count = 0

    def flush() -> None:
        nonlocal count, buffer
        if not buffer:
            return
        values = np.stack(buffer, axis=0)
        partition = _write_array_partition(values, product_root, len(partitions), count)
        partitions.append(partition)
        count = partition.stop
        buffer = []

    for element in elements:
        row = _as_tabular_row(element)
        if dtype is None:
            dtype, shape = row.dtype, row.shape
        elif row.dtype != dtype or row.shape != shape:
            raise ParquetCompatibilityError("Parquet sequence rows must have one stable dtype and shape")
        buffer.append(row)
        if len(buffer) >= partition_rows:
            flush()
    flush()
    if dtype is None or shape is None:
        raise ParquetCompatibilityError("an empty sequence requires declared tabular schema")
    index = ParquetIndex(count, dtype.str, shape, _column_names(shape), tuple(partitions))
    (product_root / "index.json").write_bytes(index.to_bytes())
    return index


def write_parquet_from_numpy_sequence(source_root: str | Path, target_root: str | Path) -> ParquetIndex:
    """Convert one NumPy shard at a time without materializing the full source."""

    _load_pyarrow()
    source_index = read_numpy_sequence_index(source_root)
    tree = source_index.tree
    if tree is None:
        raise ParquetCompatibilityError("an empty NumPy sequence has no declared tabular schema")
    if tree.get("kind") != "leaf":
        raise ParquetCompatibilityError("nested NumPy tensor trees are not Parquet-table compatible")
    dtype = np.dtype(tree["dtype"])
    shape = tuple(tree["shape"])
    _validate_tabular_schema(dtype, shape)
    product_root = Path(target_root)
    product_root.mkdir(parents=True, exist_ok=True)
    partitions = []
    for current_tree, source_shard, arrays in iter_numpy_sequence_partitions(source_root):
        if current_tree != tree or len(arrays) != 1:
            raise ParquetCompatibilityError("NumPy sequence tensor tree changed during conversion")
        values = arrays[0]
        partition = _write_array_partition(values, product_root, len(partitions), source_shard.start)
        if partition.stop != source_shard.stop:
            raise ParquetCompatibilityError("NumPy and Parquet partition row ranges disagree")
        partitions.append(partition)
    index = ParquetIndex(source_index.count, dtype.str, shape, _column_names(shape), tuple(partitions))
    (product_root / "index.json").write_bytes(index.to_bytes())
    return index


def read_parquet_index(root: str | Path) -> ParquetIndex:
    """Read and validate the compact Parquet index without importing PyArrow."""

    try:
        value = json.loads((Path(root) / "index.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise ParquetCorruptError("Parquet index is missing or unreadable") from exc
    return ParquetIndex.from_json(value)


def iter_parquet_sequence(root: str | Path) -> Iterator[np.ndarray | np.generic]:
    """Yield rows in partition order, importing PyArrow only on iteration."""

    _pa, pq = _load_pyarrow()
    product_root = Path(root)
    index = read_parquet_index(product_root)
    dtype = np.dtype(index.dtype)
    for partition in index.partitions:
        path = product_root / partition.path
        _verify_partition(path, partition)
        try:
            table = pq.read_table(path, columns=list(index.columns))
        except Exception as exc:
            raise ParquetCorruptError(f"Parquet partition {partition.path!r} is unreadable") from exc
        if table.column_names != list(index.columns) or table.num_rows != partition.stop - partition.start:
            raise ParquetCorruptError("Parquet partition schema or row count is inconsistent")
        columns = [np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=dtype) for name in index.columns]
        for row in range(table.num_rows):
            if not index.shape:
                yield columns[0][row]
            else:
                yield np.asarray([column[row] for column in columns], dtype=dtype)


def numpy_to_parquet_adapter_registry() -> AdapterRegistry:
    """Return the built-in streaming managed cache adapter registry."""

    registry = AdapterRegistry()
    registry.register(
        AdapterDescriptor(
            "dryml.numpy_sequence_to_parquet",
            RepresentationRequirement(kind="dryml.numpy_sequence"),
            RepresentationRequirement(kind=PARQUET_KIND, representation_id=PARQUET_REPRESENTATION.id),
            version="1",
            cost=1.0,
            streaming=True,
            materializes_source=False,
        ),
        runner=_run_numpy_to_parquet,
    )
    return registry


def _run_numpy_to_parquet(context):
    context.store.records.write_spec(
        PARQUET_REPRESENTATION.to_envelope(), family="representation"
    )
    record = context.source_record.record
    if len(record.storage) != 1 or record.storage[0].kind != "product-dir":
        raise ParquetCompatibilityError("NumPy sequence source storage is malformed")
    source_root = context.store.records.resolve_storage_ref(
        record.storage[0], record_id=context.source_record.ref.record_id
    )
    index = write_parquet_from_numpy_sequence(source_root, context.session.staging_dir)
    return {
        "storage_role": "parquet-table",
        "payload": {
            "row_count": index.count,
            "partition_count": len(index.partitions),
            "columns": list(index.columns),
        },
    }


def _as_tabular_row(value: Any) -> np.ndarray:
    if isinstance(value, (Mapping, tuple, list)):
        raise ParquetCompatibilityError("nested tensor structures are not Parquet-table compatible")
    row = np.asarray(value)
    _validate_tabular_schema(row.dtype, row.shape)
    return row


def _validate_tabular_schema(dtype: np.dtype, shape: tuple[int, ...]) -> None:
    _validate_dtype(dtype)
    if len(shape) > 1:
        raise ParquetCompatibilityError("Parquet rows must be scalar or one-dimensional")
    if shape and shape[0] == 0:
        raise ParquetCompatibilityError("Parquet rows must contain at least one column")


def _validate_dtype(dtype: np.dtype) -> None:
    if dtype.hasobject or dtype.kind not in "biufUS":
        raise ParquetCompatibilityError(f"NumPy dtype {dtype} is not Parquet-table compatible")


def _column_names(shape: tuple[int, ...]) -> tuple[str, ...]:
    return ("value",) if not shape else tuple(f"value_{index:06d}" for index in range(shape[0]))


def _write_array_partition(values: np.ndarray, root: Path, index: int, start: int) -> ParquetPartition:
    pa, pq = _load_pyarrow()
    row_shape = tuple(values.shape[1:])
    _validate_tabular_schema(values.dtype, row_shape)
    names = _column_names(row_shape)
    arrays = [values] if not row_shape else [values[:, column] for column in range(row_shape[0])]
    table = pa.Table.from_arrays([pa.array(value) for value in arrays], names=list(names))
    relative = f"partitions/{index:08d}.parquet"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=None)
    size = path.stat().st_size
    return ParquetPartition(index, relative, start, start + len(values), size, _sha256(path))


def _verify_partition(path: Path, partition: ParquetPartition) -> None:
    try:
        size = path.stat().st_size
    except Exception as exc:
        raise ParquetCorruptError(f"Parquet partition {partition.path!r} is missing") from exc
    if size != partition.size or _sha256(path) != partition.sha256:
        raise ParquetCorruptError(f"Parquet partition {partition.path!r} failed integrity validation")


def _load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ParquetUnavailableError() from exc
    return pa, pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_path(path: str) -> None:
    if not isinstance(path, str):
        raise ParquetCorruptError("partition path must be a string")
    pure = PurePosixPath(path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ParquetCorruptError("partition path is unsafe")


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "PARQUET_KIND",
    "PARQUET_REPRESENTATION",
    "ParquetCompatibilityError",
    "ParquetCorruptError",
    "ParquetIndex",
    "ParquetPartition",
    "ParquetUnavailableError",
    "iter_parquet_sequence",
    "numpy_to_parquet_adapter_registry",
    "read_parquet_index",
    "write_parquet_from_numpy_sequence",
    "write_parquet_sequence",
]
