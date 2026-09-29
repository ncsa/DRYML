"""Private typed streaming Parquet physical codec for CachedDataset.

This module intentionally imports PyArrow only when a Parquet operation is
selected.  The cache manifest remains the logical reconstruction authority;
the Parquet schema records bounded typed component segments for one chunk.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path

import numpy as np
from packaging.version import Version

from dryml.core.dtype import normalize_dtype
from dryml.core.tensor_spec import TensorSpec

from ._cache_model import CacheIntegrityError
from ._cache_numpy import _restore_storage_array, _storage_array


_MAX_SEGMENT_BYTES = 8 * 1024 * 1024
_WRITER_BATCH_BYTES = _MAX_SEGMENT_BYTES
_ROWS_PER_READ_BATCH = 1
_FORMAT_VERSION = b"1"
_METADATA = {
    b"dryml.cached-dataset.parquet": _FORMAT_VERSION,
    b"dryml.component-layout": b"typed-segments-v1",
}
_VALUE_TYPES = {
    "values_bool": "bool",
    "values_int8": "int8",
    "values_int16": "int16",
    "values_int32": "int32",
    "values_int64": "int64",
    "values_uint8": "uint8",
    "values_uint16": "uint16",
    "values_uint32": "uint32",
    "values_uint64": "uint64",
    "values_float32": "float32",
    "values_float64": "float64",
}
_COMPONENT_VALUE = 0
_COMPONENT_REAL = 1
_COMPONENT_IMAGINARY = 2
_COMPONENT_UTF8 = 3
_COMPONENT_STRING_OFFSETS = 4
pa = None
pq = None


def load_pyarrow():
    """Load the supported optional PyArrow runtime for a selected codec action.

    Returns:
        The ``pyarrow`` module and its ``pyarrow.parquet`` submodule.

    Raises:
        ImportError: If PyArrow is unavailable or older than the supported
            ``25.0.1`` format-qualification floor.
    """

    global pa, pq
    try:
        import pyarrow as imported_pa
        import pyarrow.parquet as imported_pq
    except ImportError as error:
        raise ImportError(
            "CachedDataset Parquet codec requires pyarrow>=25.0.1; install dryml[parquet]."
        ) from error
    if Version(imported_pa.__version__) < Version("25.0.1"):
        raise ImportError(
            "CachedDataset Parquet codec requires pyarrow>=25.0.1; "
            f"found unsupported pyarrow {imported_pa.__version__}."
        )
    pa, pq = imported_pa, imported_pq
    return pa, pq


def write_chunk(
        path: str, values: Iterable[tuple[np.ndarray, ...]], *, segment_bytes: int = _MAX_SEGMENT_BYTES,
        ) -> dict[str, object]:
    """Write one typed, bounded-segment Parquet chunk without a whole-cache table.

    Args:
        path: New private Parquet destination.
        values: Normalized logical yields in source order.
        segment_bytes: Maximum bytes represented by an ordinary list cell, capped
            at 8 MiB except for an unavoidable scalar representation.

    Returns:
        A closed, JSON-compatible deferred-integrity descriptor.

    Raises:
        ImportError: If the selected optional PyArrow runtime is unavailable.
        ValueError: If the requested segment budget is invalid.

    Each ``ParquetWriter`` batch contains a bounded number of physical rows.
    Logical yields are never split at the Dataset boundary, even when their typed
    component streams require multiple physical rows.
    """

    if type(segment_bytes) is not int or not 0 < segment_bytes <= _MAX_SEGMENT_BYTES:
        raise ValueError("CachedDataset Parquet segment budget is invalid.")
    pa, pq = load_pyarrow()
    schema = _schema(pa)
    count = 0
    rows: list[dict[str, object]] = []
    batch_bytes = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with pq.ParquetWriter(path, schema, compression="NONE") as writer:
            for yield_index, leaves in enumerate(values):
                count += 1
                for leaf_index, array in enumerate(leaves):
                    for row in _rows_for_leaf(yield_index, leaf_index, array, segment_bytes):
                        row_bytes = _row_bytes(row)
                        if rows and batch_bytes + row_bytes > _WRITER_BATCH_BYTES:
                            _write_rows(writer, pa, schema, rows)
                            rows = []
                            batch_bytes = 0
                        rows.append(row)
                        batch_bytes += row_bytes
            if rows:
                _write_rows(writer, pa, schema, rows)
    except CacheIntegrityError:
        raise
    except Exception as error:
        raise CacheIntegrityError("CachedDataset Parquet chunk could not be encoded.") from error
    return {
        "file": os.path.basename(path), "size": os.path.getsize(path),
        "sha256": _sha256(path), "count": count, "segment_bytes": segment_bytes,
    }


def read_chunk(path: str, descriptor: dict[str, object], leaves: tuple[TensorSpec, ...]) -> list[tuple[np.ndarray, ...]]:
    """Authenticate, fully validate, and decode one Parquet chunk before yielding it.

    The reader uses :meth:`pyarrow.parquet.ParquetFile.iter_batches` rather than
    materializing a table.  It validates every physical row and component before
    returning any logical yield from the chunk.
    """

    validate_chunk_descriptor(descriptor, leaves)
    try:
        valid_bytes = os.path.getsize(path) == descriptor["size"] and _sha256(path) == descriptor["sha256"]
    except OSError as error:
        raise CacheIntegrityError("CachedDataset Parquet chunk is missing.") from error
    if not valid_bytes:
        raise CacheIntegrityError("CachedDataset Parquet chunk digest does not match its manifest.")
    pa, pq = load_pyarrow()
    schema = _schema(pa)
    try:
        source = pq.ParquetFile(path)
        if not source.schema_arrow.equals(schema, check_metadata=True):
            raise CacheIntegrityError("CachedDataset Parquet schema is invalid.")
        rows: list[dict[str, object]] = []
        for batch in source.iter_batches(batch_size=_ROWS_PER_READ_BATCH):
            if not batch.schema.equals(schema, check_metadata=True):
                raise CacheIntegrityError("CachedDataset Parquet batch schema is invalid.")
            rows.extend(batch.to_pylist())
        return _decode_rows(rows, int(descriptor["count"]), leaves, int(descriptor["segment_bytes"]))
    except CacheIntegrityError:
        raise
    except ImportError:
        raise
    except Exception as error:
        raise CacheIntegrityError("CachedDataset Parquet payload cannot be decoded.") from error


def validate_chunk_descriptor(descriptor: object, leaves: tuple[TensorSpec, ...]) -> None:
    """Validate deferred Parquet descriptor metadata without importing PyArrow.

    Args:
        descriptor: Manifest entry for one chunk.
        leaves: Validated logical leaves, retained for a symmetric codec interface.

    Raises:
        CacheIntegrityError: If the closed descriptor is malformed.
    """

    del leaves
    if not isinstance(descriptor, dict) or set(descriptor) != {
            "file", "size", "sha256", "count", "segment_bytes"}:
        raise CacheIntegrityError("CachedDataset Parquet chunk descriptor is malformed.")
    if (type(descriptor["file"]) is not str
            or not re.fullmatch(r"chunk-[0-9]{8}\.parquet", descriptor["file"])
            or type(descriptor["size"]) is not int or descriptor["size"] < 0
            or type(descriptor["sha256"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
            or type(descriptor["count"]) is not int or descriptor["count"] < 0
            or type(descriptor["segment_bytes"]) is not int
            or not 0 < descriptor["segment_bytes"] <= _MAX_SEGMENT_BYTES):
        raise CacheIntegrityError("CachedDataset Parquet chunk descriptor is invalid.")


def _schema(pa):
    """Build the fixed non-null typed component-segment schema for one chunk."""

    fields = [
        pa.field("yield", pa.int64(), nullable=False),
        pa.field("leaf", pa.int32(), nullable=False),
        pa.field("component", pa.int8(), nullable=False),
        pa.field("segment", pa.int32(), nullable=False),
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("shape", _list_type(pa, pa.int64()), nullable=False),
    ]
    fields.extend(pa.field(
        name, _list_type(pa, pa.bool_() if type_name == "bool" else getattr(pa, type_name)()), nullable=False,
    ) for name, type_name in _VALUE_TYPES.items())
    return pa.schema(fields, metadata=_METADATA)


def _list_type(pa, value_type):
    """Return an ordinary bounded-offset list whose values cannot be null."""

    # Parquet normalizes list children to ``element``; matching that stable Arrow
    # spelling lets schema preflight reject any changed physical list contract.
    return pa.list_(pa.field("element", value_type, nullable=False))


def _write_rows(writer, pa, schema, rows: list[dict[str, object]]) -> None:
    """Write one bounded physical row batch under the exact schema."""

    writer.write_batch(pa.RecordBatch.from_pylist(rows, schema=schema))


def _rows_for_leaf(
        yield_index: int, leaf_index: int, array: np.ndarray, segment_bytes: int,
        ) -> Iterator[dict[str, object]]:
    """Encode one normalized leaf as ordered typed physical component rows."""

    shape = [int(item) for item in array.shape]
    dtype = normalize_dtype(array.dtype)
    if dtype.kind == "complex":
        real_dtype = np.float32 if dtype.name == "complex64" else np.float64
        yield from _numeric_rows(
            yield_index, leaf_index, _COMPONENT_REAL, np.asarray(array.real, dtype=real_dtype), shape, segment_bytes,
        )
        yield from _numeric_rows(
            yield_index, leaf_index, _COMPONENT_IMAGINARY, np.asarray(array.imag, dtype=real_dtype), shape, segment_bytes,
        )
        return
    if dtype.kind == "string":
        yield from _unicode_rows(yield_index, leaf_index, array, shape, segment_bytes)
        return
    if dtype.name == "float16":
        yield from _numeric_rows(
            yield_index, leaf_index, _COMPONENT_VALUE, array.view(np.uint16), shape, segment_bytes,
        )
        return
    storage, _ = _storage_array(array)
    yield from _numeric_rows(yield_index, leaf_index, _COMPONENT_VALUE, storage, shape, segment_bytes)


def _numeric_rows(yield_index, leaf_index, component, array, shape, segment_bytes) -> Iterator[dict[str, object]]:
    """Split one numeric component into ordered bounded typed list rows."""

    field = _field_for_dtype(array.dtype)
    flat = np.asarray(array).reshape(-1)
    item_size = max(flat.dtype.itemsize, 1)
    part_size = max(1, segment_bytes // item_size)
    for segment, start in enumerate(range(0, max(flat.size, 1), part_size)):
        end = min(start + part_size, flat.size)
        yield _row(yield_index, leaf_index, component, segment, start, end, shape, field, flat[start:end].tolist())


def _unicode_rows(yield_index, leaf_index, array, shape, segment_bytes) -> Iterator[dict[str, object]]:
    """Encode Unicode as bounded UTF-8 bytes and uint64 string-offset streams."""

    flat = np.asarray(array).reshape(-1)
    current: list[int] = []
    start = 0
    segment = 0
    for value in flat:
        encoded = str(value).encode("utf-8")
        while encoded:
            room = segment_bytes - len(current)
            current.extend(encoded[:room])
            encoded = encoded[room:]
            if len(current) == segment_bytes:
                yield _row(yield_index, leaf_index, _COMPONENT_UTF8, segment, start, start + len(current), shape, "values_uint8", current)
                start += len(current)
                segment += 1
                current = []
    if current or segment == 0:
        yield _row(yield_index, leaf_index, _COMPONENT_UTF8, segment, start, start + len(current), shape, "values_uint8", current)
    offsets = []
    offset = 0
    offset_segment = 0
    offset_start = 0
    offsets_per_segment = max(1, segment_bytes // 8)
    for value in chain((None,), flat):
        if value is not None:
            offset += len(str(value).encode("utf-8"))
        offsets.append(offset)
        if len(offsets) == offsets_per_segment:
            yield _row(yield_index, leaf_index, _COMPONENT_STRING_OFFSETS, offset_segment, offset_start, offset_start + len(offsets), shape, "values_uint64", offsets)
            offset_start += len(offsets)
            offset_segment += 1
            offsets = []
    if offsets:
        yield _row(yield_index, leaf_index, _COMPONENT_STRING_OFFSETS, offset_segment, offset_start, offset_start + len(offsets), shape, "values_uint64", offsets)


def _row(yield_index, leaf_index, component, segment, start, end, shape, selected, values):
    """Create one complete non-null physical row with exactly one populated value list."""

    row = {
        "yield": yield_index, "leaf": leaf_index, "component": component,
        "segment": segment, "start": start, "end": end, "shape": shape,
    }
    row.update({name: values if name == selected else [] for name in _VALUE_TYPES})
    return row


def _row_bytes(row: dict[str, object]) -> int:
    """Return conservative payload bytes represented by one physical row."""

    return sum(len(row[name]) * _item_size_for_field(name) for name in _VALUE_TYPES)


def _field_for_dtype(dtype: np.dtype) -> str:
    """Return the exact typed Parquet list field for physical NumPy storage."""

    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.bool_):
        return "values_bool"
    name = dtype.name
    field = f"values_{name}"
    if field not in _VALUE_TYPES:
        raise CacheIntegrityError("CachedDataset Parquet storage dtype is unsupported.")
    return field


def _decode_rows(rows, count: int, leaves: tuple[TensorSpec, ...], segment_bytes: int):
    """Validate ordered coverage and rebuild all logical values from typed rows."""

    if not leaves:
        if rows:
            raise CacheIntegrityError("CachedDataset Parquet empty-tree chunk has rows.")
        return [tuple() for _ in range(count)]
    components: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    shapes: dict[tuple[int, int], tuple[int, ...]] = {}
    previous = None
    for row in rows:
        _validate_row_shape(row)
        key = (row["yield"], row["leaf"], row["component"])
        if (row["yield"] < 0 or row["yield"] >= count or row["leaf"] < 0
                or row["leaf"] >= len(leaves) or row["component"] not in _components_for_spec(leaves[row["leaf"]])):
            raise CacheIntegrityError("CachedDataset Parquet row identifier is invalid.")
        if not _is_next_row(previous, row, leaves):
            raise CacheIntegrityError("CachedDataset Parquet segments are missing, duplicate, or reordered.")
        previous = row
        shape = tuple(row["shape"])
        if not leaves[row["leaf"]].compatible_with_shape(shape):
            raise CacheIntegrityError("CachedDataset Parquet shape violates its spec.")
        shape_key = (row["yield"], row["leaf"])
        if shape_key in shapes and shapes[shape_key] != shape:
            raise CacheIntegrityError("CachedDataset Parquet component shapes disagree.")
        shapes[shape_key] = shape
        _validate_typed_row(row, leaves[row["leaf"]], segment_bytes)
        components.setdefault(key, []).append(row)
    result = []
    for yield_index in range(count):
        output = []
        for leaf_index, spec in enumerate(leaves):
            shape = shapes.get((yield_index, leaf_index))
            if shape is None:
                raise CacheIntegrityError("CachedDataset Parquet leaf coverage is incomplete.")
            output.append(_decode_leaf(components, yield_index, leaf_index, shape, spec))
        result.append(tuple(output))
    return result


def _validate_row_shape(row: object) -> None:
    """Reject null, extra, and structurally malformed physical row values."""

    if not isinstance(row, dict) or set(row) != {
            "yield", "leaf", "component", "segment", "start", "end", "shape", *_VALUE_TYPES}:
        raise CacheIntegrityError("CachedDataset Parquet row columns are invalid.")
    if any(value is None for value in row.values()):
        raise CacheIntegrityError("CachedDataset Parquet rows cannot contain nulls.")
    if (type(row["yield"]) is not int or type(row["leaf"]) is not int or type(row["component"]) is not int
            or type(row["segment"]) is not int or type(row["start"]) is not int or type(row["end"]) is not int
            or row["segment"] < 0 or row["start"] < 0 or row["end"] < row["start"]
            or not isinstance(row["shape"], list) or any(type(item) is not int or item < 0 for item in row["shape"])
            or any(not isinstance(row[name], list) or any(item is None for item in row[name]) for name in _VALUE_TYPES)):
        raise CacheIntegrityError("CachedDataset Parquet row values are invalid.")


def _is_next_row(previous, row, leaves: tuple[TensorSpec, ...]) -> bool:
    """Return whether one row continues the closed physical source ordering."""

    if previous is None:
        return row["yield"] == row["leaf"] == row["segment"] == 0 and row["component"] == _components_for_spec(leaves[0])[0]
    if (row["yield"], row["leaf"], row["component"]) == (
            previous["yield"], previous["leaf"], previous["component"]):
        return row["segment"] == previous["segment"] + 1
    if row["segment"] != 0:
        return False
    roles = _components_for_spec(leaves[previous["leaf"]])
    role_index = roles.index(previous["component"])
    if role_index + 1 < len(roles):
        return (row["yield"], row["leaf"], row["component"]) == (
            previous["yield"], previous["leaf"], roles[role_index + 1],
        )
    if previous["leaf"] + 1 < len(leaves):
        return (row["yield"], row["leaf"], row["component"]) == (
            previous["yield"], previous["leaf"] + 1, _components_for_spec(leaves[previous["leaf"] + 1])[0],
        )
    return (row["yield"], row["leaf"], row["component"]) == (
        previous["yield"] + 1, 0, _components_for_spec(leaves[0])[0],
    )


def _components_for_spec(spec: TensorSpec) -> tuple[int, ...]:
    """Return the required physical component roles for one semantic leaf."""

    if spec.dtype.kind == "complex":
        return (_COMPONENT_REAL, _COMPONENT_IMAGINARY)
    if spec.dtype.kind == "string":
        return (_COMPONENT_UTF8, _COMPONENT_STRING_OFFSETS)
    return (_COMPONENT_VALUE,)


def _validate_typed_row(row, spec: TensorSpec, segment_bytes: int) -> None:
    """Require one role's selected list type, bounds, and empty alternatives."""

    selected = _selected_field(spec, row["component"])
    if any(row[name] for name in _VALUE_TYPES if name != selected):
        raise CacheIntegrityError("CachedDataset Parquet row has unexpected typed values.")
    values = row[selected]
    if row["end"] - row["start"] != len(values):
        raise CacheIntegrityError("CachedDataset Parquet offsets do not match typed values.")
    if len(values) * _item_size_for_field(selected) > segment_bytes and len(values) != 1:
        raise CacheIntegrityError("CachedDataset Parquet segment exceeds its budget.")
    if selected == "values_bool":
        valid = all(type(value) is bool for value in values)
    elif selected.startswith("values_float"):
        valid = all(type(value) in (float, int) and type(value) is not bool for value in values)
    else:
        bits = int(selected.removeprefix("values_uint")) if selected.startswith("values_uint") else int(selected.removeprefix("values_int"))
        signed = selected.startswith("values_int")
        lower, upper = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
        valid = all(type(value) is int and lower <= value <= upper for value in values)
    if not valid:
        raise CacheIntegrityError("CachedDataset Parquet typed values are invalid.")


def _item_size_for_field(field: str) -> int:
    """Return the fixed byte width used for bounded-list accounting."""

    if field == "values_bool":
        return 1
    if field.startswith("values_float"):
        return int(field.removeprefix("values_float")) // 8
    return int(field.removeprefix("values_uint") if field.startswith("values_uint") else field.removeprefix("values_int")) // 8


def _decode_leaf(components, yield_index, leaf_index, shape, spec):
    """Reassemble one fully covered leaf after row validation completes."""

    roles = _components_for_spec(spec)
    streams = [
        _join_component(
            components.get((yield_index, leaf_index, role)), _selected_field(spec, role),
        )
        for role in roles
    ]
    size = _shape_product(shape)
    if spec.dtype.kind == "complex":
        if any(stream.size != size for stream in streams):
            raise CacheIntegrityError("CachedDataset Parquet complex component sizes are invalid.")
        dtype = np.complex64 if spec.dtype.name == "complex64" else np.complex128
        return (streams[0] + 1j * streams[1]).astype(dtype).reshape(shape)
    if spec.dtype.kind == "string":
        data, offsets = streams
        if offsets.size != size + 1 or offsets.size == 0 or offsets[0] != 0 or offsets[-1] != data.size or np.any(offsets[1:] < offsets[:-1]):
            raise CacheIntegrityError("CachedDataset Parquet Unicode offsets are invalid.")
        try:
            values = [bytes(data[offsets[index]:offsets[index + 1]]).decode("utf-8") for index in range(size)]
        except UnicodeDecodeError as error:
            raise CacheIntegrityError("CachedDataset Parquet Unicode data is invalid.") from error
        return np.asarray(values, dtype=np.str_).reshape(shape)
    data = streams[0]
    if data.size != size:
        raise CacheIntegrityError("CachedDataset Parquet component sizes do not match its shape.")
    if spec.dtype.kind == "bfloat":
        return _restore_storage_array(data.astype(np.uint16, copy=False).reshape(shape), spec)
    if spec.dtype.name == "float16":
        return data.astype(np.uint16, copy=False).view(np.float16).reshape(shape)
    return data.astype(np.dtype(spec.dtype.name), copy=False).reshape(shape)


def _join_component(rows, field):
    """Require contiguous segment ordinals and concatenate one typed component stream."""

    if not isinstance(rows, list) or not rows:
        raise CacheIntegrityError("CachedDataset Parquet component coverage is incomplete.")
    expected_start = 0
    selected = None
    output = []
    for segment, row in enumerate(rows):
        if row["segment"] != segment or row["start"] != expected_start:
            raise CacheIntegrityError("CachedDataset Parquet segments are missing, duplicate, or reordered.")
        if selected is None:
            selected = field
        output.extend(row[selected])
        expected_start = row["end"]
    dtype = _dtype_for_field(selected)
    return np.asarray(output, dtype=dtype)


def _selected_field(spec: TensorSpec, component: int) -> str:
    """Return the sole populated typed list field for one logical component."""

    if spec.dtype.kind == "complex":
        return "values_float32" if spec.dtype.name == "complex64" else "values_float64"
    if spec.dtype.kind == "string":
        return "values_uint8" if component == _COMPONENT_UTF8 else "values_uint64"
    if spec.dtype.kind == "bfloat" or spec.dtype.name == "float16":
        return "values_uint16"
    return _field_for_dtype(np.dtype(spec.dtype.name))


def _dtype_for_field(field: str) -> np.dtype:
    """Map one physical typed list field to its NumPy storage dtype."""

    if field == "values_bool":
        return np.dtype(np.bool_)
    return np.dtype(field.removeprefix("values_"))


def _shape_product(shape: tuple[int, ...]) -> int:
    """Return a checked element count before reconstruction allocation."""

    product = 1
    for item in shape:
        product *= item
        if product > np.iinfo(np.intp).max:
            raise CacheIntegrityError("CachedDataset Parquet shape overflows allocation bounds.")
    return product


def _sha256(path: str) -> str:
    """Return the fixed-block SHA-256 of one completed private chunk file."""

    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["load_pyarrow", "read_chunk", "validate_chunk_descriptor", "write_chunk"]
