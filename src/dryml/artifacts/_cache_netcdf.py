"""Private lazy NETCDF4 physical codec for CachedDataset.

The optional native library is imported only for a selected NetCDF operation.
Every interaction with netCDF4 or netCDF-C, including file open and close, is
serialized by one process-wide reentrant lock.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import numpy as np
from packaging.version import Version

from dryml.core.tensor_spec import TensorSpec

from ._cache_model import CacheIntegrityError
from ._cache_numpy import _restore_storage_array

_MAX_SEGMENT_BYTES = 8 * 1024 * 1024
_FORMAT = "dryml.cached-dataset.netcdf"
_LAYOUT = "flat-components-v1"
_NETCDF_LOCK = threading.RLock()
_COMPONENT_NAME = re.compile(
    r"component_y[0-9]{6}_l[0-9]{6}_c(?:value|real|imaginary|utf8|string_offsets)_s[0-9]{6}\Z"
)
_METADATA_NAME = re.compile(
    r"meta_l[0-9]{6}_(?:value_offsets|shape_offsets|shape_values)_s[0-9]{6}\Z"
)
nc = None


def load_netcdf4():
    """Load and qualify the optional netCDF4 runtime under the native lock.

    Returns:
        The supported ``netCDF4`` module.

    Raises:
        ImportError: If netCDF4 is unavailable or older than version 1.7.4.
    """

    global nc
    with _NETCDF_LOCK:
        try:
            import netCDF4 as imported_nc
        except ImportError as error:
            raise ImportError(
                "CachedDataset NetCDF codec requires netCDF4>=1.7.4; install dryml[netcdf]."
            ) from error
        if Version(imported_nc.__version__) < Version("1.7.4"):
            raise ImportError(
                "CachedDataset NetCDF codec requires netCDF4>=1.7.4; "
                f"found unsupported netCDF4 {imported_nc.__version__}."
            )
        nc = imported_nc
        return nc


def write_chunk(
    path: str,
    values: Iterable[tuple[np.ndarray, ...]],
    *,
    segment_bytes: int = _MAX_SEGMENT_BYTES,
) -> dict[str, object]:
    """Write one NETCDF4 chunk as bounded one-dimensional typed streams.

    Args:
        path: New private ``.nc`` chunk destination.
        values: Normalized logical yields in source order.
        segment_bytes: Maximum logical bytes in an ordinary physical variable.

    Returns:
        A closed JSON-compatible descriptor used for deferred validation.

    Raises:
        ImportError: If the selected optional runtime is unavailable.
        ValueError: If ``segment_bytes`` is outside the codec contract.
        CacheIntegrityError: If the physical chunk cannot be encoded.

    Tensor dimensions are metadata values, never NetCDF dimensions. Empty
    streams use one fixed placeholder element so a zero-sized tensor cannot
    accidentally create an unlimited NetCDF dimension.
    """

    if type(segment_bytes) is not int or not 0 < segment_bytes <= _MAX_SEGMENT_BYTES:
        raise ValueError("CachedDataset NetCDF segment budget is invalid.")
    module = load_netcdf4()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _NETCDF_LOCK:
        dataset = None
        try:
            dataset = module.Dataset(path, "w", format="NETCDF4")
            dataset.set_fill_off()
            count = _encode_dataset(dataset, values, segment_bytes)
        except CacheIntegrityError:
            raise
        except Exception as error:
            raise CacheIntegrityError(
                "CachedDataset NetCDF chunk could not be encoded."
            ) from error
        finally:
            if dataset is not None:
                try:
                    dataset.close()
                except Exception as error:
                    raise CacheIntegrityError(
                        "CachedDataset NetCDF chunk could not be closed."
                    ) from error
    return {
        "file": os.path.basename(path),
        "size": os.path.getsize(path),
        "sha256": _sha256(path),
        "count": count,
        "segment_bytes": segment_bytes,
    }


def read_chunk(
    path: str,
    descriptor: dict[str, object],
    leaves: tuple[TensorSpec, ...],
) -> list[tuple[np.ndarray, ...]]:
    """Authenticate and fully validate one NETCDF4 chunk before returning values.

    Args:
        path: Snapshot-local chunk path.
        descriptor: Closed manifest entry for the chunk.
        leaves: Validated logical tensor leaves in reconstruction order.

    Returns:
        Fully reconstructed NumPy leaf tuples. No value is returned until the
        complete physical schema and content have passed validation.

    Raises:
        ImportError: If the selected optional runtime is unavailable.
        CacheIntegrityError: If the descriptor, digest, schema, or content is
            incomplete, inconsistent, masked, scaled, or otherwise unsupported.
    """

    validate_chunk_descriptor(descriptor, leaves)
    try:
        valid = (
            os.path.getsize(path) == descriptor["size"]
            and _sha256(path) == descriptor["sha256"]
        )
    except OSError as error:
        raise CacheIntegrityError("CachedDataset NetCDF chunk is missing.") from error
    if not valid:
        raise CacheIntegrityError(
            "CachedDataset NetCDF chunk digest does not match its manifest."
        )
    module = load_netcdf4()
    with _NETCDF_LOCK:
        dataset = None
        try:
            dataset = module.Dataset(path, "r")
            dataset.set_auto_maskandscale(False)
            dataset.set_auto_mask(False)
            dataset.set_auto_scale(False)
            return _decode_dataset(
                dataset,
                int(descriptor["count"]),
                leaves,
                int(descriptor["segment_bytes"]),
            )
        except CacheIntegrityError:
            raise
        except ImportError:
            raise
        except Exception as error:
            raise CacheIntegrityError(
                "CachedDataset NetCDF payload cannot be decoded."
            ) from error
        finally:
            if dataset is not None:
                try:
                    dataset.close()
                except Exception as error:
                    raise CacheIntegrityError(
                        "CachedDataset NetCDF payload could not be closed."
                    ) from error


def validate_chunk_descriptor(
    descriptor: object, leaves: tuple[TensorSpec, ...]
) -> None:
    """Validate deferred NetCDF descriptor metadata without importing netCDF4.

    Args:
        descriptor: Manifest entry for one chunk.
        leaves: Validated logical leaves, retained for the shared codec interface.

    Raises:
        CacheIntegrityError: If the descriptor is not closed and canonical.
    """

    del leaves
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "file",
        "size",
        "sha256",
        "count",
        "segment_bytes",
    }:
        raise CacheIntegrityError("CachedDataset NetCDF chunk descriptor is malformed.")
    if (
        type(descriptor["file"]) is not str
        or not re.fullmatch(r"chunk-[0-9]{8}\.nc", descriptor["file"])
        or type(descriptor["size"]) is not int
        or descriptor["size"] < 0
        or type(descriptor["sha256"]) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
        or type(descriptor["count"]) is not int
        or descriptor["count"] < 0
        or type(descriptor["segment_bytes"]) is not int
        or not 0 < descriptor["segment_bytes"] <= _MAX_SEGMENT_BYTES
    ):
        raise CacheIntegrityError("CachedDataset NetCDF chunk descriptor is invalid.")


def _encode_dataset(dataset, values, segment_bytes: int) -> int:
    """Encode one logical chunk into an already-open NETCDF4 Dataset."""

    count = 0
    leaf_count = None
    value_offsets: list[list[int]] = []
    shape_offsets: list[list[int]] = []
    shape_values: list[list[int]] = []
    for yield_index, leaves in enumerate(values):
        if not isinstance(leaves, tuple):
            raise CacheIntegrityError(
                "CachedDataset NetCDF logical yield is malformed."
            )
        if leaf_count is None:
            leaf_count = len(leaves)
            value_offsets = [[0] for _ in leaves]
            shape_offsets = [[0] for _ in leaves]
            shape_values = [[] for _ in leaves]
        elif len(leaves) != leaf_count:
            raise CacheIntegrityError(
                "CachedDataset NetCDF logical leaf count changed."
            )
        for leaf_index, array in enumerate(leaves):
            if not isinstance(array, np.ndarray):
                raise CacheIntegrityError(
                    "CachedDataset NetCDF leaf is not normalized."
                )
            shape = [int(item) for item in array.shape]
            size = _shape_product(shape)
            value_offsets[leaf_index].append(value_offsets[leaf_index][-1] + size)
            shape_values[leaf_index].extend(shape)
            shape_offsets[leaf_index].append(len(shape_values[leaf_index]))
            _write_leaf(dataset, yield_index, leaf_index, array, segment_bytes)
        count += 1
    if leaf_count is None:
        leaf_count = 0
    for leaf_index in range(leaf_count):
        prefix = f"meta_l{leaf_index:06d}"
        _write_stream(
            dataset,
            f"{prefix}_value_offsets",
            np.asarray(value_offsets[leaf_index], dtype=np.uint64),
            segment_bytes,
        )
        _write_stream(
            dataset,
            f"{prefix}_shape_offsets",
            np.asarray(shape_offsets[leaf_index], dtype=np.uint64),
            segment_bytes,
        )
        _write_stream(
            dataset,
            f"{prefix}_shape_values",
            np.asarray(shape_values[leaf_index], dtype=np.int64),
            segment_bytes,
        )
    dataset.setncattr("dryml_format", _FORMAT)
    dataset.setncattr("layout", _LAYOUT)
    dataset.setncattr("version", np.int64(1))
    dataset.setncattr("yield_count", np.int64(count))
    dataset.setncattr("leaf_count", np.int64(leaf_count))
    dataset.setncattr("segment_bytes", np.int64(segment_bytes))
    return count


def _write_leaf(
    dataset, yield_index: int, leaf_index: int, array: np.ndarray, segment_bytes: int
) -> None:
    """Write all required physical component streams for one logical leaf."""

    prefix = f"component_y{yield_index:06d}_l{leaf_index:06d}"
    if array.dtype.kind == "c":
        dtype = np.dtype(
            np.float32 if array.dtype == np.dtype(np.complex64) else np.float64
        )
        _write_array_component(
            dataset, f"{prefix}_creal", array, dtype, segment_bytes, "real"
        )
        _write_array_component(
            dataset, f"{prefix}_cimaginary", array, dtype, segment_bytes, "imaginary"
        )
        return
    if array.dtype.kind == "U":
        _write_unicode(dataset, prefix, array, segment_bytes)
        return
    if array.dtype == np.dtype(np.float16) or array.dtype.kind == "V":
        _write_array_component(
            dataset,
            f"{prefix}_cvalue",
            array,
            np.dtype(np.uint16),
            segment_bytes,
            "bits",
        )
        return
    dtype = np.dtype(np.uint8) if array.dtype == np.dtype(np.bool_) else array.dtype
    _write_array_component(
        dataset, f"{prefix}_cvalue", array, dtype, segment_bytes, "value"
    )


def _write_array_component(
    dataset,
    base: str,
    array: np.ndarray,
    dtype: np.dtype,
    segment_bytes: int,
    mode: str,
) -> None:
    """Write one numeric component using only bounded conversion slices."""

    for segment, (start, end) in enumerate(
        _segments(array.size, dtype.itemsize, segment_bytes)
    ):
        if start == end:
            part = np.zeros(1, dtype=dtype)
        else:
            source = np.asarray(array.flat[start:end])
            if mode == "real":
                part = np.asarray(source.real, dtype=dtype)
            elif mode == "imaginary":
                part = np.asarray(source.imag, dtype=dtype)
            elif mode == "bits":
                part = np.ascontiguousarray(source).view(np.uint16)
            elif array.dtype == np.dtype(np.bool_):
                part = np.asarray(source, dtype=np.uint8)
            else:
                part = np.asarray(source, dtype=dtype)
        _write_variable(dataset, f"{base}_s{segment:06d}", dtype, part)


def _write_unicode(dataset, prefix: str, array: np.ndarray, segment_bytes: int) -> None:
    """Write strict UTF-8 bytes and uint64 offsets without one giant byte buffer."""

    offsets: list[int] = []
    offsets_per_segment = max(1, segment_bytes // np.dtype(np.uint64).itemsize)
    offset_segment = 0
    current = bytearray()
    total = 0
    segment = 0
    for value in chain((None,), array.flat):
        if value is None:
            offsets.append(0)
        else:
            encoded = str(value).encode("utf-8")
            total += len(encoded)
            offsets.append(total)
            while encoded:
                room = segment_bytes - len(current)
                current.extend(encoded[:room])
                encoded = encoded[room:]
                if len(current) == segment_bytes:
                    _write_variable(
                        dataset,
                        f"{prefix}_cutf8_s{segment:06d}",
                        np.dtype(np.uint8),
                        np.frombuffer(bytes(current), dtype=np.uint8),
                    )
                    segment += 1
                    current.clear()
        if len(offsets) == offsets_per_segment:
            _write_variable(
                dataset,
                f"{prefix}_cstring_offsets_s{offset_segment:06d}",
                np.dtype(np.uint64),
                np.asarray(offsets, dtype=np.uint64),
            )
            offset_segment += 1
            offsets = []
    if current:
        _write_variable(
            dataset,
            f"{prefix}_cutf8_s{segment:06d}",
            np.dtype(np.uint8),
            np.frombuffer(bytes(current), dtype=np.uint8),
        )
    elif segment == 0:
        _write_variable(
            dataset,
            f"{prefix}_cutf8_s000000",
            np.dtype(np.uint8),
            np.zeros(1, dtype=np.uint8),
        )
    if offsets:
        _write_variable(
            dataset,
            f"{prefix}_cstring_offsets_s{offset_segment:06d}",
            np.dtype(np.uint64),
            np.asarray(offsets, dtype=np.uint64),
        )


def _write_stream(dataset, base: str, values: np.ndarray, segment_bytes: int) -> None:
    """Write one metadata or logical stream as bounded fixed-size variables."""

    for segment, (start, end) in enumerate(
        _segments(values.size, values.dtype.itemsize, segment_bytes)
    ):
        part = values[start:end] if start != end else np.zeros(1, dtype=values.dtype)
        _write_variable(dataset, f"{base}_s{segment:06d}", values.dtype, part)


def _write_variable(dataset, name: str, dtype: np.dtype, values: np.ndarray) -> None:
    """Create one fixed non-fill contiguous variable and populate it exactly once."""

    dimension = f"{name}_dim"
    dataset.createDimension(dimension, int(values.size))
    variable = dataset.createVariable(
        name,
        dtype,
        (dimension,),
        fill_value=False,
        contiguous=True,
    )
    variable.set_auto_maskandscale(False)
    variable.set_auto_mask(False)
    variable.set_auto_scale(False)
    variable[:] = values


def _decode_dataset(
    dataset, count: int, leaves: tuple[TensorSpec, ...], segment_bytes: int
):
    """Validate the closed physical model and reconstruct every logical yield."""

    if dataset.data_model != "NETCDF4" or dataset.file_format != "NETCDF4":
        raise CacheIntegrityError("CachedDataset NetCDF file model is unsupported.")
    expected_attrs = {
        "dryml_format",
        "layout",
        "version",
        "yield_count",
        "leaf_count",
        "segment_bytes",
    }
    if set(dataset.ncattrs()) != expected_attrs:
        raise CacheIntegrityError("CachedDataset NetCDF global attributes are invalid.")
    if (
        dataset.getncattr("dryml_format") != _FORMAT
        or dataset.getncattr("layout") != _LAYOUT
        or not _attribute_int(dataset, "version", 1)
        or not _attribute_int(dataset, "yield_count", count)
        or not _attribute_int(dataset, "leaf_count", len(leaves))
        or not _attribute_int(dataset, "segment_bytes", segment_bytes)
    ):
        raise CacheIntegrityError(
            "CachedDataset NetCDF global metadata is inconsistent."
        )
    names = set(dataset.variables)
    if any(
        not (_METADATA_NAME.fullmatch(name) or _COMPONENT_NAME.fullmatch(name))
        for name in names
    ):
        raise CacheIntegrityError("CachedDataset NetCDF variable inventory is invalid.")
    expected_names: set[str] = set()
    expected_dimensions: set[str] = set()
    shapes: list[list[tuple[int, ...]]] = []
    for leaf_index, spec in enumerate(leaves):
        prefix = f"meta_l{leaf_index:06d}"
        value_offsets = _read_stream(
            dataset,
            f"{prefix}_value_offsets",
            np.dtype(np.uint64),
            count + 1,
            segment_bytes,
            expected_names,
            expected_dimensions,
        )
        shape_offsets = _read_stream(
            dataset,
            f"{prefix}_shape_offsets",
            np.dtype(np.uint64),
            count + 1,
            segment_bytes,
            expected_names,
            expected_dimensions,
        )
        _validate_offsets(value_offsets, "value")
        _validate_offsets(shape_offsets, "shape")
        shape_values = _read_stream(
            dataset,
            f"{prefix}_shape_values",
            np.dtype(np.int64),
            int(shape_offsets[-1]),
            segment_bytes,
            expected_names,
            expected_dimensions,
        )
        if np.any(shape_values < 0):
            raise CacheIntegrityError("CachedDataset NetCDF shape values are invalid.")
        leaf_shapes = []
        for yield_index in range(count):
            start, end = int(shape_offsets[yield_index]), int(
                shape_offsets[yield_index + 1]
            )
            shape = tuple(int(item) for item in shape_values[start:end])
            if not spec.compatible_with_shape(shape):
                raise CacheIntegrityError(
                    "CachedDataset NetCDF shape violates its spec."
                )
            if _shape_product(shape) != int(
                value_offsets[yield_index + 1] - value_offsets[yield_index]
            ):
                raise CacheIntegrityError(
                    "CachedDataset NetCDF value offsets disagree with shape metadata."
                )
            leaf_shapes.append(shape)
        shapes.append(leaf_shapes)

    result: list[list[np.ndarray]] = [[] for _ in range(count)]
    for yield_index in range(count):
        for leaf_index, spec in enumerate(leaves):
            shape = shapes[leaf_index][yield_index]
            size = _shape_product(shape)
            prefix = f"component_y{yield_index:06d}_l{leaf_index:06d}"
            array = _read_leaf(
                dataset,
                prefix,
                spec,
                shape,
                size,
                segment_bytes,
                expected_names,
                expected_dimensions,
            )
            result[yield_index].append(array)
    if names != expected_names or set(dataset.dimensions) != expected_dimensions:
        raise CacheIntegrityError(
            "CachedDataset NetCDF schema has missing or extra variables or dimensions."
        )
    if any(
        dimension.isunlimited() or len(dimension) <= 0
        for dimension in dataset.dimensions.values()
    ):
        raise CacheIntegrityError(
            "CachedDataset NetCDF dimensions must be fixed and nonempty."
        )
    return [tuple(item) for item in result]


def _read_leaf(
    dataset,
    prefix: str,
    spec: TensorSpec,
    shape: tuple[int, ...],
    size: int,
    segment_bytes: int,
    expected_names: set[str],
    expected_dimensions: set[str],
) -> np.ndarray:
    """Read one leaf after its explicit shape and value offsets are valid."""

    if spec.dtype.kind == "complex":
        dtype = np.dtype(np.float32 if spec.dtype.name == "complex64" else np.float64)
        real = _read_stream(
            dataset,
            f"{prefix}_creal",
            dtype,
            size,
            segment_bytes,
            expected_names,
            expected_dimensions,
        )
        imaginary = _read_stream(
            dataset,
            f"{prefix}_cimaginary",
            dtype,
            size,
            segment_bytes,
            expected_names,
            expected_dimensions,
        )
        output_dtype = np.complex64 if spec.dtype.name == "complex64" else np.complex128
        output = np.empty(size, dtype=output_dtype)
        output.real = real
        output.imag = imaginary
        return output.reshape(shape)
    if spec.dtype.kind == "string":
        offsets = _read_stream(
            dataset,
            f"{prefix}_cstring_offsets",
            np.dtype(np.uint64),
            size + 1,
            segment_bytes,
            expected_names,
            expected_dimensions,
        )
        _validate_offsets(offsets, "Unicode")
        data = _read_stream(
            dataset,
            f"{prefix}_cutf8",
            np.dtype(np.uint8),
            int(offsets[-1]),
            segment_bytes,
            expected_names,
            expected_dimensions,
        )
        try:
            values = [
                bytes(data[int(offsets[index]):int(offsets[index + 1])]).decode(
                    "utf-8"
                )
                for index in range(size)
            ]
        except UnicodeDecodeError as error:
            raise CacheIntegrityError(
                "CachedDataset NetCDF Unicode data is invalid."
            ) from error
        return np.asarray(values, dtype=np.str_).reshape(shape)
    if spec.dtype.kind == "bfloat" or spec.dtype.name == "float16":
        data = _read_stream(
            dataset,
            f"{prefix}_cvalue",
            np.dtype(np.uint16),
            size,
            segment_bytes,
            expected_names,
            expected_dimensions,
        ).reshape(shape)
        if spec.dtype.kind == "bfloat":
            return _restore_storage_array(data, spec)
        return data.view(np.float16)
    dtype = (
        np.dtype(np.uint8) if spec.dtype.name == "bool" else np.dtype(spec.dtype.name)
    )
    data = _read_stream(
        dataset,
        f"{prefix}_cvalue",
        dtype,
        size,
        segment_bytes,
        expected_names,
        expected_dimensions,
    )
    if spec.dtype.name == "bool":
        if np.any(data > 1):
            raise CacheIntegrityError(
                "CachedDataset NetCDF boolean values are invalid."
            )
        data = data.astype(np.bool_)
    return data.reshape(shape)


def _read_stream(
    dataset,
    base: str,
    dtype: np.dtype,
    count: int,
    segment_bytes: int,
    expected_names: set[str],
    expected_dimensions: set[str],
) -> np.ndarray:
    """Validate and join one exact sequence of bounded physical variables."""

    part_size = max(1, segment_bytes // max(dtype.itemsize, 1))
    segment_count = 1 if count == 0 else (count + part_size - 1) // part_size
    if type(count) is not int or count < 0 or segment_count > len(dataset.variables):
        raise CacheIntegrityError(
            "CachedDataset NetCDF component length exceeds its physical inventory."
        )
    parts = []
    for segment, (start, end) in enumerate(
        _segments(count, dtype.itemsize, segment_bytes)
    ):
        name = f"{base}_s{segment:06d}"
        dimension_name = f"{name}_dim"
        expected_names.add(name)
        expected_dimensions.add(dimension_name)
        if name not in dataset.variables or dimension_name not in dataset.dimensions:
            raise CacheIntegrityError(
                "CachedDataset NetCDF component coverage is incomplete."
            )
        variable = dataset.variables[name]
        variable.set_auto_maskandscale(False)
        variable.set_auto_mask(False)
        variable.set_auto_scale(False)
        physical_count = end - start if end != start else 1
        if (
            np.dtype(variable.dtype) != dtype
            or variable.dimensions != (dimension_name,)
            or len(dataset.dimensions[dimension_name]) != physical_count
            or dataset.dimensions[dimension_name].isunlimited()
            or variable.ncattrs()
        ):
            raise CacheIntegrityError(
                "CachedDataset NetCDF component schema is invalid."
            )
        filters = variable.filters()
        if (
            any(bool(value) for key, value in filters.items() if key != "complevel")
            or filters.get("complevel", 0) != 0
        ):
            raise CacheIntegrityError(
                "CachedDataset NetCDF component uses unsupported filters."
            )
        if variable.chunking() != "contiguous" or variable.quantization() is not None:
            raise CacheIntegrityError(
                "CachedDataset NetCDF component uses unsupported storage controls."
            )
        value = variable[:]
        if (
            np.ma.isMaskedArray(value)
            or not isinstance(value, np.ndarray)
            or value.dtype != dtype
            or value.shape != (physical_count,)
        ):
            raise CacheIntegrityError("CachedDataset NetCDF component data is invalid.")
        if start == end:
            if value[0] != 0:
                raise CacheIntegrityError(
                    "CachedDataset NetCDF empty component placeholder is invalid."
                )
        else:
            parts.append(value)
    if count == 0:
        return np.empty(0, dtype=dtype)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _validate_offsets(offsets: np.ndarray, kind: str) -> None:
    """Require a zero-based monotonic uint64 offset stream."""

    if offsets.size == 0 or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise CacheIntegrityError(
            f"CachedDataset NetCDF {kind} offsets are invalid."
        )


def _attribute_int(dataset, name: str, expected: int) -> bool:
    """Return whether one global attribute is an exact non-boolean integer value."""

    value = dataset.getncattr(name)
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) == expected
    )


def _segments(count: int, item_size: int, segment_bytes: int):
    """Yield contiguous logical ranges, including one range for an empty stream."""

    part_size = max(1, segment_bytes // max(item_size, 1))
    if count == 0:
        yield 0, 0
        return
    for start in range(0, count, part_size):
        yield start, min(start + part_size, count)


def _shape_product(shape) -> int:
    """Return a checked platform-sized element count for one physical shape."""

    product = 1
    for item in shape:
        product *= int(item)
        if product > np.iinfo(np.intp).max:
            raise CacheIntegrityError(
                "CachedDataset NetCDF shape overflows allocation bounds."
            )
    return product


def _sha256(path: str) -> str:
    """Return the fixed-block SHA-256 digest of one completed chunk."""

    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["load_netcdf4", "read_chunk", "validate_chunk_descriptor", "write_chunk"]
