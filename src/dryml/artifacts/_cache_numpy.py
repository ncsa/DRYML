"""Private streaming NumPy physical codec for CachedDataset."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np

from dryml.core.dtype import normalize_dtype
from dryml.core.tensor_spec import Dim, TensorSpec

from ._cache_model import CacheIntegrityError


_MAX_SEGMENT_BYTES = 8 * 1024 * 1024
_MAX_ARCHIVE_MEMBER_BYTES = 1024 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 16 * 1024
_MAX_ARCHIVE_MEMBERS = 100_000
_MEMBER_NAME = re.compile(r"y[0-9]+_l[0-9]+_s[0-9]+\.npy\Z")


def normalize_leaf(value: object, spec: TensorSpec) -> np.ndarray:
    """Convert one installed dense tensor leaf to a validated NumPy array."""

    torch = sys.modules.get("torch")
    tensorflow = sys.modules.get("tensorflow")
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            try:
                import ml_dtypes
            except ImportError as error:
                raise TypeError("Torch bfloat16 caching requires ml_dtypes.") from error
            value = value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    if tensorflow is not None and isinstance(value, tensorflow.Tensor):
        value = value.numpy()
    try:
        array = np.asarray(value)
    except Exception as error:
        raise TypeError("CachedDataset leaf cannot be converted to NumPy.") from error
    try:
        dtype = normalize_dtype(array.dtype)
    except (TypeError, ValueError) as error:
        raise TypeError("CachedDataset leaf dtype is unsupported.") from error
    if array.dtype.kind in {"O", "S"} or (array.dtype.kind == "V" and dtype.kind != "bfloat"):
        raise TypeError("CachedDataset does not support object, byte-string, or structured leaves.")
    if dtype != spec.dtype:
        raise ValueError("CachedDataset leaf dtype does not match its declared spec.")
    if not spec.compatible_with_shape(array.shape):
        raise ValueError("CachedDataset leaf shape does not match its declared spec.")
    if spec.batched and array.ndim == 0:
        raise ValueError("CachedDataset batched leaf must have a leading dimension.")
    return array


def write_chunk(
        path: str, values: Iterable[tuple[np.ndarray, ...]], *, segment_bytes: int = _MAX_SEGMENT_BYTES,
) -> dict[str, object]:
    """Write one uncompressed bounded-segment NPZ chunk and its inventory.

    Args:
        path: New private chunk destination.
        values: Normalized logical yields in chunk order.
        segment_bytes: Maximum byte budget for one physical segment, capped at
            8 MiB by the logical cache contract.

    Returns:
        A closed JSON-compatible descriptor, including the segment budget needed
        for deferred validation.

    Raises:
        ValueError: If the requested segment budget is invalid.

    The writer holds only the current source array and its current view.  It
    never accumulates arrays from prior physical segments or logical yields.
    """

    if type(segment_bytes) is not int or not 0 < segment_bytes <= _MAX_SEGMENT_BYTES:
        raise ValueError("CachedDataset segment budget is invalid.")

    entries = []
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for yield_index, leaves in enumerate(values):
            leaf_entries = []
            for leaf_index, array in enumerate(leaves):
                storage, logical_dtype = _storage_array(array)
                count = max(1, (storage.nbytes + segment_bytes - 1) // segment_bytes)
                flat = storage.reshape(-1)
                # Empty arrays retain their dtype and shape in one empty component.
                item_size = max(storage.dtype.itemsize, 1)
                part_size = max(1, segment_bytes // item_size)
                names = []
                for segment_index in range(count):
                    name = f"y{yield_index}_l{leaf_index}_s{segment_index}"
                    part = flat[segment_index * part_size:(segment_index + 1) * part_size]
                    with archive.open(f"{name}.npy", "w") as member:
                        np.lib.format.write_array(member, part, allow_pickle=False)
                    names.append(name)
                leaf_entries.append({
                    "shape": list(array.shape), "dtype": storage.dtype.str,
                    "logical_dtype": logical_dtype, "names": names,
                })
            entries.append(leaf_entries)
    digest = _sha256(path)
    return {
        "file": os.path.basename(path), "size": os.path.getsize(path),
        "sha256": digest, "count": len(entries), "entries": entries,
        "segment_bytes": segment_bytes,
    }


def read_chunk(path: str, descriptor: dict[str, object], leaves: tuple[TensorSpec, ...]) -> list[tuple[np.ndarray, ...]]:
    """Verify, fully decode, and validate one NPZ chunk before exposing a yield.

    Archive inventory and every NPY header are checked before a component array
    can be allocated.  SHA-256 then authenticates all physical bytes before any
    logical result is returned.
    """

    validate_chunk_descriptor(descriptor, leaves)
    try:
        valid_bytes = os.path.getsize(path) == descriptor["size"] and _sha256(path) == descriptor["sha256"]
    except OSError as error:
        raise CacheIntegrityError("CachedDataset chunk is missing.") from error
    if not valid_bytes:
        raise CacheIntegrityError("CachedDataset chunk digest does not match its manifest.")
    entries = descriptor["entries"]
    segment_bytes = descriptor["segment_bytes"]
    try:
        with zipfile.ZipFile(path) as archive:
            expected = _descriptor_member_names(entries, leaves, segment_bytes)
            members = archive.infolist()
            if len(members) > _MAX_ARCHIVE_MEMBERS or len(set(expected)) != len(expected):
                raise CacheIntegrityError("CachedDataset NPZ member inventory is invalid.")
            by_name = {member.filename: member for member in members}
            if len(by_name) != len(members) or set(by_name) != {f"{name}.npy" for name in expected}:
                raise CacheIntegrityError("CachedDataset NPZ names do not match its inventory.")
            for name in expected:
                _validate_member_header(archive, by_name[f"{name}.npy"], name, segment_bytes)

            result = []
            for entry, specs in zip(entries, (leaves,) * len(entries)):
                output = []
                for leaf, spec in zip(entry, specs):
                    dtype = np.dtype(leaf["dtype"])
                    parts = [_read_member(archive, f"{name}.npy", dtype) for name in leaf["names"]]
                    product = _shape_product(leaf["shape"], dtype)
                    if sum(part.size for part in parts) != product:
                        raise CacheIntegrityError("CachedDataset component sizes do not match its shape.")
                    array = _restore_storage_array(
                        np.concatenate(parts).reshape(tuple(leaf["shape"])), spec,
                    )
                    if not spec.compatible_with_shape(array.shape):
                        raise CacheIntegrityError("CachedDataset decoded shape violates its spec.")
                    output.append(array)
                result.append(tuple(output))
            return result
    except CacheIntegrityError:
        raise
    except Exception as error:
        raise CacheIntegrityError("CachedDataset NPZ payload cannot be decoded.") from error


def validate_chunk_descriptor(descriptor: object, leaves: tuple[TensorSpec, ...]) -> None:
    """Validate a closed chunk descriptor without opening its payload file.

    This is used by completed-state restoration: manifest structure, declared
    shapes, dtypes, and segment inventory are eager authority, while file hashing
    and array decoding remain deferred until the corresponding chunk is consumed.
    """

    try:
        if not isinstance(descriptor, dict) or set(descriptor) != {
                "file", "size", "sha256", "count", "entries", "segment_bytes"}:
            raise CacheIntegrityError("CachedDataset chunk descriptor is malformed.")
        if (type(descriptor["file"]) is not str or not re.fullmatch(r"chunk-[0-9]{8}\.npz", descriptor["file"])
                or type(descriptor["size"]) is not int or descriptor["size"] < 0
                or type(descriptor["sha256"]) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])):
            raise CacheIntegrityError("CachedDataset chunk descriptor is invalid.")
        segment_bytes = descriptor["segment_bytes"]
        entries = descriptor["entries"]
        if (type(segment_bytes) is not int or not 0 < segment_bytes <= _MAX_SEGMENT_BYTES
                or type(descriptor["count"]) is not int or descriptor["count"] < 0
                or not isinstance(entries, list) or descriptor["count"] != len(entries)):
            raise CacheIntegrityError("CachedDataset chunk entry inventory is malformed.")
        _descriptor_member_names(entries, leaves, segment_bytes)
    except CacheIntegrityError:
        raise
    except Exception as error:
        raise CacheIntegrityError("CachedDataset chunk descriptor is malformed.") from error


def _descriptor_member_names(entries, leaves, segment_bytes: int) -> list[str]:
    """Return exact expected member stems after validating all logical entries."""

    expected: list[str] = []
    for yield_index, entry in enumerate(entries):
        _validate_entry(yield_index, entry, leaves, segment_bytes, expected)
    if len(expected) > _MAX_ARCHIVE_MEMBERS or len(set(expected)) != len(expected):
        raise CacheIntegrityError("CachedDataset NPZ member inventory is invalid.")
    return expected


def _validate_entry(yield_index: int, entry, leaves, segment_bytes: int, expected: list[str]) -> None:
    """Validate one declared logical yield without opening component arrays."""

    if not isinstance(entry, list) or len(entry) != len(leaves):
        raise CacheIntegrityError("CachedDataset chunk leaf inventory is malformed.")
    for leaf_index, (leaf, spec) in enumerate(zip(entry, leaves)):
        if not isinstance(leaf, dict) or set(leaf) != {"shape", "dtype", "logical_dtype", "names"}:
            raise CacheIntegrityError("CachedDataset chunk component inventory is malformed.")
        shape, names = leaf["shape"], leaf["names"]
        if not isinstance(shape, list) or not isinstance(names, list) or not names:
            raise CacheIntegrityError("CachedDataset component shape or names are invalid.")
        if any(type(item) is not int or item < 0 for item in shape):
            raise CacheIntegrityError("CachedDataset component shape is invalid.")
        try:
            dtype = np.dtype(leaf["dtype"])
        except (TypeError, ValueError) as error:
            raise CacheIntegrityError("CachedDataset component dtype is invalid.") from error
        if (dtype.kind in {"O", "S", "V"} or dtype.str != leaf["dtype"]
                or type(leaf["logical_dtype"]) is not str):
            raise CacheIntegrityError("CachedDataset component dtype is invalid.")
        try:
            logical_dtype = normalize_dtype(leaf["logical_dtype"])
        except (TypeError, ValueError) as error:
            raise CacheIntegrityError("CachedDataset component dtype is invalid.") from error
        if logical_dtype != spec.dtype:
            raise CacheIntegrityError("CachedDataset component dtype is invalid.")
        if logical_dtype.kind == "bfloat":
            if dtype != np.dtype(np.uint16):
                raise CacheIntegrityError("CachedDataset bfloat storage dtype is invalid.")
        elif normalize_dtype(dtype) != logical_dtype:
            raise CacheIntegrityError("CachedDataset component dtype is invalid.")
        product = _shape_product(shape, dtype)
        expected_count = max(1, (product * dtype.itemsize + segment_bytes - 1) // segment_bytes)
        if len(names) != expected_count or any(type(name) is not str for name in names):
            raise CacheIntegrityError("CachedDataset component segmentation is invalid.")
        for index, name in enumerate(names):
            if name != f"y{yield_index}_l{leaf_index}_s{index}":
                raise CacheIntegrityError("CachedDataset component names are reordered or invalid.")
            expected.append(name)


def _shape_product(shape: list[int], dtype: np.dtype) -> int:
    """Return a checked element count before any NPY payload allocation."""

    product = 1
    for item in shape:
        product *= item
        if product > sys.maxsize // max(dtype.itemsize, 1):
            raise CacheIntegrityError("CachedDataset component shape overflows allocation bounds.")
    return product


def _validate_member_header(archive, member, name: str, segment_bytes: int) -> None:
    """Check physical ZIP/NPY metadata and bounded declared allocation shape."""

    if (not _MEMBER_NAME.fullmatch(member.filename) or member.flag_bits & 1
            or member.compress_type != zipfile.ZIP_STORED
            or member.file_size > _MAX_ARCHIVE_MEMBER_BYTES):
        raise CacheIntegrityError("CachedDataset NPZ uses an unsupported physical encoding.")
    with archive.open(member, "r") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise CacheIntegrityError("CachedDataset NPY version is unsupported.")
        header_end = source.tell()
    if fortran or len(shape) != 1 or dtype.kind in {"O", "S", "V"}:
        raise CacheIntegrityError("CachedDataset NPY header is invalid.")
    is_unavoidable_scalar = shape == (1,) and dtype.itemsize > segment_bytes
    if (header_end > _MAX_NPY_HEADER_BYTES or type(shape[0]) is not int or shape[0] < 0
            or (shape[0] * dtype.itemsize > segment_bytes and not is_unavoidable_scalar)
            or member.file_size != header_end + shape[0] * dtype.itemsize):
        raise CacheIntegrityError("CachedDataset NPY segment exceeds its declared budget.")


def _read_member(archive, name: str, dtype: np.dtype) -> np.ndarray:
    """Read one already header-validated bounded NPY component."""

    with archive.open(name, "r") as source:
        value = np.lib.format.read_array(source, allow_pickle=False)
    if value.dtype != dtype or value.ndim != 1:
        raise CacheIntegrityError("CachedDataset NPY component changed during decoding.")
    return value


def _storage_array(array: np.ndarray) -> tuple[np.ndarray, str]:
    """Return physical storage and semantic dtype for one validated leaf.

    ``ml_dtypes.bfloat16`` is represented by NumPy as a void dtype, whose NPY
    descriptor loses bfloat semantics.  Persist its exact bits as ``uint16`` and
    retain the semantic dtype separately in the closed chunk inventory.
    """

    logical_dtype = normalize_dtype(array.dtype)
    if logical_dtype.kind == "bfloat":
        return array.view(np.uint16), logical_dtype.name
    return array, logical_dtype.name


def _restore_storage_array(array: np.ndarray, spec: TensorSpec) -> np.ndarray:
    """Restore the documented NumPy representation from validated storage bits."""

    if spec.dtype.kind != "bfloat":
        return array
    try:
        import ml_dtypes
    except ImportError as error:
        raise CacheIntegrityError("CachedDataset bfloat16 decoding requires ml_dtypes.") from error
    return array.view(ml_dtypes.bfloat16)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["normalize_leaf", "read_chunk", "validate_chunk_descriptor", "write_chunk"]
