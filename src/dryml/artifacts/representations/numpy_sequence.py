"""Bounded sharded storage for backend-neutral NumPy tensor trees."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from dryml.records import RepresentationSpec


NUMPY_SEQUENCE_KIND = "dryml.numpy_sequence"
NUMPY_SEQUENCE_SCHEMA = "dryml.numpy_sequence.v1"
NUMPY_SEQUENCE_REPRESENTATION = RepresentationSpec.create(
    NUMPY_SEQUENCE_KIND,
    version="1",
    traits=("array-sequence", "tensor-tree", "stream-readable"),
    storage_kinds=("product-dir",),
    payload={"index": "index.json", "shards": "numpy-npz"},
)


class NumpySequenceCorruptError(RuntimeError):
    """Raised when sequence metadata or shard bytes fail strict validation."""


@dataclass(frozen=True, slots=True)
class NumpySequenceShard:
    """One contiguous row range in a NumPy sequence product."""

    index: int
    path: str
    start: int
    stop: int
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise NumpySequenceCorruptError("shard index must be non-negative")
        _validate_path(self.path)
        if type(self.start) is not int or type(self.stop) is not int:
            raise NumpySequenceCorruptError("shard row bounds must be integers")
        if self.start < 0 or self.stop <= self.start:
            raise NumpySequenceCorruptError("shard row bounds are invalid")
        if type(self.size) is not int or self.size < 0:
            raise NumpySequenceCorruptError("shard size must be non-negative")
        if not _is_digest(self.sha256):
            raise NumpySequenceCorruptError("shard digest is invalid")

    def to_json(self) -> dict[str, Any]:
        """Return strict JSON metadata for this shard."""

        return {
            "index": self.index,
            "path": self.path,
            "start": self.start,
            "stop": self.stop,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, value: Any) -> "NumpySequenceShard":
        """Decode one strict shard entry."""

        fields = {"index", "path", "start", "stop", "size", "sha256"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise NumpySequenceCorruptError("shard metadata fields are malformed")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class NumpySequenceIndex:
    """Compact sequence index independent of the Store product manifest."""

    count: int
    tree: Mapping[str, Any] | None
    shards: tuple[NumpySequenceShard, ...] = ()

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise NumpySequenceCorruptError("sequence count must be non-negative")
        if self.tree is not None:
            _validate_tree(self.tree)
        shards = tuple(
            item if isinstance(item, NumpySequenceShard) else NumpySequenceShard.from_json(item)
            for item in self.shards
        )
        expected_row = 0
        for index, shard in enumerate(shards):
            if shard.index != index or shard.start != expected_row:
                raise NumpySequenceCorruptError("shards are not contiguous and ordered")
            expected_row = shard.stop
        if expected_row != self.count:
            raise NumpySequenceCorruptError("shard rows do not match sequence count")
        if self.count and self.tree is None:
            raise NumpySequenceCorruptError("non-empty sequence has no tensor tree")
        object.__setattr__(self, "shards", shards)

    def to_json(self) -> dict[str, Any]:
        """Return the versioned compact index document."""

        return {
            "schema": NUMPY_SEQUENCE_SCHEMA,
            "schema_version": 1,
            "count": self.count,
            "tree": self.tree,
            "shards": [item.to_json() for item in self.shards],
        }

    def to_bytes(self) -> bytes:
        """Return deterministic UTF-8 JSON bytes."""

        return json.dumps(
            self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")

    @classmethod
    def from_json(cls, value: Any) -> "NumpySequenceIndex":
        """Decode and validate a strict index document."""

        fields = {"schema", "schema_version", "count", "tree", "shards"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise NumpySequenceCorruptError("sequence index fields are malformed")
        if value.get("schema") != NUMPY_SEQUENCE_SCHEMA or value.get("schema_version") != 1:
            raise NumpySequenceCorruptError("sequence index schema is unsupported")
        shards = value.get("shards")
        if not isinstance(shards, list):
            raise NumpySequenceCorruptError("sequence shards must be an array")
        return cls(
            count=value.get("count"),
            tree=value.get("tree"),
            shards=tuple(NumpySequenceShard.from_json(item) for item in shards),
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "NumpySequenceIndex":
        """Decode index bytes and normalize all parse failures."""

        try:
            value = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise NumpySequenceCorruptError("sequence index is not valid UTF-8 JSON") from exc
        return cls.from_json(value)


def write_numpy_sequence(
    elements: Iterable[Any],
    root: str | Path,
    *,
    shard_rows: int = 1024,
    shard_bytes: int = 64 * 1024 * 1024,
) -> NumpySequenceIndex:
    """Write a bounded sequence product beneath *root*.

    Memory is bounded by one configured row buffer and encoded shard. The file
    count is one compact index plus one file per shard, never one file per row.
    """

    product_root = Path(root)
    product_root.mkdir(parents=True, exist_ok=True)

    def write_file(path: str, payload: bytes) -> None:
        target = product_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    return write_numpy_sequence_stream(
        elements,
        write_file,
        shard_rows=shard_rows,
        shard_bytes=shard_bytes,
    )


def write_numpy_sequence_stream(
    elements: Iterable[Any],
    write_file: Callable[[str, bytes], None],
    *,
    shard_rows: int = 1024,
    shard_bytes: int = 64 * 1024 * 1024,
    prior: NumpySequenceIndex | None = None,
    on_flush: Callable[[NumpySequenceIndex], None] | None = None,
) -> NumpySequenceIndex:
    """Encode rows incrementally through a caller-owned durable write callback.

    ``prior`` names already committed contiguous shards during exact resume.
    ``on_flush`` runs only after all shards produced from a buffer are durable,
    so a pipeline cursor captured there cannot advance beyond its bytes.
    """

    _validate_limits(shard_rows, shard_bytes)
    existing = prior or NumpySequenceIndex(0, None, ())
    tree = existing.tree
    shards = list(existing.shards)
    row = existing.count
    buffer: list[tuple[np.ndarray, ...]] = []
    buffered_bytes = 0

    def flush() -> None:
        nonlocal buffer, buffered_bytes, row, tree
        if not buffer:
            return
        for rows, payload in _encode_bounded(buffer, shard_bytes):
            path = f"shards/{len(shards):08d}.npz"
            shard = NumpySequenceShard(
                index=len(shards),
                path=path,
                start=row,
                stop=row + rows,
                size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            write_file(path, payload)
            shards.append(shard)
            row += rows
        buffer = []
        buffered_bytes = 0
        if on_flush is not None:
            on_flush(NumpySequenceIndex(row, tree, tuple(shards)))

    for element in elements:
        element_tree, leaves = _flatten_element(element)
        if tree is None:
            tree = element_tree
        elif element_tree != tree:
            raise TypeError("NumPy sequence elements must have one stable tensor-tree structure")
        leaf_bytes = sum(item.nbytes for item in leaves)
        if buffer and (len(buffer) >= shard_rows or buffered_bytes + leaf_bytes > shard_bytes):
            flush()
        buffer.append(leaves)
        buffered_bytes += leaf_bytes
        if len(buffer) >= shard_rows:
            flush()
    flush()
    index = NumpySequenceIndex(row, tree, tuple(shards))
    write_file("index.json", index.to_bytes())
    return index


def iter_numpy_sequence(root: str | Path) -> Iterator[Any]:
    """Yield rows lazily after validating the compact index and every shard."""

    for tree, shard, arrays in iter_numpy_sequence_partitions(root):
        for offset in range(shard.stop - shard.start):
            leaves = tuple(array[offset] for array in arrays)
            yield _rebuild_element(tree, iter(leaves))


def read_numpy_sequence_index(root: str | Path) -> NumpySequenceIndex:
    """Read and validate only the compact NumPy sequence index."""

    product_root = Path(root)
    try:
        return NumpySequenceIndex.from_bytes((product_root / "index.json").read_bytes())
    except NumpySequenceCorruptError:
        raise
    except Exception as exc:
        raise NumpySequenceCorruptError("sequence index is missing or unreadable") from exc


def iter_numpy_sequence_partitions(root: str | Path):
    """Yield one validated decoded shard at a time for streaming adapters."""

    product_root = Path(root)
    index = read_numpy_sequence_index(product_root)
    for shard in index.shards:
        path = product_root / shard.path
        try:
            payload = path.read_bytes()
        except Exception as exc:
            raise NumpySequenceCorruptError(f"sequence shard {shard.path!r} is missing") from exc
        if len(payload) != shard.size or hashlib.sha256(payload).hexdigest() != shard.sha256:
            raise NumpySequenceCorruptError(f"sequence shard {shard.path!r} failed integrity validation")
        arrays = _decode_shard(payload, index.tree, shard.stop - shard.start)
        yield index.tree, shard, arrays


def _flatten_element(value: Any) -> tuple[Mapping[str, Any], tuple[np.ndarray, ...]]:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("NumPy sequence mapping keys must be strings")
        children = []
        leaves = []
        for key, item in value.items():
            child, child_leaves = _flatten_element(item)
            children.append([key, child])
            leaves.extend(child_leaves)
        return {"kind": "dict", "children": children}, tuple(leaves)
    if isinstance(value, tuple):
        return _flatten_sequence("tuple", value)
    if isinstance(value, list):
        return _flatten_sequence("list", value)
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("NumPy sequence leaves cannot use object dtype")
    tree = {"kind": "leaf", "dtype": array.dtype.str, "shape": list(array.shape)}
    return tree, (array,)


def _flatten_sequence(kind: str, values: Sequence[Any]):
    children = []
    leaves = []
    for value in values:
        child, child_leaves = _flatten_element(value)
        children.append(child)
        leaves.extend(child_leaves)
    return {"kind": kind, "children": children}, tuple(leaves)


def _encode_bounded(
    rows: list[tuple[np.ndarray, ...]],
    shard_bytes: int,
) -> list[tuple[int, bytes]]:
    payload = _encode_shard(rows)
    if len(payload) <= shard_bytes or len(rows) == 1:
        return [(len(rows), payload)]
    midpoint = len(rows) // 2
    return [
        *_encode_bounded(rows[:midpoint], shard_bytes),
        *_encode_bounded(rows[midpoint:], shard_bytes),
    ]


def _encode_shard(rows: list[tuple[np.ndarray, ...]]) -> bytes:
    leaf_count = len(rows[0])
    if any(len(row) != leaf_count for row in rows):
        raise TypeError("NumPy sequence leaf count changed")
    arrays = {}
    for index in range(leaf_count):
        first = rows[0][index]
        if any(item[index].dtype != first.dtype or item[index].shape != first.shape for item in rows):
            raise TypeError("NumPy sequence leaf dtype or shape changed")
        arrays[f"leaf_{index:04d}"] = np.stack([item[index] for item in rows], axis=0)
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _decode_shard(
    payload: bytes,
    tree: Mapping[str, Any] | None,
    rows: int,
) -> tuple[np.ndarray, ...]:
    expected_leaves = tuple(_leaf_specs(tree))
    expected_keys = tuple(f"leaf_{index:04d}" for index in range(len(expected_leaves)))
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(sorted(archive.files)) != expected_keys:
                raise NumpySequenceCorruptError("sequence shard leaf set is inconsistent")
            arrays = tuple(np.asarray(archive[key]) for key in expected_keys)
    except NumpySequenceCorruptError:
        raise
    except Exception as exc:
        raise NumpySequenceCorruptError("sequence shard is not a valid NumPy archive") from exc
    for array, (dtype, shape) in zip(arrays, expected_leaves):
        if array.shape != (rows, *shape) or array.dtype != np.dtype(dtype):
            raise NumpySequenceCorruptError("sequence shard leaf metadata is inconsistent")
    return arrays


def _leaf_specs(tree: Mapping[str, Any] | None):
    if tree is None:
        return
    kind = tree["kind"]
    if kind == "leaf":
        yield tree["dtype"], tuple(tree["shape"])
    elif kind == "dict":
        for _key, child in tree["children"]:
            yield from _leaf_specs(child)
    else:
        for child in tree["children"]:
            yield from _leaf_specs(child)


def _rebuild_element(tree: Mapping[str, Any] | None, leaves: Iterator[Any]) -> Any:
    if tree is None:
        raise NumpySequenceCorruptError("cannot rebuild an element without tensor-tree metadata")
    kind = tree["kind"]
    if kind == "leaf":
        return next(leaves)
    if kind == "dict":
        return {key: _rebuild_element(child, leaves) for key, child in tree["children"]}
    values = [_rebuild_element(child, leaves) for child in tree["children"]]
    return tuple(values) if kind == "tuple" else values


def _validate_tree(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise NumpySequenceCorruptError("tensor tree node must be an object")
    kind = value.get("kind")
    if kind == "leaf":
        if set(value) != {"kind", "dtype", "shape"}:
            raise NumpySequenceCorruptError("tensor leaf fields are malformed")
        try:
            np.dtype(value["dtype"])
        except Exception as exc:
            raise NumpySequenceCorruptError("tensor leaf dtype is invalid") from exc
        if not isinstance(value["shape"], list) or any(type(dim) is not int or dim < 0 for dim in value["shape"]):
            raise NumpySequenceCorruptError("tensor leaf shape is invalid")
        return
    if kind not in {"dict", "tuple", "list"} or set(value) != {"kind", "children"}:
        raise NumpySequenceCorruptError("tensor container fields are malformed")
    children = value["children"]
    if not isinstance(children, list):
        raise NumpySequenceCorruptError("tensor container children must be an array")
    if kind == "dict":
        keys = []
        for item in children:
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                raise NumpySequenceCorruptError("tensor mapping child is malformed")
            keys.append(item[0])
            _validate_tree(item[1])
        if len(set(keys)) != len(keys):
            raise NumpySequenceCorruptError("tensor mapping keys are duplicated")
    else:
        for child in children:
            _validate_tree(child)


def _validate_limits(shard_rows: int, shard_bytes: int) -> None:
    if type(shard_rows) is not int or shard_rows < 1:
        raise ValueError("shard_rows must be a positive integer")
    if type(shard_bytes) is not int or shard_bytes < 1:
        raise ValueError("shard_bytes must be a positive integer")


def _validate_path(path: str) -> None:
    if not isinstance(path, str):
        raise NumpySequenceCorruptError("shard path must be a string")
    pure = PurePosixPath(path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise NumpySequenceCorruptError("shard path is unsafe")


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


__all__ = [
    "NUMPY_SEQUENCE_KIND",
    "NUMPY_SEQUENCE_REPRESENTATION",
    "NumpySequenceCorruptError",
    "NumpySequenceIndex",
    "NumpySequenceShard",
    "iter_numpy_sequence",
    "iter_numpy_sequence_partitions",
    "read_numpy_sequence_index",
    "write_numpy_sequence",
    "write_numpy_sequence_stream",
]
