from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..cdef_identity import same_cdef
from ..definition import ConcreteDefinition


def cdef_equal(left: ConcreteDefinition, right: ConcreteDefinition) -> bool:
    return same_cdef(left, right)


def stable_hash_to_blob(stable_hash: str) -> bytes:
    try:
        return bytes.fromhex(stable_hash)
    except ValueError as exc:
        raise ValueError(f"Stable hash must be hexadecimal, got {stable_hash!r}.") from exc


def stable_hash_from_blob(blob: bytes) -> str:
    if not isinstance(blob, bytes):
        raise TypeError(f"Stable hash blob must be bytes, got {type(blob).__name__}.")
    return blob.hex()


def chunked(values: Iterable, size: int) -> Iterator[tuple]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    chunk = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield tuple(chunk)
            chunk = []
    if chunk:
        yield tuple(chunk)
