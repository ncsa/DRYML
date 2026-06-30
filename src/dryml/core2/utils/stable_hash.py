import hashlib
import struct
from inspect import isclass
from datetime import date, datetime, time
from decimal import Decimal
from uuid import UUID
from pathlib import Path
from enum import Enum

import numpy as np

from .types import is_dictlike, is_collection
from .recurse import cycle_detect
from .graph import GraphHasher, GraphCtx

def stable_int_hash(s: str, *, bits: int = 64) -> int:
    # blake2b is fast and stable; digest_size controls output size
    digest = hashlib.blake2b(s.encode("utf-8"), digest_size=bits // 8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)

# ---------- LEAF ENCODING (DETERMINISTIC) ----------

def _stable_leaf_bytes(value) -> bytes:
    """
    Canonical byte representation for a single *leaf* value.
    No use of Python's built-in hash().
    """

    # Classes: use module + qualname
    if isclass(value):
        s = f"class:{value.__module__}.{value.__qualname__}"
        return s.encode("utf-8")

    # Enums: type + name
    if isinstance(value, Enum):
        s = f"enum:{type(value).__module__}.{type(value).__qualname__}:{value.name}"
        return s.encode("utf-8")

    # None
    if value is None:
        return b"N"

    # Booleans first (bool is a subclass of int)
    if isinstance(value, bool):
        return b"B1" if value else b"B0"

    # Plain ints
    if isinstance(value, int):
        # Stable decimal representation
        return b"I" + str(value).encode("ascii")

    # Floats (Python float = IEEE754 binary64)
    if isinstance(value, float):
        # binary representation to avoid repr/locale issues
        return b"F" + struct.pack(">d", value)

    # Strings
    if isinstance(value, str):
        return b"S" + value.encode("utf-8")

    # Raw bytes / byte-like
    if isinstance(value, (bytes, bytearray, memoryview)):
        return b"Y" + bytes(value)

    # NumPy arrays: include shape + dtype + raw bytes
    if isinstance(value, np.ndarray):
        return (
            b"A"
            + str(value.shape).encode("ascii")
            + b"|"
            + str(value.dtype).encode("ascii")
            + b"|"
            + value.tobytes()
        )

    # NumPy scalar
    if isinstance(value, np.generic):
        return (
            b"Ng"
            + str(value.dtype).encode("ascii")
            + b"|"
            + value.tobytes()
        )

    # Datetime-like
    if isinstance(value, (datetime, date, time)):
        # isoformat is deterministic for these
        return b"D" + value.isoformat().encode("ascii")

    # Decimal
    if isinstance(value, Decimal):
        # normalize to canonical form, then 'f' to avoid sci-notation variability
        return b"De" + format(value.normalize(), "f").encode("ascii")

    # UUID
    if isinstance(value, UUID):
        return b"U" + value.hex.encode("ascii")

    # Path
    if isinstance(value, Path):
        # string form is stable enough across runs
        return b"P" + str(value).encode("utf-8")

    # custom implemented stable hash
    if hasattr(value, "__stable_leaf_bytes__"):
        return value.__stable_leaf_bytes__()

    # If you want to support more types, add them explicitly above.
    # Falling back to repr() or pickle here *would* risk non-determinism.
    raise TypeError(f"Unsupported leaf type for stable hashing: {type(value)!r}")


def stable_hash_value(value) -> str:
    """
    Deterministic hash for a single leaf value (no containers).
    """
    return hashlib.sha256(_stable_leaf_bytes(value)).hexdigest()


# ----------------------------------------------------------------------
# graph hasher
# ----------------------------------------------------------------------

class StableHashGraphHasher(GraphHasher):
    def is_atomic(self, obj, ctx: GraphCtx) -> bool:
        from ..canonical import node_kind, NodeKind

        kind = node_kind(obj)
        return kind in {
            NodeKind.POD,
            NodeKind.TYPE,
            NodeKind.IDENTITY_VALUE,
            NodeKind.NDARRAY,
            NodeKind.FROZEN_NDARRAY,
            NodeKind.IMPORT_REF,
            NodeKind.SOURCE_SPEC,
        }

    def hash_atomic(self, obj, ctx: GraphCtx) -> str:
        return stable_hash_value(obj)

    def should_track_cycle(self, obj, ctx: GraphCtx) -> bool:
        from ..canonical import node_kind, NodeKind

        kind = node_kind(obj)
        return kind in {
            NodeKind.LIST,
            NodeKind.TUPLE,
            NodeKind.SET,
            NodeKind.DICT,
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT,
            NodeKind.DEFINITION,
            NodeKind.CONCRETE_DEFINITION,
            NodeKind.FROZEN_CONCRETE_DEFINITION,
            NodeKind.FROZEN_DEFINITION,
            NodeKind.OBJECT,
        }

    def dispatch(self, obj, ctx: GraphCtx) -> str:
        from ..canonical import node_kind, NodeKind
        from ..definition import Definition, ConcreteDefinition, FrozenConcreteDefinition, FrozenDefinition
        from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
        from ..object import Object

        kind = node_kind(obj)

        if kind is NodeKind.OBJECT:
            return self.hash(obj.definition, ctx.child("definition"))

        if kind in {NodeKind.LIST, NodeKind.FROZEN_LIST}:
            return self._hash_sequence("builtins.list", obj, ctx)

        if kind in {NodeKind.TUPLE, NodeKind.FROZEN_TUPLE}:
            return self._hash_sequence("builtins.tuple", obj, ctx)

        if kind in {NodeKind.SET, NodeKind.FROZEN_SET}:
            return self._hash_set("builtins.set", obj, ctx)

        if kind in {NodeKind.DICT, NodeKind.FROZEN_DICT}:
            return self._hash_mapping("builtins.dict", obj, ctx)

        if isinstance(obj, (Definition, ConcreteDefinition)):
            type_marker = f"{type(obj).__module__}.{type(obj).__qualname__}"
            items = [(k, obj[k]) for k in obj]
            return self._hash_mapping(type_marker, dict(items), ctx)

        if isinstance(obj, FrozenConcreteDefinition):
            type_marker = f"{type(obj).__module__}.{type(obj).__qualname__}"
            return self._hash_mapping(type_marker, {"target": obj.target}, ctx)

        if isinstance(obj, FrozenDefinition):
            type_marker = f"{type(obj).__module__}.{type(obj).__qualname__}"
            return self._hash_mapping(
                type_marker,
                {"cls": obj.cls, "args": obj.args, "kwargs": obj.kwargs},
                ctx,
            )

        raise TypeError(f"Unsupported type {type(obj)} for stable hashing")

    def _hash_sequence(self, type_marker: str, seq, ctx: GraphCtx) -> str:
        """
        Order-sensitive hashing.
        """
        hasher = hashlib.sha256()
        hasher.update(b"T" + type_marker.encode("utf-8"))
        hasher.update(b"|" + str(len(seq)).encode("ascii"))

        for i, v in enumerate(seq):
            child_hash = self.hash(v, ctx.child(i))
            hasher.update(b"I" + str(i).encode("ascii"))
            hasher.update(b"V" + child_hash.encode("utf-8"))

        return hasher.hexdigest()

    def _hash_set(self, type_marker: str, st, ctx: GraphCtx) -> str:
        """
        Order-insensitive hashing by child hash.
        """
        hasher = hashlib.sha256()
        hasher.update(b"T" + type_marker.encode("utf-8"))
        hasher.update(b"|" + str(len(st)).encode("ascii"))

        child_hashes = sorted(self.hash(v, ctx.child(i)) for i, v in enumerate(st))
        for child_hash in child_hashes:
            hasher.update(b"V" + child_hash.encode("utf-8"))

        return hasher.hexdigest()

    def _hash_mapping(self, type_marker: str, mp, ctx: GraphCtx) -> str:
        """
        Order-insensitive by key hash, key-sensitive, value-sensitive.
        """
        hasher = hashlib.sha256()
        hasher.update(b"T" + type_marker.encode("utf-8"))
        hasher.update(b"|" + str(len(mp)).encode("ascii"))

        key_val_hashes = []
        for k, v in mp.items():
            key_hash = self.hash(k, ctx.child("<key>"))
            val_hash = self.hash(v, ctx.child(k if isinstance(k, (str, int)) else str(k)))
            key_val_hashes.append((key_hash, val_hash))

        key_val_hashes.sort(key=lambda kv: kv[0])

        for key_hash, val_hash in key_val_hashes:
            hasher.update(b"K" + key_hash.encode("utf-8"))
            hasher.update(b"V" + val_hash.encode("utf-8"))

        return hasher.hexdigest()


def stable_hash_function(structure, cache=None) -> str:
    ctx = GraphCtx(memo={} if cache is None else cache)
    return StableHashGraphHasher().hash(structure, ctx)
