import hashlib
import struct
from inspect import isclass
from datetime import date, datetime, time
from decimal import Decimal
from uuid import UUID
from pathlib import Path
from enum import Enum

import numpy as np

from .general import get_definition_view, is_dictlike
from boltons.iterutils import remap, is_collection, default_enter

# ---------- ENTER HOOK FOR Definition ----------

def stable_definition_enter(path, key, value):
    from ..definition import Definition
    """Treat Definition objects via their normalized 'view'."""
    if isinstance(value, Definition):
        # Hash the normalized view instead of the raw Definition
        return {}, get_definition_view(value)
    else:
        return default_enter(path, key, value)


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


# ---------- CONTAINER DETECTION ----------

def _is_container(value) -> bool:
    """
    Container in the sense of remap traversal.
    Note: np.ndarray is explicitly *not* treated as a container here.
    """
    return (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray)


def _stable_hash_any(value) -> str:
    """
    Hash arbitrary value: dispatch to leaf or full structural hash.
    """
    if _is_container(value):
        return stable_hash_function(value)
    else:
        return stable_hash_value(value)


# ---------- MAIN STRUCTURAL HASH ----------

def stable_hash_function(structure) -> str:
    """
    Deterministic structural hash of an arbitrary (possibly nested) Python structure.

    - Dicts / mappings: order-independent but key-sensitive.
    - Sets / frozensets: order-independent.
    - Lists / tuples: order-dependent.
    - np.ndarray: hashed by shape + dtype + contents.
    - Definition: hashed via get_definition_view().
    """

    # Handle scalar / leaf roots directly (remap would not call exit() on them)
    if not _is_container(structure):
        return stable_hash_value(structure)

    class HashHelper:
        """
        Wrapper used during traversal so that parent nodes can distinguish:
        - a fully-hashed child container (HashHelper)
        - a raw value
        """

        __slots__ = ("hash",)

        def __init__(self, the_hash: str):
            self.hash = the_hash

    def _visit(path, key, value):
        # If child is already a helper, unwrap to its digest so parents see strings
        if isinstance(value, HashHelper):
            return key, value.hash

        # Traverse into containers (except np.ndarray which is a leaf)
        if _is_container(value):
            return key, value

        # Leaf -> deterministic digest string
        return key, stable_hash_value(value)

    def _exit(path, key, old_parent, new_parent, new_items):
        # new_items: list[(child_key, child_digest_or_container)]

        # For mappings and sets, we want order-independent hashing:
        if is_dictlike(old_parent) or isinstance(old_parent, (set, frozenset)):
            # sort by a deterministic token derived from the key
            new_items = sorted(
                new_items,
                key=lambda kv: _stable_hash_any(kv[0]),
            )
        # For lists/tuples/etc we keep traversal order (insertion order).

        hasher = hashlib.sha256()

        # Include container type and length so different containers don't collide
        type_marker = f"{type(old_parent).__module__}.{type(old_parent).__qualname__}"
        hasher.update(b"T" + type_marker.encode("utf-8"))
        hasher.update(b"|" + str(len(new_items)).encode("ascii"))

        for child_key, child_val in new_items:
            # Hash the key structurally (so keys are part of the identity)
            key_digest = _stable_hash_any(child_key)
            hasher.update(b"K")
            hasher.update(key_digest.encode("ascii"))

            # child_val should already be a digest string at this point
            if not isinstance(child_val, str):
                raise TypeError(
                    f"Expected child digest string in _exit, got {type(child_val)!r}"
                )

            hasher.update(b"V")
            hasher.update(child_val.encode("ascii"))

        return HashHelper(hasher.hexdigest())

    result = remap(
        structure,
        enter=stable_definition_enter,
        visit=_visit,
        exit=_exit,
    )

    # For container roots, we expect a HashHelper; for leaf roots we handled above.
    if isinstance(result, HashHelper):
        return result.hash
    else:
        # Fallback (in case remap ever returns a raw value for some reason)
        return _stable_hash_any(result)
