import hashlib
import struct
from inspect import isclass
from datetime import date, datetime, time
from decimal import Decimal
from uuid import UUID
from pathlib import Path
from enum import Enum

import numpy as np

from .general import is_dictlike
from .recurse import cycle_detect
from boltons.iterutils import remap, is_collection, default_enter

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


# ---------- CONTAINER DETECTION ----------

def _is_container(value) -> bool:
    """
    Container in the sense of remap traversal.
    Note: np.ndarray is explicitly *not* treated as a container here.
    """
    return (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray)


# ---------- MAIN STRUCTURAL HASH ----------

def stable_hash_container(cont, cache=None) -> str:
    # For lists/tuples/etc we keep traversal order (insertion order).
    if cache is None:
        cache = {}

    hasher = hashlib.sha256()

    # Include container type and length so different containers don't collide
    # Canonicalize container markers so Frozen* hashes match their source types
    from ..freeze import FrozenList, FrozenDict, FrozenSet
    from ..definition import Definition, ConcreteDefinition
    if isinstance(cont, (list, FrozenList)):
        type_marker = "builtins.list"
        cont_iter = enumerate(cont)
    elif isinstance(cont, (tuple,)):
        type_marker = "builtins.tuple"
        cont_iter = enumerate(cont)
    elif isinstance(cont, (dict, FrozenDict)):
        type_marker = "builtins.dict"
        cont_iter = cont.items()
    elif isinstance(cont, (set, FrozenSet)):
        type_marker = "builtins.set"
        cont_iter = enumerate(cont)
    elif isinstance(cont, (Definition, ConcreteDefinition)):
        type_marker = f"{type(cont).__module__}.{type(cont).__qualname__}"
        keys = iter(cont)
        cont_iter = map(lambda k: (k, cont[k]), keys)
    else:
        raise TypeError(f"Unsupported container type {type(cont)} for stable_hash_container")
    hasher.update(b"T" + type_marker.encode("utf-8"))
    hasher.update(b"|" + str(len(cont)).encode("ascii"))

    key_val_list = list(cont_iter)
    # Get hash vals for keys and vals
    key_val_hash_list = list(map(
        lambda kv: (
            stable_hash_function(kv[0], cache=cache),
            stable_hash_function(kv[1], cache=cache),
        ),
        key_val_list,
    ))

    # Sort by key hash
    key_val_hash_list.sort(key=lambda kv: kv[0])

    for child_key_hash, child_val_hash in key_val_hash_list:
        # Hash the key structurally (so keys are part of the identity)
        hasher.update(b"K")
        hasher.update(child_key_hash.encode("utf-8"))

        hasher.update(b"V")
        hasher.update(child_val_hash.encode('utf-8'))

    return hasher.hexdigest()


@cycle_detect
def stable_hash_function(structure, cache=None) -> str:
    """
    Deterministic structural hash of an arbitrary (possibly nested) Python structure.

    - Dicts / mappings: order-independent but key-sensitive.
    - Sets / frozensets: order-independent.
    - Lists / tuples: order-dependent.
    - np.ndarray: hashed by shape + dtype + contents.
    - Definition: hashed via get_definition_view().
    """
    from ..object import Object

    if cache is None:
        cache = {}

    if id(structure) in cache:
        return cache[id(structure)]

    if _is_container(structure):
        hash_value = stable_hash_container(structure, cache=cache)
        cache[id(structure)] = hash_value
        return hash_value
    elif isinstance(structure, Object):
        hash_value = stable_hash_function(structure.definition)
        cache[id(structure)] = hash_value
        return hash_value
    else:
        hash_value = stable_hash_value(structure)
        cache[id(structure)] = hash_value
        return hash_value
