from __future__ import annotations

from inspect import isclass
from typing import Mapping, Iterable, Iterator, Any


_ATOMIC_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    bytes,
    str,
)


def is_nonclass_callable(obj):
    return callable(obj) and not isclass(obj)


def is_dictlike(val):
    return isinstance(val, Mapping)


def is_collection(val) -> bool:
    """
    True for non-mapping iterables (lists, tuples, sets, generators, etc.),
    but False for string/bytes-like objects and mappings.
    """
    if val is None:
        return False

    # Treat these as scalars/leaves, not collections
    if isinstance(val, (str, bytes, bytearray, memoryview)):
        return False

    # Leave mappings to is_dictlike()
    if isinstance(val, Mapping):
        return False

    # Iterable => collection
    return isinstance(val, Iterable)


def is_stream(obj) -> bool:
    return isinstance(obj, io.IOBase)


def is_iterator(obj:Any) -> bool:
    return isinstance(obj, Iterator)


def is_namedtuple(x: Any) -> bool:
    """
    True for instances of collections.namedtuple subclasses.

    We intentionally check the type for `_fields` so this works for
    standard namedtuple instances while avoiding false positives from
    arbitrary tuple-like objects.
    """
    return isinstance(x, tuple) and hasattr(type(x), "_fields")
