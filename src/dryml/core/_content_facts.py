"""Internal primitive representation and owned-size rules for Repo content facts.

This dependency-light helper owns no cache, parser, domain values or resources.
Sizes overcount repeated immutable leaves and prospective tuple/container slots;
callers add source/context and facility bookkeeping before reserving storage.
"""

import sys
import weakref
from types import MappingProxyType


def _primitive_size(value):
    """Bound prospective owned primitive containers, without allocating copies."""
    if type(value) in (dict, MappingProxyType):
        return 128 + 16 * len(value) + sum(
            96 + sys.getsizeof(key) + _primitive_size(item)
            for key, item in value.items()
        )
    if type(value) in (list, tuple):
        return 128 + 16 * len(value) + sum(_primitive_size(item) for item in value)
    if type(value) in (str, bytes, int, float, bool, type(None), weakref.ReferenceType):
        return sys.getsizeof(value)
    raise TypeError("Content fact context/value is not an exact primitive.")


def _context_matches(left, right):
    """Compare closed keys without invoking user equality or reusing dead IDs."""
    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        return len(left) == len(right) and all(_context_matches(a, b) for a, b in zip(left, right))
    if type(left) is weakref.ReferenceType:
        target = left()
        return target is not None and target is right()
    return left == right


def _freeze_json_fact(value):
    if type(value) in (dict, MappingProxyType):
        return (0, tuple((key, _freeze_json_fact(item)) for key, item in value.items()))
    if type(value) in (list, tuple):
        return (1, tuple(_freeze_json_fact(item) for item in value))
    return value


def _thaw_json_fact(value):
    """Rebuild the existing canonical parser's detached mapping/tuple projection."""
    if type(value) is tuple:
        kind, items = value
        if kind == 0:
            return MappingProxyType({key: _thaw_json_fact(item) for key, item in items})
        return tuple(_thaw_json_fact(item) for item in items)
    return value
