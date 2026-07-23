from __future__ import annotations

from .definition import ConcreteDefinition


def same_cdef(left: ConcreteDefinition, right: ConcreteDefinition) -> bool:
    if left is right:
        return True
    if left.stable_hash() != right.stable_hash():
        return False
    try:
        return left == right
    except TypeError:
        return False
