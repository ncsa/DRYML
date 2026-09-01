"""Passive immutable key/value carriers for live annotation targets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import AnnotationValidationError

_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_MAX_KEY_LENGTH = 128


def _validate_key(key: str) -> str:
    """Validate one consumer-selected annotation key.

    Args:
        key: ASCII identifier used for exact collection filtering.

    Returns:
        The validated key.

    Raises:
        AnnotationValidationError: If the key is not a 1-128 character ASCII
            identifier matching the kernel grammar.
    """

    if (
        not isinstance(key, str)
        or not 1 <= len(key) <= _MAX_KEY_LENGTH
        or not key.isascii()
        or _KEY_PATTERN.fullmatch(key) is None
    ):
        raise AnnotationValidationError("annotation key is invalid", context={"key": key})
    return key


@dataclass(frozen=True, slots=True, eq=False)
class Annotation:
    """One process-local, identity-based consumer annotation entry.

    Args:
        key: A validated consumer-selected classification key.
        value: An opaque consumer-owned value retained without copying,
            comparison, hashing, serialization, or deep-freezing.

    Raises:
        AnnotationValidationError: If ``key`` does not satisfy the generic
            annotation-key grammar.

    Side Effects:
        None. Values remain process-local until attached to a target by the
        separate attachment API.
    """

    key: str
    value: Any

    def __post_init__(self) -> None:
        """Validate the carrier key without inspecting the opaque value."""

        _validate_key(self.key)


__all__ = ["Annotation"]
