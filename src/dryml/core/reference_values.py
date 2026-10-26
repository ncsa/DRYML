"""Namespace-scoped UUID identities for stateful Node occurrences.

Identity allocation and decoding perform no construction, Store IO, or state
selection. Legacy compound graph references are no longer part of this module;
identities and state selections belong to Node-local facets.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import re
from typing import Any, Iterator
from uuid import UUID, uuid4


_NAMESPACE = ContextVar("dryml_object_namespace", default=())
_NAMESPACE_PART = re.compile(r"[A-Za-z0-9]+\Z")
_STATE_HASH = re.compile(r"[A-Za-z0-9]{1,32}-[0-9a-f]{64}\Z")
_MAX_NAMESPACE_DEPTH = 16
_MAX_NAMESPACE_PART_LENGTH = 64
_MAX_NAMESPACE_LENGTH = 256


def _normalize_namespace(namespace: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(namespace, (tuple, list)):
        raise TypeError("ObjectId namespace must be a tuple or list of strings.")
    if len(namespace) > _MAX_NAMESPACE_DEPTH:
        raise ValueError(f"ObjectId namespace has more than {_MAX_NAMESPACE_DEPTH} parts.")
    parts = []
    for part in namespace:
        if not isinstance(part, str):
            raise TypeError("ObjectId namespace parts must be strings.")
        if not _NAMESPACE_PART.fullmatch(part):
            raise ValueError("ObjectId namespace parts must be non-empty ASCII alphanumeric strings.")
        if len(part) > _MAX_NAMESPACE_PART_LENGTH:
            raise ValueError(f"ObjectId namespace parts must be at most {_MAX_NAMESPACE_PART_LENGTH} characters.")
        parts.append(part)
    if sum(len(part) for part in parts) > _MAX_NAMESPACE_LENGTH:
        raise ValueError(f"ObjectId namespace must be at most {_MAX_NAMESPACE_LENGTH} characters.")
    return tuple(parts)


@dataclass(frozen=True, slots=True, init=False)
class ObjectId:
    """Immutable namespace and UUID identity for one stateful occurrence.

    Args:
        namespace (tuple[str, ...] | list[str] | None): New identity namespace.
            None inherits object_namespace(); () explicitly selects no namespace.
    Raises:
        TypeError: If namespace components have unsupported types.
        ValueError: If namespace text or size is outside the supported bounds.
    Side Effects:
        Allocates a fresh UUID4 nonce. No Store lookup, publication, construction,
        or reservation occurs. Use from_data() to preserve a recorded nonce.
    """

    namespace: tuple[str, ...]
    nonce: UUID

    def __init__(self, namespace: tuple[str, ...] | list[str] | None = None):
        selected = _NAMESPACE.get() if namespace is None else namespace
        object.__setattr__(self, "namespace", _normalize_namespace(selected))
        object.__setattr__(self, "nonce", uuid4())

    @classmethod
    def _trusted(cls, namespace: tuple[str, ...] | list[str], nonce: UUID) -> "ObjectId":
        if not isinstance(nonce, UUID):
            raise TypeError("ObjectId nonce must be a UUID.")
        result = object.__new__(cls)
        object.__setattr__(result, "namespace", _normalize_namespace(namespace))
        object.__setattr__(result, "nonce", nonce)
        return result

    @classmethod
    def from_data(cls, data: Any) -> "ObjectId":
        """Decode a closed namespace/nonce mapping, preserving the full UUID.

        Returns an immutable ObjectId without allocating a nonce. Unsupported
        field types raise TypeError; malformed fields or UUIDs raise ValueError.
        No external state is read or changed.
        """
        if not isinstance(data, Mapping) or set(data) != {"namespace", "nonce"}:
            raise ValueError("ObjectId records require exactly namespace and nonce fields.")
        try:
            nonce = UUID(data["nonce"])
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("ObjectId nonce must be a full UUID string.") from error
        return cls._trusted(data["namespace"], nonce)

    def to_data(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible namespace/nonce mapping without effects."""
        return {"namespace": list(self.namespace), "nonce": str(self.nonce)}

    def __stable_leaf_bytes__(self) -> bytes:
        return json.dumps(("object-id", self.namespace, self.nonce.hex),
                          separators=(",", ":"), ensure_ascii=True).encode("ascii")

    def __str__(self) -> str:
        prefix = "/".join(self.namespace)
        return f"{prefix + '~' if prefix else '~'}{self.nonce.hex[:12]}..."


@contextmanager
def object_namespace(*parts: str) -> Iterator[tuple[str, ...]]:
    """Append validated namespace parts within a thread/task-local context.

    Args:
        *parts (str): ASCII alphanumeric namespace components.
    Yields:
        tuple[str, ...]: Effective immutable namespace for new ObjectIds.
    Raises:
        TypeError: If a component is not a string.
        ValueError: If the combined namespace violates text or size limits.
    Side Effects:
        Installs a ContextVar value and restores it on normal or exceptional exit.
        Existing identities are unchanged; entering the context allocates no ID.
    """
    namespace = _normalize_namespace(_NAMESPACE.get() + tuple(parts))
    token = _NAMESPACE.set(namespace)
    try:
        yield namespace
    finally:
        _NAMESPACE.reset(token)


def _validate_state_hash(value: str) -> None:
    if not isinstance(value, str) or not _STATE_HASH.fullmatch(value):
        raise ValueError("State hashes must be '<codec>-<64 lowercase hexadecimal digest>' values.")
