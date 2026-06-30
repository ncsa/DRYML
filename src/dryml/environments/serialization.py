"""Canonical JSON serialization helpers for environment metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .errors import EnvironmentSerializationError


JsonPrimitive = str | int | float | bool | None
FrozenJson = JsonPrimitive | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


def json_ready(value: Any) -> Any:
    """Return a JSON-compatible mutable view of frozen/canonical metadata."""

    if isinstance(value, MappingProxyType):
        value = dict(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, set | frozenset):
        return [json_ready(item) for item in sorted(value, key=repr)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [json_ready(item) for item in value]
    return value


def deep_freeze_json(value: Any) -> FrozenJson:
    """Return deeply immutable canonical JSON metadata.

    Mappings are keyed by strings in deterministic order, sequences become
    tuples, sets are sorted, and non-JSON values are rejected before they can
    participate in a content-addressed ID.
    """

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): deep_freeze_json(value[key]) for key in sorted(value, key=str)}
        )
    if isinstance(value, list | tuple):
        return tuple(deep_freeze_json(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(deep_freeze_json(item) for item in sorted(value, key=repr))
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise EnvironmentSerializationError(
        f"environment metadata value {value!r} is not JSON serializable",
        context={"type": type(value).__name__},
    )


def canonical_json_dumps(data: Any) -> str:
    """Return a stable JSON string for JSON-compatible data.

    Dictionaries are sorted by key and separators are fixed so the output is
    suitable for content-addressed IDs and exact serialization tests.
    """

    try:
        return json.dumps(
            json_ready(data),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EnvironmentSerializationError(
            "environment metadata is not JSON serializable",
            context={"error": str(exc)},
        ) from exc


def canonical_json_bytes(data: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for content hashing."""

    return canonical_json_dumps(data).encode("utf-8")


def freeze_mapping(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable copy of a mapping with deterministic string keys."""

    return MappingProxyType({str(key): mapping[key] for key in sorted(mapping or {}, key=str)})


__all__ = [
    "FrozenJson",
    "canonical_json_dumps",
    "canonical_json_bytes",
    "deep_freeze_json",
    "freeze_mapping",
    "json_ready",
]
