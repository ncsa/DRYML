"""Canonical JSON serialization helpers for environment metadata."""

from __future__ import annotations

import json
import math
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
        return {key: json_ready(value[key]) for key in _sorted_string_keys(value)}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, set | frozenset):
        return [json_ready(item) for item in sorted(value, key=repr)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise EnvironmentSerializationError(
            "environment metadata floats must be finite",
            context={"value": repr(value)},
        )
    return value


def deep_freeze_json(value: Any) -> FrozenJson:
    """Return deeply immutable canonical JSON metadata.

    Mappings are keyed by strings in deterministic order, sequences become
    tuples, sets are sorted, and non-JSON values are rejected before they can
    participate in a content-addressed ID.
    """

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: deep_freeze_json(value[key]) for key in _sorted_string_keys(value)}
        )
    if isinstance(value, list | tuple):
        return tuple(deep_freeze_json(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(deep_freeze_json(item) for item in sorted(value, key=repr))
    if isinstance(value, float) and not math.isfinite(value):
        raise EnvironmentSerializationError(
            "environment metadata floats must be finite",
            context={"value": repr(value)},
        )
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
    """Return an immutable shallow copy with deterministic string keys.

    Use this only for mappings whose values are already immutable domain
    objects. Use :func:`deep_freeze_json` for arbitrary JSON metadata.
    """

    return MappingProxyType({str(key): mapping[key] for key in sorted(mapping or {}, key=str)})


def _sorted_string_keys(mapping: Mapping[Any, Any]) -> tuple[str, ...]:
    keys = []
    for key in mapping:
        if not isinstance(key, str):
            raise EnvironmentSerializationError(
                "environment metadata mapping keys must be strings",
                context={"key": repr(key), "type": type(key).__name__},
            )
        keys.append(key)
    return tuple(sorted(keys))


__all__ = [
    "FrozenJson",
    "canonical_json_dumps",
    "canonical_json_bytes",
    "deep_freeze_json",
    "freeze_mapping",
    "json_ready",
]
