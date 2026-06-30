"""Canonical JSON serialization helpers for environment metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .errors import EnvironmentSerializationError


def _json_ready(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        value = dict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_json_ready(item) for item in sorted(value, key=repr)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_ready(item) for item in value]
    return value


def canonical_json_dumps(data: Any) -> str:
    """Return a stable JSON string for JSON-compatible data.

    Dictionaries are sorted by key and separators are fixed so the output is
    suitable for content-addressed IDs and exact serialization tests.
    """

    try:
        return json.dumps(
            _json_ready(data),
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
    """Return an immutable shallow copy of a mapping with deterministic keys."""

    return MappingProxyType({str(key): mapping[key] for key in sorted(mapping or {}, key=str)})


__all__ = ["canonical_json_dumps", "canonical_json_bytes", "freeze_mapping"]
