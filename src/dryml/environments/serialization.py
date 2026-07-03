"""Compatibility wrappers for environment metadata serialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.formats.canonical import (
    FrozenJson,
    JsonPrimitive,
    canonical_json_bytes as _canonical_json_bytes,
    canonical_json_dumps as _canonical_json_dumps,
    canonical_json_load_bytes as _canonical_json_load_bytes,
    canonical_json_loads as _canonical_json_loads,
    deep_freeze_json as _deep_freeze_json,
    freeze_mapping as _freeze_mapping,
    json_ready as _json_ready,
)
from dryml.formats.errors import CanonicalJSONError

from .errors import EnvironmentSerializationError


def json_ready(value: Any) -> Any:
    """Return a JSON-compatible mutable view of frozen/canonical metadata."""

    return _convert_error(_json_ready, value)


def deep_freeze_json(value: Any) -> FrozenJson:
    """Return deeply immutable canonical JSON metadata.

    Mappings are keyed by strings in deterministic order, sequences become
    tuples, sets are sorted, and non-JSON values are rejected before they can
    participate in a content-addressed ID.
    """

    return _convert_error(_deep_freeze_json, value)


def canonical_json_dumps(data: Any) -> str:
    """Return a stable JSON string for JSON-compatible data.

    Dictionaries are sorted by key and separators are fixed so the output is
    suitable for content-addressed IDs and exact serialization tests.
    """

    return _convert_error(_canonical_json_dumps, data)


def canonical_json_loads(text: str) -> Any:
    """Decode JSON text using environment serialization errors."""

    return _convert_error(_canonical_json_loads, text)


def canonical_json_bytes(data: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for content hashing."""

    return _convert_error(_canonical_json_bytes, data)


def canonical_json_load_bytes(data: bytes) -> Any:
    """Decode UTF-8 JSON bytes using environment serialization errors."""

    return _convert_error(_canonical_json_load_bytes, data)


def freeze_mapping(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable shallow copy with deterministic string keys.

    Use this only for mappings whose values are already immutable domain
    objects. Use :func:`deep_freeze_json` for arbitrary JSON metadata.
    """

    return _convert_error(_freeze_mapping, mapping)


def _convert_error(func, *args):
    try:
        return func(*args)
    except CanonicalJSONError as exc:
        raise EnvironmentSerializationError(_environment_message(str(exc)), context=exc.context) from exc


def _environment_message(message: str) -> str:
    if message.startswith("canonical JSON"):
        return "environment metadata" + message[len("canonical JSON"):]
    return message


__all__ = [
    "FrozenJson",
    "JsonPrimitive",
    "canonical_json_dumps",
    "canonical_json_bytes",
    "canonical_json_loads",
    "canonical_json_load_bytes",
    "deep_freeze_json",
    "freeze_mapping",
    "json_ready",
]
