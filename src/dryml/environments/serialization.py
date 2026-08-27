"""Environment-facing adapters for the shared bounded v1.1 codec."""

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


def json_ready(value: Any, **bounds: Any) -> Any:
    """Return a detached mutable JSON projection after bounded validation."""

    return _convert_error(_json_ready, value, **bounds)


def deep_freeze_json(value: Any, **bounds: Any) -> FrozenJson:
    """Return a deeply immutable bounded JSON-compatible projection."""

    return _convert_error(_deep_freeze_json, value, **bounds)


def canonical_json_dumps(data: Any, **bounds: Any) -> str:
    """Return compact deterministic JSON text after bounded validation."""

    return _convert_error(_canonical_json_dumps, data, **bounds)


def canonical_json_loads(text: str, **bounds: Any) -> Any:
    """Decode duplicate-aware JSON text into an immutable projection."""

    return _convert_error(_canonical_json_loads, text, **bounds)


def canonical_json_bytes(data: Any, **bounds: Any) -> bytes:
    """Return compact deterministic UTF-8 JSON after bounded validation."""

    return _convert_error(_canonical_json_bytes, data, **bounds)


def canonical_json_load_bytes(data: bytes, **bounds: Any) -> Any:
    """Decode UTF-8 JSON bytes into an immutable bounded projection."""

    return _convert_error(_canonical_json_load_bytes, data, **bounds)


def freeze_mapping(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable deterministic copy of a string-keyed mapping."""

    return _convert_error(_freeze_mapping, mapping)


def _convert_error(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except CanonicalJSONError as exc:
        message = str(exc)
        if message.startswith("canonical JSON"):
            message = "environment metadata" + message[len("canonical JSON"):]
        raise EnvironmentSerializationError(message, context=exc.context) from exc


__all__ = [
    "FrozenJson",
    "JsonPrimitive",
    "canonical_json_bytes",
    "canonical_json_dumps",
    "canonical_json_load_bytes",
    "canonical_json_loads",
    "deep_freeze_json",
    "freeze_mapping",
    "json_ready",
]
