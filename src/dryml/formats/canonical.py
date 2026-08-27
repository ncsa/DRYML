"""Bounded, duplicate-aware canonical JSON primitives for v1.1 metadata."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .errors import CanonicalJSONError

DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_NODES = 1024
DEFAULT_MAX_ENTRIES = 64
DEFAULT_MAX_STRING = 4096
DEFAULT_MAX_INT_BITS = 4096

JsonPrimitive = str | int | float | bool | None
FrozenJson = JsonPrimitive | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


def deep_freeze_json(
    value: Any,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_string: int = DEFAULT_MAX_STRING,
    max_int_bits: int = DEFAULT_MAX_INT_BITS,
) -> FrozenJson:
    """Validate and detach a bounded JSON value into an immutable projection.

    Mapping keys must be strings; sequences preserve their order.  Sets are
    intentionally rejected because they have no JSON meaning.  Bounds apply to
    the complete value, including scalar nodes.
    """

    nodes = [0]

    def freeze(item: Any, depth: int) -> FrozenJson:
        nodes[0] += 1
        if nodes[0] > max_nodes:
            raise CanonicalJSONError("canonical JSON exceeds node bound", context={"limit": max_nodes})
        if depth > max_depth:
            raise CanonicalJSONError("canonical JSON exceeds depth bound", context={"limit": max_depth})
        if isinstance(item, Mapping):
            if len(item) > max_entries:
                raise CanonicalJSONError("canonical JSON mapping exceeds entry bound", context={"limit": max_entries})
            keys = []
            for key in item:
                if not isinstance(key, str):
                    raise CanonicalJSONError("canonical JSON mapping keys must be strings", context={"type": type(key).__name__})
                check_string(key, max_string)
                keys.append(key)
            return MappingProxyType({key: freeze(item[key], depth + 1) for key in sorted(keys)})
        if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            if len(item) > max_entries:
                raise CanonicalJSONError("canonical JSON sequence exceeds entry bound", context={"limit": max_entries})
            return tuple(freeze(child, depth + 1) for child in item)
        if isinstance(item, str):
            check_string(item, max_string)
            return item
        if isinstance(item, bool) or item is None:
            return item
        if isinstance(item, int):
            if item.bit_length() > max_int_bits:
                raise CanonicalJSONError("canonical JSON integer exceeds bit bound", context={"limit": max_int_bits})
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise CanonicalJSONError("canonical JSON floats must be finite")
            return item
        raise CanonicalJSONError("canonical JSON value is not JSON compatible", context={"type": type(item).__name__})

    return freeze(value, 0)


def json_ready(value: Any, **bounds: Any) -> Any:
    """Return a detached mutable JSON projection after bounded validation."""

    frozen = deep_freeze_json(value, **bounds)

    def thaw(item: FrozenJson) -> Any:
        if isinstance(item, Mapping):
            return {key: thaw(value) for key, value in item.items()}
        if isinstance(item, tuple):
            return [thaw(value) for value in item]
        return item

    return thaw(frozen)


def canonical_json_dumps(data: Any, **bounds: Any) -> str:
    """Encode a bounded value as compact, sorted canonical JSON text."""

    return json.dumps(json_ready(data, **bounds), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_json_bytes(data: Any, **bounds: Any) -> bytes:
    """Encode a bounded value as compact UTF-8 canonical JSON bytes."""

    return canonical_json_dumps(data, **bounds).encode("utf-8")


def canonical_json_loads(text: str, **bounds: Any) -> Any:
    """Parse JSON text into a duplicate-aware immutable bounded projection."""

    if not isinstance(text, str):
        raise CanonicalJSONError("canonical JSON text must be a string")
    try:
        decoded = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicate_keys)
    except CanonicalJSONError:
        raise
    except json.JSONDecodeError as exc:
        raise CanonicalJSONError("canonical JSON text could not be decoded", context={"error": str(exc)}) from exc
    return deep_freeze_json(decoded, **bounds)


def canonical_json_load_bytes(data: bytes, **bounds: Any) -> Any:
    """Parse UTF-8 JSON bytes with duplicate-key and bound validation."""

    if not isinstance(data, bytes):
        raise CanonicalJSONError("canonical JSON bytes must be bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalJSONError("canonical JSON bytes must be valid UTF-8", context={"error": str(exc)}) from exc
    return canonical_json_loads(text, **bounds)


def check_string(value: str, limit: int) -> None:
    """Raise when a JSON string exceeds its declared code-point bound."""

    if len(value) > limit:
        raise CanonicalJSONError("canonical JSON string exceeds length bound", context={"limit": limit})


def freeze_mapping(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable deterministic copy of a string-keyed mapping."""

    if mapping is None:
        return MappingProxyType({})
    frozen = deep_freeze_json(mapping)
    if not isinstance(frozen, Mapping):
        raise CanonicalJSONError("canonical JSON value must be a mapping")
    return frozen


def _reject_constant(value: str) -> None:
    raise CanonicalJSONError("canonical JSON floats must be finite", context={"value": value})


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError("canonical JSON object contains duplicate key", context={"key": key})
        result[key] = value
    return result


__all__ = ["DEFAULT_MAX_DEPTH", "DEFAULT_MAX_ENTRIES", "DEFAULT_MAX_INT_BITS", "DEFAULT_MAX_NODES", "DEFAULT_MAX_STRING", "FrozenJson", "JsonPrimitive", "canonical_json_bytes", "canonical_json_dumps", "canonical_json_load_bytes", "canonical_json_loads", "deep_freeze_json", "freeze_mapping", "json_ready"]
