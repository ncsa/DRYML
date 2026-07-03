"""Canonical JSON helpers for content-addressed DRYML metadata."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .errors import CanonicalJSONError


JsonPrimitive = str | int | float | bool | None
FrozenJson = JsonPrimitive | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


def json_ready(value: Any) -> Any:
    """Return a mutable JSON-compatible view of canonical metadata.

    Parameters
    ----------
    value:
        JSON-compatible data, frozen canonical data, or supported containers.

    Returns
    -------
    Any
        Data made from JSON primitives, dictionaries, and lists.
    """

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
        raise CanonicalJSONError(
            "canonical JSON floats must be finite",
            context={"value": repr(value)},
        )
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise CanonicalJSONError(
        f"canonical JSON value {value!r} is not JSON serializable",
        context={"type": type(value).__name__},
    )


def deep_freeze_json(value: Any) -> FrozenJson:
    """Return deeply immutable canonical JSON metadata.

    Parameters
    ----------
    value:
        JSON-compatible data to normalize and freeze.

    Returns
    -------
    FrozenJson
        Immutable JSON primitives, tuples, and mapping proxies with sorted keys.
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
        raise CanonicalJSONError(
            "canonical JSON floats must be finite",
            context={"value": repr(value)},
        )
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise CanonicalJSONError(
        f"canonical JSON value {value!r} is not JSON serializable",
        context={"type": type(value).__name__},
    )


def canonical_json_dumps(data: Any) -> str:
    """Return a stable compact JSON string for JSON-compatible data.

    Parameters
    ----------
    data:
        JSON-compatible data to serialize.

    Returns
    -------
    str
        Canonical JSON with lexicographically sorted mapping keys.
    """

    try:
        return json.dumps(
            json_ready(data),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except CanonicalJSONError:
        raise
    except (TypeError, ValueError) as exc:
        raise CanonicalJSONError(
            "canonical JSON data is not JSON serializable",
            context={"error": str(exc)},
        ) from exc


def canonical_json_loads(text: str) -> Any:
    """Decode strict canonical JSON text into ordinary JSON data.

    Parameters
    ----------
    text:
        JSON text to decode.

    Returns
    -------
    Any
        Decoded JSON-compatible Python data. Non-finite JSON constants and
        duplicate object keys are rejected because their canonical meaning is
        ambiguous.
    """

    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except CanonicalJSONError:
        raise
    except json.JSONDecodeError as exc:
        raise CanonicalJSONError(
            "canonical JSON text could not be decoded",
            context={"error": str(exc)},
        ) from exc
    except TypeError as exc:
        raise CanonicalJSONError(
            "canonical JSON text must be a string",
            context={"error": str(exc)},
        ) from exc


def canonical_json_bytes(data: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for content hashing.

    Parameters
    ----------
    data:
        JSON-compatible data to serialize.

    Returns
    -------
    bytes
        UTF-8 encoded canonical JSON.
    """

    return canonical_json_dumps(data).encode("utf-8")


def canonical_json_load_bytes(data: bytes) -> Any:
    """Decode UTF-8 canonical JSON bytes into ordinary JSON data.

    Parameters
    ----------
    data:
        UTF-8 encoded JSON bytes.

    Returns
    -------
    Any
        Decoded JSON-compatible Python data.
    """

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalJSONError(
            "canonical JSON bytes must be UTF-8",
            context={"error": str(exc)},
        ) from exc
    return canonical_json_loads(text)


def freeze_mapping(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable shallow copy with deterministic string keys.

    Parameters
    ----------
    mapping:
        Mapping whose values are already immutable domain objects.

    Returns
    -------
    Mapping[str, Any]
        A mapping proxy ordered by each key's string representation.
    """

    return MappingProxyType({str(key): mapping[key] for key in sorted(mapping or {}, key=str)})


def _sorted_string_keys(mapping: Mapping[Any, Any]) -> tuple[str, ...]:
    keys = []
    for key in mapping:
        if not isinstance(key, str):
            raise CanonicalJSONError(
                "canonical JSON mapping keys must be strings",
                context={"key": repr(key), "type": type(key).__name__},
            )
        keys.append(key)
    return tuple(sorted(keys))


def _reject_json_constant(value: str) -> None:
    raise CanonicalJSONError(
        "canonical JSON non-finite number is not allowed",
        context={"value": value},
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(
                "canonical JSON object contains duplicate key",
                context={"key": key},
            )
        result[key] = value
    return result


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
