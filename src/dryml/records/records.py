"""Generic bounded, content-addressed envelopes for sidecar record families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dryml.formats import EnvelopeError, canonical_json_bytes, deep_freeze_json, json_ready, make_envelope, semantic_id, validate_envelope


GENERIC_RECORD_SCHEMA = "dryml.record.v1.1"
"""The closed v1.1 schema shared by generic sidecar records."""

_RECORD_PREFIX = "record"
_RECORD_MAX_BYTES = 32 * 1024 * 1024
_RECORD_BOUNDS = {
    "max_depth": 64,
    "max_nodes": 131_072,
    "max_entries": 65_536,
    "max_string": 4096,
    "max_int_bits": 4096,
}


@dataclass(frozen=True, slots=True, init=False)
class GenericRecord:
    """One detached generic record payload without domain interpretation.

    Args:
        kind: Domain-owned record-family name. Generic records do not import or
            dispatch to an owner based on this value.
        version: Positive integer grammar version owned by ``kind``.
        data: String-keyed JSON-compatible domain payload.

    Raises:
        TypeError: If ``kind``, ``version``, or ``data`` has an unsupported type.
        ValueError: If ``version`` is not positive or the payload exceeds common
            canonical JSON bounds.

    The constructor detaches input. :attr:`data` returns a fresh mutable copy so
    callers cannot alter a record retained by another codec layer.
    """

    kind: str
    version: int
    _data: Mapping[str, Any]

    def __init__(self, kind: str, version: int, data: Mapping[str, Any]):
        if not isinstance(kind, str) or not kind:
            raise ValueError("record kind must be a non-empty string")
        if type(version) is not int:
            raise TypeError("record version must be an integer")
        if version < 1:
            raise ValueError("record version must be positive")
        if not isinstance(data, Mapping):
            raise TypeError("record data must be a mapping")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "_data", deep_freeze_json(data, **_RECORD_BOUNDS))

    @property
    def data(self) -> dict[str, Any]:
        """Return a detached mutable copy of the domain payload."""

        return json_ready(self._data, **_RECORD_BOUNDS)


def encode_record(record: GenericRecord, *, max_bytes: int = _RECORD_MAX_BYTES) -> dict[str, Any]:
    """Encode a generic record into a closed, semantic-ID-checked envelope.

    Args:
        record: Detached generic record whose domain data is already owned by a
            caller-specific validator.
        max_bytes: Maximum encoded size from one through 32 MiB.

    Returns:
        A detached canonical JSON-compatible envelope.

    Raises:
        TypeError: If arguments have unsupported types.
        ValueError: If ``max_bytes`` is outside the supported range.
        EnvelopeError: If the complete envelope exceeds the chosen bound.
    """

    if not isinstance(record, GenericRecord):
        raise TypeError("record must be a GenericRecord")
    max_bytes = _validate_max_bytes(max_bytes)
    payload = {"version": record.version, "data": record.data}
    return make_envelope(
        schema=GENERIC_RECORD_SCHEMA,
        kind=record.kind,
        prefix=_RECORD_PREFIX,
        payload=payload,
        semantic_id=semantic_id(_RECORD_PREFIX, GENERIC_RECORD_SCHEMA, record.kind, payload, **_RECORD_BOUNDS),
        max_bytes=max_bytes,
        **_RECORD_BOUNDS,
    )


def decode_record(data: Mapping[str, Any], *, max_bytes: int = _RECORD_MAX_BYTES) -> GenericRecord:
    """Validate and decode one generic sidecar record envelope.

    Args:
        data: Decoded JSON envelope with the common record schema.
        max_bytes: Maximum encoded size from one through 32 MiB.

    Returns:
        A detached generic payload for a domain owner to validate.

    Raises:
        TypeError: If ``max_bytes`` has an unsupported type.
        ValueError: If ``max_bytes`` is outside the supported range.
        EnvelopeError: If the envelope, attached ID, or nested wrapper is
            malformed. Unknown record kinds and versions remain for owners to
            reject.
    """

    max_bytes = _validate_max_bytes(max_bytes)
    if not isinstance(data, Mapping):
        raise EnvelopeError("generic record envelope must be a mapping")
    raw = dict(data)
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        raise EnvelopeError("generic record kind must be a non-empty string")
    envelope = validate_envelope(
        raw,
        schema=GENERIC_RECORD_SCHEMA,
        kind=kind,
        prefix=_RECORD_PREFIX,
        identifying_payload=raw.get("payload", {}),
        max_bytes=max_bytes,
        **_RECORD_BOUNDS,
    )
    payload = envelope["payload"]
    if set(payload) != {"version", "data"}:
        raise EnvelopeError("generic record payload fields are closed")
    try:
        return GenericRecord(kind, payload["version"], payload["data"])
    except (TypeError, ValueError) as error:
        raise EnvelopeError("generic record payload is malformed") from error


def _validate_max_bytes(value: int) -> int:
    """Validate the shared record-envelope byte limit."""

    if type(value) is not int:
        raise TypeError("record max_bytes must be an integer")
    if not 1 <= value <= _RECORD_MAX_BYTES:
        raise ValueError(f"record max_bytes must be between 1 and {_RECORD_MAX_BYTES}")
    return value


__all__ = ["GENERIC_RECORD_SCHEMA", "GenericRecord", "decode_record", "encode_record"]
