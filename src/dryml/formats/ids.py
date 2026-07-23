"""Stable hash and schema-versioned content-ID helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json_bytes
from .errors import ContentIDError


_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CONTENT_ID_RE = re.compile(
    r"^(?P<prefix>[a-z][a-z0-9_]*)-v(?P<version>[1-9][0-9]*)-(?P<digest>[0-9a-f]{64})$"
)


@dataclass(frozen=True, slots=True)
class ContentIDParts:
    """Parsed components of a DRYML content ID."""

    prefix: str
    schema_version: int
    digest: str
    raw: str


def stable_hash(data: Any, *, algorithm: str = "sha256") -> str:
    """Return a lowercase hex digest over canonical JSON bytes.

    Parameters
    ----------
    data:
        JSON-compatible data to hash.
    algorithm:
        Hash algorithm name. Sprint 0 supports ``"sha256"``.

    Returns
    -------
    str
        Lowercase SHA-256 hex digest.
    """

    if algorithm != "sha256":
        raise ContentIDError(
            "unsupported content hash algorithm",
            context={"algorithm": algorithm},
        )
    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def content_id(prefix: str, schema_version: int, data: Any) -> str:
    """Return a namespaced content ID for schema-versioned data.

    Parameters
    ----------
    prefix:
        Lowercase ID namespace prefix.
    schema_version:
        Positive schema version integer.
    data:
        JSON-compatible payload data.

    Returns
    -------
    str
        ``<prefix>-v<schema_version>-<sha256>``.
    """

    prefix = validate_id_prefix(prefix)
    schema_version = validate_schema_version(schema_version)
    payload = {
        "id_prefix": prefix,
        "schema_version": schema_version,
        "data": data,
    }
    return f"{prefix}-v{schema_version}-{stable_hash(payload)}"


def parse_content_id(value: str) -> ContentIDParts:
    """Parse a DRYML content ID into validated components.

    Parameters
    ----------
    value:
        Content ID string to parse.

    Returns
    -------
    ContentIDParts
        Parsed content-ID parts including the original raw string.
    """

    if not isinstance(value, str):
        raise ContentIDError(
            "content ID must be a string",
            context={"type": type(value).__name__},
        )
    match = _CONTENT_ID_RE.fullmatch(value)
    if match is None:
        raise ContentIDError("invalid content ID", context={"value": value})
    return ContentIDParts(
        prefix=match.group("prefix"),
        schema_version=int(match.group("version")),
        digest=match.group("digest"),
        raw=value,
    )


def is_content_id(value: object, *, prefix: str | None = None) -> bool:
    """Return whether *value* is a valid content ID.

    Parameters
    ----------
    value:
        Candidate value.
    prefix:
        Optional required content-ID prefix.

    Returns
    -------
    bool
        ``True`` when the value parses and matches the optional prefix.
    """

    try:
        parts = parse_content_id(value)  # type: ignore[arg-type]
    except ContentIDError:
        return False
    return prefix is None or parts.prefix == prefix


def validate_id_prefix(prefix: str) -> str:
    """Validate and return a content-ID prefix.

    Parameters
    ----------
    prefix:
        Candidate prefix.

    Returns
    -------
    str
        The validated prefix.
    """

    if not isinstance(prefix, str):
        raise ContentIDError(
            "content ID prefix must be a string",
            context={"type": type(prefix).__name__},
        )
    if _PREFIX_RE.fullmatch(prefix) is None:
        raise ContentIDError("invalid content ID prefix", context={"prefix": prefix})
    return prefix


def validate_schema_version(schema_version: int) -> int:
    """Validate and return a positive schema version integer.

    Parameters
    ----------
    schema_version:
        Candidate schema version.

    Returns
    -------
    int
        The validated schema version.
    """

    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ContentIDError(
            "schema version must be an integer",
            context={"type": type(schema_version).__name__},
        )
    if schema_version < 1:
        raise ContentIDError(
            "schema version must be positive",
            context={"schema_version": schema_version},
        )
    return schema_version


__all__ = [
    "ContentIDParts",
    "content_id",
    "is_content_id",
    "parse_content_id",
    "stable_hash",
    "validate_id_prefix",
    "validate_schema_version",
]
