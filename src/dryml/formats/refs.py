"""Reserved reference and literal escape helpers for DRYML formats."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .canonical import json_ready
from .errors import CanonicalJSONError, ContentIDError, ReferenceParseError
from .ids import ContentIDParts, parse_content_id, validate_schema_version


_CDEF_ID_RE = re.compile(r"^cdef-v(?P<version>[1-9][0-9]*)-(?P<digest>[0-9a-f]{16,})$")
_REF_CDEF_RE = re.compile(r"^ref\((?P<cdef_id>cdef-v[1-9][0-9]*-[0-9a-f]{16,})\)$")
_RESERVED_CONTENT_PREFIXES = frozenset(
    {
        "annotation",
        "blob",
        "env",
        "envlock",
        "envrec",
        "envreq",
        "envspec",
        "op",
        "record",
        "repr",
        "runtime",
        "spec",
        "world",
        "worldreq",
    }
)


@dataclass(frozen=True, slots=True)
class CDefID:
    """Parsed components of a concrete-definition ID string."""

    schema_version: int
    digest: str
    raw: str


@dataclass(frozen=True, slots=True)
class ReservedRef:
    """Parsed reserved reference or literal escape."""

    kind: str
    raw: Any
    target: str | None = None
    prefix: str | None = None
    schema_version: int | None = None
    digest: str | None = None


def format_cdef_id(digest: str, *, schema_version: int = 4) -> str:
    """Format a concrete-definition ID from digest and schema version.

    Parameters
    ----------
    digest:
        Lowercase hex digest with at least 16 characters.
    schema_version:
        Positive CDef ID schema version.

    Returns
    -------
    str
        ``cdef-v<schema_version>-<digest>``.
    """

    validate_schema_version(schema_version)
    _validate_cdef_digest(digest)
    return f"cdef-v{schema_version}-{digest}"


def parse_cdef_id(value: str) -> CDefID:
    """Parse a concrete-definition ID string.

    Parameters
    ----------
    value:
        Candidate ``cdef-v...`` string.

    Returns
    -------
    CDefID
        Parsed CDef ID parts.
    """

    if not isinstance(value, str):
        raise ReferenceParseError("CDef ID must be a string", context={"type": type(value).__name__})
    match = _CDEF_ID_RE.fullmatch(value)
    if match is None:
        raise ReferenceParseError("invalid CDef ID", context={"value": value})
    return CDefID(
        schema_version=int(match.group("version")),
        digest=match.group("digest"),
        raw=value,
    )


def is_cdef_id(value: object) -> bool:
    """Return whether *value* is a valid concrete-definition ID."""

    try:
        parse_cdef_id(value)  # type: ignore[arg-type]
    except ReferenceParseError:
        return False
    return True


def format_ref_cdef(cdef_id: str) -> str:
    """Format a non-materializing CDef reference string.

    Parameters
    ----------
    cdef_id:
        Valid concrete-definition ID string.

    Returns
    -------
    str
        ``ref(<cdef_id>)``.
    """

    parsed = parse_cdef_id(cdef_id)
    return f"ref({parsed.raw})"


def parse_ref_cdef(value: str) -> CDefID:
    """Parse a non-materializing CDef reference string."""

    if not isinstance(value, str):
        raise ReferenceParseError("CDef ref must be a string", context={"type": type(value).__name__})
    match = _REF_CDEF_RE.fullmatch(value)
    if match is None:
        raise ReferenceParseError("invalid CDef ref", context={"value": value})
    return parse_cdef_id(match.group("cdef_id"))


def is_ref_cdef(value: object) -> bool:
    """Return whether *value* is a valid ``ref(cdef-v...)`` string."""

    try:
        parse_ref_cdef(value)  # type: ignore[arg-type]
    except ReferenceParseError:
        return False
    return True


def literal_escape(value: Any) -> dict[str, Any]:
    """Wrap a value so reserved-reference scanners treat it as a literal."""

    try:
        return {"$literal": json_ready(value)}
    except CanonicalJSONError as exc:
        raise ReferenceParseError("literal escape value is not JSON serializable", context=exc.context) from exc


def is_literal_escape(value: object) -> bool:
    """Return whether *value* is exactly a literal escape mapping."""

    return isinstance(value, Mapping) and tuple(value.keys()) == ("$literal",)


def unwrap_literal_escape(value: Mapping[str, Any]) -> Any:
    """Return the escaped literal value from a literal escape mapping."""

    if not isinstance(value, Mapping):
        raise ReferenceParseError("literal escape must be a mapping", context={"type": type(value).__name__})
    if "$literal" not in value:
        raise ReferenceParseError("literal escape missing $literal")
    if len(value) != 1:
        raise ReferenceParseError("literal escape must contain only $literal", context={"keys": tuple(value.keys())})
    try:
        return json_ready(value["$literal"])
    except CanonicalJSONError as exc:
        raise ReferenceParseError("literal escape value is not JSON serializable", context=exc.context) from exc


def parse_reserved_ref(value: Any) -> ReservedRef | None:
    """Parse one reserved reference or literal escape value.

    Parameters
    ----------
    value:
        Candidate string or literal escape mapping.

    Returns
    -------
    ReservedRef | None
        Parsed reserved reference, parsed literal escape, or ``None``.
    """

    if isinstance(value, Mapping):
        if "$literal" not in value:
            return None
        unwrap_literal_escape(value)
        return ReservedRef(kind="literal", raw=value)
    if not isinstance(value, str):
        return None
    if value.startswith("ref(") or value.startswith("cdef-v"):
        try:
            cdef = parse_ref_cdef(value) if value.startswith("ref(") else parse_cdef_id(value)
        except ReferenceParseError:
            raise
        return ReservedRef(
            kind="ref_cdef" if value.startswith("ref(") else "cdef",
            raw=value,
            target=cdef.raw,
            schema_version=cdef.schema_version,
            digest=cdef.digest,
        )
    try:
        content = parse_content_id(value)
    except ContentIDError as exc:
        if _looks_like_reserved_content_ref(value):
            raise ReferenceParseError(
                "invalid reserved content reference",
                context={"value": value, "error": str(exc)},
            ) from exc
        return None
    return _reserved_content_ref(value, content)


def is_reserved_ref(value: Any) -> bool:
    """Return whether *value* is a recognized reserved reference."""

    try:
        return parse_reserved_ref(value) is not None
    except ReferenceParseError:
        return False


def _reserved_content_ref(raw: str, content: ContentIDParts) -> ReservedRef:
    return ReservedRef(
        kind="content_id",
        raw=raw,
        target=raw,
        prefix=content.prefix,
        schema_version=content.schema_version,
        digest=content.digest,
    )


def _looks_like_reserved_content_ref(value: str) -> bool:
    prefix, separator, _rest = value.partition("-v")
    return bool(separator) and prefix in _RESERVED_CONTENT_PREFIXES


def _validate_cdef_digest(digest: str) -> None:
    if not isinstance(digest, str):
        raise ReferenceParseError("CDef digest must be a string", context={"type": type(digest).__name__})
    if re.fullmatch(r"[0-9a-f]{16,}", digest) is None:
        raise ReferenceParseError("invalid CDef digest", context={"digest": digest})


__all__ = [
    "CDefID",
    "ReservedRef",
    "format_cdef_id",
    "format_ref_cdef",
    "is_cdef_id",
    "is_literal_escape",
    "is_ref_cdef",
    "is_reserved_ref",
    "literal_escape",
    "parse_cdef_id",
    "parse_ref_cdef",
    "parse_reserved_ref",
    "unwrap_literal_escape",
]
