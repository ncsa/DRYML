"""Recursive reference scanning for validated record/spec JSON payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dryml.formats.errors import ContentIDError, ReferenceParseError
from dryml.formats.ids import parse_content_id
from dryml.formats.refs import parse_cdef_id, parse_reserved_ref

from .errors import RecordValidationError, SpecValidationError
from .kinds import SPEC_FAMILIES, SPEC_FAMILY_BY_PREFIX
from .records import attach_record_id, validate_record
from .specs import attach_spec_id, spec_family_for_id, validate_spec


_SourceKind = Literal["record", "spec"]
_TargetKind = Literal["cdef", "content_id"]
_RefKind = Literal["cdef", "ref_cdef", "content_id"]
_CDefSemantics = Literal["materialize", "reference"]

_TYPED_KEYS: dict[str, tuple[str, str, bool, tuple[str, ...] | None]] = {
    "subject_cdef_id": ("cdef", "subject", False, None),
    "owner_cdef_id": ("cdef", "owner", False, None),
    "input_cdef_ids": ("cdef", "input", True, None),
    "output_cdef_ids": ("cdef", "output", True, None),
    "consumed_cdef_ids": ("cdef", "consumed", True, None),
    "produced_cdef_ids": ("cdef", "produced", True, None),
    "operation_id": ("content_id", "operation", False, ("op",)),
    "operation_ids": ("content_id", "operation", True, ("op",)),
    "representation_id": ("content_id", "representation", False, ("repr",)),
    "environment_id": ("content_id", "environment", False, ("envrec", "env")),
    "environment_record_id": ("content_id", "environment_record", False, ("envrec",)),
    "environment_requirement_id": ("content_id", "environment_requirement", False, ("envreq",)),
    "environment_spec_id": ("content_id", "environment_spec", False, ("envspec",)),
    "environment_lock_id": ("content_id", "environment_lock", False, ("envlock",)),
    "world_requirement_id": ("content_id", "world_requirement", False, ("worldreq",)),
    "world_id": ("content_id", "world", False, ("world",)),
    "world_allocation_id": ("content_id", "world_allocation", False, ("worldalloc",)),
    "runtime_id": ("content_id", "runtime", False, ("runtime",)),
    "record_id": ("content_id", "record", False, ("record",)),
    "record_ids": ("content_id", "record", True, ("record",)),
    "derived_from": ("content_id", "derived_from", True, ("record",)),
    "consumed_records": ("content_id", "consumed_record", True, ("record",)),
    "produced_records": ("content_id", "produced_record", True, ("record",)),
}


@dataclass(frozen=True, slots=True)
class ReferenceMention:
    """One deterministic reference mention found in a record or spec payload."""

    source_kind: _SourceKind
    source_id: str
    source_family: str | None
    path: str
    target_kind: _TargetKind
    target_id: str
    ref_kind: _RefKind
    cdef_semantics: _CDefSemantics | None = None
    prefix: str | None = None
    schema_version: int | None = None
    typed_key: str | None = None
    typed_role: str | None = None

    def __post_init__(self) -> None:
        _validate_scan_source(self.source_kind, self.source_id, self.source_family)
        if not isinstance(self.path, str) or not self.path.startswith("/"):
            raise RecordValidationError("reference mention path must be a JSON Pointer", context={"path": self.path})
        if self.target_kind == "cdef":
            cdef = parse_cdef_id(self.target_id)
            if self.ref_kind not in {"cdef", "ref_cdef"}:
                raise RecordValidationError("CDef mentions require cdef/ref_cdef ref_kind")
            if self.cdef_semantics not in {"materialize", "reference"}:
                raise RecordValidationError("CDef mentions require cdef_semantics")
            if self.schema_version != cdef.schema_version:
                raise RecordValidationError("CDef mention schema_version mismatch")
        elif self.target_kind == "content_id":
            parts = _parse_content_id_as(self.target_id, None)
            if self.ref_kind != "content_id":
                raise RecordValidationError("content ID mentions require content_id ref_kind")
            if self.prefix != parts.prefix or self.schema_version != parts.schema_version:
                raise RecordValidationError("content ID mention prefix/schema_version mismatch")
        else:
            raise RecordValidationError("reference mention has invalid target_kind", context={"target_kind": self.target_kind})

    def to_json(self) -> dict[str, Any]:
        """Return the canonical JSON form of this mention."""

        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_family": self.source_family,
            "path": self.path,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "ref_kind": self.ref_kind,
            "cdef_semantics": self.cdef_semantics,
            "prefix": self.prefix,
            "schema_version": self.schema_version,
            "typed_key": self.typed_key,
            "typed_role": self.typed_role,
        }

    @classmethod
    def from_json(cls, data: Any) -> "ReferenceMention":
        """Build and validate a mention from JSON data."""

        if not isinstance(data, Mapping):
            raise RecordValidationError("reference mention JSON must be an object", context={"type": type(data).__name__})
        return cls(
            source_kind=data.get("source_kind"),
            source_id=data.get("source_id"),
            source_family=data.get("source_family"),
            path=data.get("path"),
            target_kind=data.get("target_kind"),
            target_id=data.get("target_id"),
            ref_kind=data.get("ref_kind"),
            cdef_semantics=data.get("cdef_semantics"),
            prefix=data.get("prefix"),
            schema_version=data.get("schema_version"),
            typed_key=data.get("typed_key"),
            typed_role=data.get("typed_role"),
        )


def scan_json_refs(
    value: Any,
    *,
    source_kind: _SourceKind,
    source_id: str,
    source_family: str | None = None,
    base_path: str = "/payload",
) -> tuple[ReferenceMention, ...]:
    """Scan JSON-ready data for reserved refs and typed reference keys."""

    _validate_scan_source(source_kind, source_id, source_family)
    mentions: list[ReferenceMention] = []
    _scan_value(value, source_kind=source_kind, source_id=source_id, source_family=source_family, path=base_path, mentions=mentions)
    return _sort_dedupe(mentions)


def scan_record_refs(record: Mapping[str, Any]) -> tuple[ReferenceMention, ...]:
    """Validate *record* and scan only its semantic ``payload``."""

    attached = attach_record_id(record)
    validate_record(attached)
    return scan_json_refs(attached["payload"], source_kind="record", source_id=attached["id"], source_family=None)


def scan_spec_refs(spec: Mapping[str, Any], *, family: str | None = None) -> tuple[ReferenceMention, ...]:
    """Validate *spec* and scan only its semantic ``payload``."""

    attached = attach_spec_id(spec, family=family)
    resolved_family = family or spec_family_for_id(attached["id"])
    validate_spec(attached, family=resolved_family)
    return scan_json_refs(attached["payload"], source_kind="spec", source_id=attached["id"], source_family=resolved_family)


def scan_store_refs(record_io: Any) -> tuple[ReferenceMention, ...]:
    """Scan all records and specs exposed by a ``RecordStoreIO`` instance."""

    mentions: list[ReferenceMention] = []
    for record in record_io.iter_records():
        mentions.extend(scan_record_refs(record))
    for spec in record_io.iter_specs():
        mentions.extend(scan_spec_refs(spec))
    return _sort_dedupe(mentions)


def _scan_value(
    value: Any,
    *,
    source_kind: _SourceKind,
    source_id: str,
    source_family: str | None,
    path: str,
    mentions: list[ReferenceMention],
) -> None:
    if isinstance(value, Mapping):
        if "$literal" in value:
            try:
                ref = parse_reserved_ref(value)
            except ReferenceParseError as exc:
                raise RecordValidationError("invalid literal escape", context=exc.context) from exc
            if ref is not None:
                return
        for key in sorted(value):
            child_path = f"{path}/{_escape_json_pointer(str(key))}"
            if key in _TYPED_KEYS:
                _scan_typed_key(
                    key,
                    value[key],
                    source_kind=source_kind,
                    source_id=source_id,
                    source_family=source_family,
                    path=child_path,
                    mentions=mentions,
                )
            else:
                _scan_value(value[key], source_kind=source_kind, source_id=source_id, source_family=source_family, path=child_path, mentions=mentions)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _scan_value(item, source_kind=source_kind, source_id=source_id, source_family=source_family, path=f"{path}/{index}", mentions=mentions)
        return
    if isinstance(value, str):
        mention = _mention_from_value(value, source_kind=source_kind, source_id=source_id, source_family=source_family, path=path)
        if mention is not None:
            mentions.append(mention)


def _scan_typed_key(
    key: str,
    value: Any,
    *,
    source_kind: _SourceKind,
    source_id: str,
    source_family: str | None,
    path: str,
    mentions: list[ReferenceMention],
) -> None:
    target_type, role, is_list, prefixes = _TYPED_KEYS[key]
    if is_list:
        if not isinstance(value, list):
            raise RecordValidationError("typed reference key requires a list", context={"key": key, "type": type(value).__name__})
        for index, item in enumerate(value):
            mentions.append(_typed_mention(key, role, item, target_type, prefixes, source_kind, source_id, source_family, f"{path}/{index}"))
        return
    if isinstance(value, list):
        raise RecordValidationError("typed reference key requires a scalar", context={"key": key})
    mentions.append(_typed_mention(key, role, value, target_type, prefixes, source_kind, source_id, source_family, path))


def _typed_mention(
    key: str,
    role: str,
    value: Any,
    target_type: str,
    prefixes: tuple[str, ...] | None,
    source_kind: _SourceKind,
    source_id: str,
    source_family: str | None,
    path: str,
) -> ReferenceMention:
    if target_type == "cdef":
        if not isinstance(value, str):
            raise RecordValidationError("typed CDef reference must be a string", context={"key": key, "type": type(value).__name__})
        try:
            cdef = parse_cdef_id(value)
        except ReferenceParseError as exc:
            raise RecordValidationError("typed CDef reference is invalid", context={"key": key, **exc.context}) from exc
        return ReferenceMention(
            source_kind=source_kind,
            source_id=source_id,
            source_family=source_family,
            path=path,
            target_kind="cdef",
            target_id=cdef.raw,
            ref_kind="cdef",
            cdef_semantics="materialize",
            schema_version=cdef.schema_version,
            typed_key=key,
            typed_role=role,
        )
    if not isinstance(value, str):
        raise RecordValidationError("typed content reference must be a string", context={"key": key, "type": type(value).__name__})
    parts = _parse_content_id_as(value, None)
    if prefixes is not None and parts.prefix not in prefixes:
        raise RecordValidationError("typed content reference prefix mismatch", context={"key": key, "expected": prefixes, "observed": parts.prefix})
    _validate_typed_content_schema_version(key, parts.prefix, parts.schema_version)
    return ReferenceMention(
        source_kind=source_kind,
        source_id=source_id,
        source_family=source_family,
        path=path,
        target_kind="content_id",
        target_id=value,
        ref_kind="content_id",
        prefix=parts.prefix,
        schema_version=parts.schema_version,
        typed_key=key,
        typed_role=role,
    )


def _mention_from_value(
    value: str,
    *,
    source_kind: _SourceKind,
    source_id: str,
    source_family: str | None,
    path: str,
) -> ReferenceMention | None:
    try:
        ref = parse_reserved_ref(value)
    except ReferenceParseError as exc:
        raise RecordValidationError("invalid reserved reference", context=exc.context) from exc
    if ref is None:
        return None
    if ref.kind in {"cdef", "ref_cdef"}:
        return ReferenceMention(
            source_kind=source_kind,
            source_id=source_id,
            source_family=source_family,
            path=path,
            target_kind="cdef",
            target_id=ref.target,  # type: ignore[arg-type]
            ref_kind=ref.kind,  # type: ignore[arg-type]
            cdef_semantics="materialize" if ref.kind == "cdef" else "reference",
            schema_version=ref.schema_version,
        )
    if ref.kind == "content_id":
        return ReferenceMention(
            source_kind=source_kind,
            source_id=source_id,
            source_family=source_family,
            path=path,
            target_kind="content_id",
            target_id=ref.target,  # type: ignore[arg-type]
            ref_kind="content_id",
            prefix=ref.prefix,
            schema_version=ref.schema_version,
        )
    return None


def _sort_dedupe(mentions: Iterable[ReferenceMention]) -> tuple[ReferenceMention, ...]:
    seen: set[ReferenceMention] = set()
    result: list[ReferenceMention] = []
    for mention in sorted(mentions, key=_mention_sort_key):
        if mention not in seen:
            seen.add(mention)
            result.append(mention)
    return tuple(result)


def _mention_sort_key(mention: ReferenceMention) -> tuple[Any, ...]:
    return (
        mention.source_kind,
        mention.source_family or "",
        mention.source_id,
        mention.path,
        mention.target_kind,
        mention.target_id,
        mention.ref_kind,
        mention.typed_key or "",
        mention.typed_role or "",
    )


def _escape_json_pointer(segment: str) -> str:
    return segment.replace("~", "~0").replace("/", "~1")


def _validate_scan_source(source_kind: str, source_id: str, source_family: str | None) -> None:
    if source_kind == "record":
        parts = _parse_content_id_as(source_id, "record")
        if parts.schema_version != 1:
            raise RecordValidationError("record source ID must use record-v1 prefix", context={"source_id": source_id, "schema_version": parts.schema_version})
        if source_family is not None:
            raise RecordValidationError("record mentions must not have source_family")
        return
    if source_kind == "spec":
        parts = _parse_content_id_as(source_id, None)
        if source_family not in SPEC_FAMILIES:
            raise SpecValidationError("spec source has unknown source_family", context={"source_family": source_family})
        info = SPEC_FAMILIES[source_family]
        if info.prefix != parts.prefix:
            raise SpecValidationError("spec source_family does not match source_id", context={"source_family": source_family})
        if info.schema_version != parts.schema_version:
            raise SpecValidationError(
                "spec source ID schema version does not match source_family",
                context={"source_family": source_family, "expected_version": info.schema_version, "observed_version": parts.schema_version},
            )
        return
    raise RecordValidationError("scan source_kind must be record or spec", context={"source_kind": source_kind})


def _validate_typed_content_schema_version(key: str, prefix: str, schema_version: int) -> None:
    if prefix == "record":
        expected_version = 1
    elif prefix in SPEC_FAMILY_BY_PREFIX:
        expected_version = SPEC_FAMILIES[SPEC_FAMILY_BY_PREFIX[prefix]].schema_version
    else:
        return
    if schema_version != expected_version:
        raise RecordValidationError(
            "typed content reference schema version mismatch",
            context={"key": key, "prefix": prefix, "expected_version": expected_version, "observed_version": schema_version},
        )


def _parse_content_id_as(value: str, prefix: str | None):
    try:
        parts = parse_content_id(value)
    except ContentIDError as exc:
        raise RecordValidationError("invalid content ID", context=exc.context) from exc
    if prefix is not None and parts.prefix != prefix:
        raise RecordValidationError("content ID prefix mismatch", context={"expected": prefix, "observed": parts.prefix})
    return parts


__all__ = ["ReferenceMention", "scan_json_refs", "scan_record_refs", "scan_spec_refs", "scan_store_refs"]
