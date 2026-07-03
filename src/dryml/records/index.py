"""JSON-backed rebuildable reference index for record/spec sidecars."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dryml.formats.ids import stable_hash

from .errors import RecordError, RecordValidationError, SpecValidationError
from .refs import LocatedRecordRef, LocatedSpecRef
from .scanner import ReferenceMention, scan_record_refs, scan_spec_refs
from .specs import spec_family_for_id


RECORD_REF_INDEX_SCHEMA = "dryml.records.ref_index.v1"
RECORD_REF_INDEX_SCHEMA_VERSION = 1
RECORD_REF_INDEX_FILENAME = "ref-index-v1.json"


class RecordRefIndexError(RecordError):
    """Base class for record reference index errors."""


class RecordRefIndexMissing(RecordRefIndexError):
    """Raised when the optional reference index is absent."""


class RecordRefIndexDirty(RecordRefIndexError):
    """Raised when callers require a clean reference index but it is dirty."""


class RecordRefIndexValidationError(RecordRefIndexError):
    """Raised when a reference index JSON document is malformed."""


@dataclass(frozen=True, slots=True)
class RecordRefIndex:
    """Validated in-memory representation of ``ref-index-v1.json``."""

    store_ref: str
    sources: tuple[dict[str, Any], ...]
    mentions: tuple[ReferenceMention, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.store_ref, str) or not self.store_ref:
            raise RecordRefIndexValidationError("index store_ref must be a non-empty string")
        object.__setattr__(self, "sources", tuple(_validate_source(source) for source in self.sources))
        object.__setattr__(self, "mentions", tuple(sorted(self.mentions, key=_mention_sort_key)))

    @property
    def source_count(self) -> int:
        """Return the number of indexed source documents."""

        return len(self.sources)

    @property
    def mention_count(self) -> int:
        """Return the number of indexed reference mentions."""

        return len(self.mentions)

    def to_json(self) -> dict[str, Any]:
        """Return canonical JSON-compatible index data."""

        return {
            "schema": RECORD_REF_INDEX_SCHEMA,
            "schema_version": RECORD_REF_INDEX_SCHEMA_VERSION,
            "store_ref": self.store_ref,
            "source_count": self.source_count,
            "mention_count": self.mention_count,
            "sources": list(self.sources),
            "mentions": [mention.to_json() for mention in self.mentions],
        }

    @classmethod
    def from_json(cls, data: Any) -> "RecordRefIndex":
        """Build and validate an index from decoded JSON data."""

        return validate_record_ref_index(data)

    def filter_mentions(
        self,
        *,
        target_id: str | None = None,
        target_kind: str | None = None,
        cdef_semantics: str | None = None,
        source_kind: str | None = None,
        source_family: str | None = None,
    ) -> tuple[ReferenceMention, ...]:
        """Return mentions matching the supplied optional filters."""

        result = []
        for mention in self.mentions:
            if target_id is not None and mention.target_id != target_id:
                continue
            if target_kind is not None and mention.target_kind != target_kind:
                continue
            if cdef_semantics is not None and mention.cdef_semantics != cdef_semantics:
                continue
            if source_kind is not None and mention.source_kind != source_kind:
                continue
            if source_family is not None and mention.source_family != source_family:
                continue
            result.append(mention)
        return tuple(result)

    def located_record_refs(self, mentions: tuple[ReferenceMention, ...] | None = None) -> tuple[LocatedRecordRef, ...]:
        """Return deterministic located record refs for record-source mentions."""

        source_mentions = self.mentions if mentions is None else mentions
        refs = [LocatedRecordRef(store_ref=self.store_ref, record_id=m.source_id) for m in source_mentions if m.source_kind == "record"]
        return _dedupe(refs)

    def located_spec_refs(self, mentions: tuple[ReferenceMention, ...] | None = None) -> tuple[LocatedSpecRef, ...]:
        """Return deterministic located spec refs for spec-source mentions."""

        source_mentions = self.mentions if mentions is None else mentions
        refs = [LocatedSpecRef(store_ref=self.store_ref, spec_id=m.source_id, kind=m.source_family) for m in source_mentions if m.source_kind == "spec"]
        return _dedupe(refs)


@dataclass(frozen=True, slots=True)
class RecordRefIndexRebuildReport:
    """Summary returned after rebuilding a store-local reference index."""

    store_ref: str
    changed: bool
    source_count: int
    mention_count: int
    records_scanned: int
    specs_scanned: int
    index_path: str


def build_record_ref_index(record_io: Any) -> tuple[RecordRefIndex, int, int]:
    """Build a reference index from authoritative record/spec JSON files."""

    sources: list[dict[str, Any]] = []
    mentions: list[ReferenceMention] = []
    records_scanned = 0
    specs_scanned = 0
    for record in record_io.iter_records():
        records_scanned += 1
        sources.append(
            {
                "source_kind": "record",
                "source_family": None,
                "source_id": record["id"],
                "document_hash": stable_hash(record),
            }
        )
        mentions.extend(scan_record_refs(record))
    for spec in record_io.iter_specs():
        specs_scanned += 1
        family = spec_family_for_id(spec["id"])
        sources.append(
            {
                "source_kind": "spec",
                "source_family": family,
                "source_id": spec["id"],
                "document_hash": stable_hash(spec),
            }
        )
        mentions.extend(scan_spec_refs(spec, family=family))
    sources = sorted(sources, key=lambda source: (source["source_kind"], source.get("source_family") or "", source["source_id"]))
    index = RecordRefIndex(store_ref=record_io._store_ref(), sources=tuple(sources), mentions=tuple(_dedupe(sorted(mentions, key=_mention_sort_key))))
    return index, records_scanned, specs_scanned


def validate_record_ref_index(data: Any) -> RecordRefIndex:
    """Validate decoded ``ref-index-v1.json`` data."""

    if not isinstance(data, Mapping):
        raise RecordRefIndexValidationError("reference index root must be an object", context={"type": type(data).__name__})
    if data.get("schema") != RECORD_REF_INDEX_SCHEMA:
        raise RecordRefIndexValidationError("reference index schema mismatch", context={"schema": data.get("schema")})
    if data.get("schema_version") != RECORD_REF_INDEX_SCHEMA_VERSION:
        raise RecordRefIndexValidationError("reference index schema_version mismatch", context={"schema_version": data.get("schema_version")})
    sources_data = data.get("sources")
    mentions_data = data.get("mentions")
    if not isinstance(sources_data, list):
        raise RecordRefIndexValidationError("reference index sources must be a list")
    if not isinstance(mentions_data, list):
        raise RecordRefIndexValidationError("reference index mentions must be a list")
    try:
        sources = tuple(_validate_source(source) for source in sources_data)
        mentions = tuple(ReferenceMention.from_json(mention) for mention in mentions_data)
        index = RecordRefIndex(store_ref=data.get("store_ref"), sources=sources, mentions=mentions)
    except (RecordValidationError, SpecValidationError) as exc:
        raise RecordRefIndexValidationError(str(exc), context=getattr(exc, "context", {})) from exc
    if data.get("source_count") != index.source_count:
        raise RecordRefIndexValidationError("reference index source_count mismatch")
    if data.get("mention_count") != index.mention_count:
        raise RecordRefIndexValidationError("reference index mention_count mismatch")
    return index


def _validate_source(source: Any) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise RecordRefIndexValidationError("reference index source must be an object", context={"type": type(source).__name__})
    source_kind = source.get("source_kind")
    source_id = source.get("source_id")
    source_family = source.get("source_family")
    document_hash = source.get("document_hash")
    if source_kind == "record":
        ReferenceMention(source_kind="record", source_id=source_id, source_family=None, path="/payload", target_kind="content_id", target_id=source_id, ref_kind="content_id", prefix="record", schema_version=1)
        source_family = None
    elif source_kind == "spec":
        family = spec_family_for_id(source_id)
        if source_family != family:
            raise RecordRefIndexValidationError("source family does not match source ID", context={"source_id": source_id})
    else:
        raise RecordRefIndexValidationError("source_kind must be record or spec", context={"source_kind": source_kind})
    if document_hash is not None and not isinstance(document_hash, str):
        raise RecordRefIndexValidationError("document_hash must be a string when present")
    result = {"source_kind": source_kind, "source_family": source_family, "source_id": source_id}
    if document_hash is not None:
        result["document_hash"] = document_hash
    return result


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
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


__all__ = [
    "RECORD_REF_INDEX_FILENAME",
    "RECORD_REF_INDEX_SCHEMA",
    "RECORD_REF_INDEX_SCHEMA_VERSION",
    "RecordRefIndex",
    "RecordRefIndexDirty",
    "RecordRefIndexError",
    "RecordRefIndexMissing",
    "RecordRefIndexRebuildReport",
    "RecordRefIndexValidationError",
    "build_record_ref_index",
    "validate_record_ref_index",
]
