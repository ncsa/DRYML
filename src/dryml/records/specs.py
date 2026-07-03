"""Canonical spec envelope helpers for store-owned DRYML sidecars."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.formats.envelope import envelope_payload_for_id, make_envelope, validate_envelope
from dryml.formats.errors import ContentIDError, EnvelopeError
from dryml.formats.ids import content_id, parse_content_id

from .errors import SpecValidationError
from .kinds import SPEC_FAMILIES, SPEC_FAMILY_BY_PREFIX, SPEC_FAMILY_BY_SCHEMA


_SPEC_TOP_LEVEL_FIELDS = frozenset({"schema", "schema_version", "id", "kind", "payload", "metadata"})


def make_spec(
    *,
    family: str,
    kind: str,
    payload: Mapping[str, Any] | None = None,
    id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    schema: str | None = None,
) -> dict[str, Any]:
    """Build a canonical JSON-ready spec envelope for a known family."""

    family_info = _family_info(family)
    schema_name = schema if family == "generic" else family_info.schema
    if not isinstance(schema_name, str) or not schema_name:
        raise SpecValidationError("generic specs require a non-empty schema")
    try:
        spec = make_envelope(
            schema=schema_name,
            schema_version=family_info.schema_version,
            kind=kind,
            payload=payload,
            id=id,
            metadata=metadata,
        )
    except EnvelopeError as exc:
        raise SpecValidationError("invalid spec envelope", context=exc.context) from exc
    return validate_spec(spec, family=family)


def validate_spec(spec: Mapping[str, Any], *, family: str | None = None, kind: str | None = None) -> Mapping[str, Any]:
    """Validate a spec envelope and any attached spec ID."""

    resolved_family = _validate_spec_shape(spec, family=family, kind=kind)
    if "id" in spec:
        _validate_spec_id(spec["id"], family=resolved_family)
        computed = compute_spec_id(spec, family=resolved_family)
        if spec["id"] != computed:
            raise SpecValidationError(
                "spec ID does not match spec payload",
                context={"expected": computed, "observed": spec["id"]},
            )
    return spec


def spec_payload_for_id(spec: Mapping[str, Any], *, family: str | None = None) -> dict[str, Any]:
    """Return spec envelope fields that participate in spec identity."""

    _validate_spec_shape(spec, family=family)
    return envelope_payload_for_id(spec)


def compute_spec_id(spec: Mapping[str, Any], *, family: str | None = None) -> str:
    """Compute the canonical content ID for a spec envelope."""

    resolved_family = _validate_spec_shape(spec, family=family)
    info = _family_info(resolved_family)
    return content_id(info.prefix, info.schema_version, spec_payload_for_id(spec, family=resolved_family))


def attach_spec_id(spec: Mapping[str, Any], *, family: str | None = None, verify_existing: bool = True) -> dict[str, Any]:
    """Return a copy of *spec* with its canonical spec ID attached."""

    resolved_family = _validate_spec_shape(spec, family=family)
    computed = compute_spec_id(spec, family=resolved_family)
    if verify_existing and "id" in spec and spec["id"] != computed:
        raise SpecValidationError(
            "spec ID does not match spec payload",
            context={"expected": computed, "observed": spec["id"]},
        )
    result = dict(spec)
    result["id"] = computed
    return result


def spec_family_for_id(spec_id: str) -> str | None:
    """Return the known spec family for an ID prefix, or ``None``."""

    try:
        parts = parse_content_id(spec_id)
    except ContentIDError as exc:
        raise SpecValidationError("invalid spec ID", context=exc.context) from exc
    family = SPEC_FAMILY_BY_PREFIX.get(parts.prefix)
    if family is None:
        return None
    expected_version = _family_info(family).schema_version
    if parts.schema_version != expected_version:
        raise SpecValidationError(
            "spec ID schema version does not match spec family",
            context={"family": family, "expected_version": expected_version, "observed_version": parts.schema_version},
        )
    return family


def spec_dir_name(family: str) -> str:
    """Return the sidecar directory name for *family*."""

    return _family_info(family).dir_name


def spec_id_prefix(family: str) -> str:
    """Return the content-ID prefix for *family*."""

    return _family_info(family).prefix


def _validate_spec_shape(spec: Mapping[str, Any], *, family: str | None = None, kind: str | None = None) -> str:
    resolved_family = family or _family_for_spec(spec)
    info = _family_info(resolved_family)
    expected_schema = None if resolved_family == "generic" else info.schema
    try:
        validate_envelope(spec, schema=expected_schema, kind=kind)
    except EnvelopeError as exc:
        raise SpecValidationError(str(exc), context=exc.context) from exc
    if spec.get("schema_version") != info.schema_version:
        raise SpecValidationError(
            "spec schema_version mismatch",
            context={"expected": info.schema_version, "observed": spec.get("schema_version")},
        )
    unknown = set(spec) - _SPEC_TOP_LEVEL_FIELDS
    if unknown:
        raise SpecValidationError("spec contains unknown top-level fields", context={"fields": sorted(unknown)})
    if "id" in spec:
        _validate_spec_id(spec["id"], family=resolved_family)
    return resolved_family


def _family_for_spec(spec: Mapping[str, Any]) -> str:
    if not isinstance(spec, Mapping):
        raise SpecValidationError("spec must be a mapping", context={"type": type(spec).__name__})
    schema = spec.get("schema")
    if schema in SPEC_FAMILY_BY_SCHEMA:
        return SPEC_FAMILY_BY_SCHEMA[schema]
    if "id" in spec:
        family = spec_family_for_id(spec["id"])
        if family is not None:
            return family
    return "generic"


def _family_info(family: str):
    if family not in SPEC_FAMILIES:
        raise SpecValidationError("unknown spec family", context={"family": family})
    return SPEC_FAMILIES[family]


def _validate_spec_id(spec_id: str, *, family: str) -> None:
    try:
        parts = parse_content_id(spec_id)
    except ContentIDError as exc:
        raise SpecValidationError("invalid spec ID", context=exc.context) from exc
    expected_prefix = spec_id_prefix(family)
    if parts.prefix != expected_prefix:
        raise SpecValidationError(
            "spec ID prefix does not match spec family",
            context={"family": family, "expected_prefix": expected_prefix, "observed_prefix": parts.prefix},
        )
    expected_version = _family_info(family).schema_version
    if parts.schema_version != expected_version:
        raise SpecValidationError(
            "spec ID schema version does not match spec family",
            context={"family": family, "expected_version": expected_version, "observed_version": parts.schema_version},
        )


__all__ = [
    "SPEC_FAMILIES",
    "attach_spec_id",
    "compute_spec_id",
    "make_spec",
    "spec_dir_name",
    "spec_family_for_id",
    "spec_id_prefix",
    "spec_payload_for_id",
    "validate_spec",
]
