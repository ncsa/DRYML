"""Annotation spec helpers using the records annotation family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.records import attach_spec_id, compute_spec_id, make_spec, spec_payload_for_id, validate_spec
from dryml.records.errors import SpecValidationError

from .errors import AnnotationValidationError
from .model import AnnotationFragment


def make_annotation_spec(fragment: AnnotationFragment, *, kind: str = "annotation_fragment", metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a canonical annotation sidecar spec for *fragment*."""

    if not isinstance(fragment, AnnotationFragment):
        raise AnnotationValidationError("make_annotation_spec expects an AnnotationFragment")
    return make_spec(family="annotation", kind=kind, payload=fragment.to_data(), metadata=metadata)


def validate_annotation_spec(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate an annotation spec envelope and semantic fragment payload."""

    try:
        validate_spec(spec, family="annotation")
        AnnotationFragment.from_data(spec["payload"])
    except (SpecValidationError, AnnotationValidationError) as exc:
        context = getattr(exc, "context", {})
        raise AnnotationValidationError(str(exc), context=context) from exc
    return spec


def compute_annotation_id(spec: Mapping[str, Any]) -> str:
    """Compute the stable ``annotation-v1-*`` ID for an annotation spec."""

    validate_annotation_spec(spec)
    spec_id = compute_spec_id(spec, family="annotation")
    if not spec_id.startswith("annotation-v1-"):
        raise AnnotationValidationError("annotation ID has unexpected prefix", context={"id": spec_id})
    return spec_id


def attach_annotation_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *spec* with its canonical annotation ID attached."""

    attached = attach_spec_id(spec, family="annotation")
    validate_annotation_spec(attached)
    return attached


def annotation_payload_for_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return annotation spec fields that participate in ID computation."""

    validate_annotation_spec(spec)
    return spec_payload_for_id(spec, family="annotation")


__all__ = [
    "annotation_payload_for_id",
    "attach_annotation_id",
    "compute_annotation_id",
    "make_annotation_spec",
    "validate_annotation_spec",
]
