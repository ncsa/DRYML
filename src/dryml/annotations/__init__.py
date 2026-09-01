"""Passive key/value annotations for deterministic live-target collection."""

from .attachment import ANNOTATION_ATTR, attach_annotation, own_annotations
from .collect import annotations_for_class, annotations_for_members, annotations_for_method, collect_annotations
from .errors import AnnotationError, AnnotationValidationError, UnsupportedAnnotationTargetError
from .model import AnnotatedMember, Annotation

__all__ = [
    "Annotation",
    "AnnotatedMember",
    "ANNOTATION_ATTR",
    "attach_annotation",
    "own_annotations",
    "collect_annotations",
    "annotations_for_class",
    "annotations_for_members",
    "annotations_for_method",
    "AnnotationError",
    "AnnotationValidationError",
    "UnsupportedAnnotationTargetError",
]
