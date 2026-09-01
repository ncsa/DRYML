"""Passive key/value annotations for deterministic live-target collection."""

from .attachment import ANNOTATION_ATTR, attach_annotation, own_annotations
from .collect import annotations_for_class, annotations_for_method, collect_annotations
from .errors import AnnotationError, AnnotationValidationError, UnsupportedAnnotationTargetError
from .model import Annotation

__all__ = [
    "Annotation",
    "ANNOTATION_ATTR",
    "attach_annotation",
    "own_annotations",
    "collect_annotations",
    "annotations_for_class",
    "annotations_for_method",
    "AnnotationError",
    "AnnotationValidationError",
    "UnsupportedAnnotationTargetError",
]
