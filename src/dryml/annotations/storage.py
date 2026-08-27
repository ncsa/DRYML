"""Direct immutable sidecar storage for annotation fragments."""

from __future__ import annotations

from typing import Any

from .errors import AnnotationValidationError, UnsupportedAnnotationTargetError
from .model import AnnotationFragment

FRAGMENT_ATTR = "__dryml_annotation_fragments__"


def own_fragments(target: Any) -> tuple[AnnotationFragment, ...]:
    """Return only fragments attached directly to one supplied live target.

    Args:
        target: Live object to inspect without unwrapping, importing, or calling.

    Returns:
        The immutable direct fragment tuple, or an empty tuple.

    Raises:
        AnnotationValidationError: If direct metadata was corrupted.
    """

    values = getattr(target, "__dict__", {}).get(FRAGMENT_ATTR, ())
    if not isinstance(values, tuple) or not all(isinstance(item, AnnotationFragment) for item in values):
        raise AnnotationValidationError("annotation target contains malformed direct fragment metadata")
    return values


def attach_fragment(target: Any, fragment: AnnotationFragment) -> Any:
    """Append one fragment directly and return the exact original target.

    Args:
        target: Extensible live function, class, or descriptor.
        fragment: Immutable annotation fragment to store.

    Returns:
        The identical ``target`` object.

    Raises:
        UnsupportedAnnotationTargetError: If the target cannot accept direct
            metadata. No mutation occurs in this case.
        AnnotationValidationError: If ``fragment`` or prior direct storage is
            malformed.
    """

    if not isinstance(fragment, AnnotationFragment):
        raise AnnotationValidationError("annotation storage requires an AnnotationFragment")
    if isinstance(target, property) or not hasattr(target, "__dict__"):
        raise UnsupportedAnnotationTargetError("annotation target does not support direct metadata", context={"type": type(target).__name__})
    try:
        existing = own_fragments(target)
        setattr(target, FRAGMENT_ATTR, existing + (fragment,))
    except (AttributeError, TypeError) as error:
        raise UnsupportedAnnotationTargetError("annotation target does not support direct metadata", context={"type": type(target).__name__}) from error
    return target


__all__ = ["FRAGMENT_ATTR", "attach_fragment", "own_fragments"]
