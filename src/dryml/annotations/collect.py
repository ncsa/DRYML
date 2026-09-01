"""Static deterministic collection from live annotation targets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .attachment import _is_class, _native_descriptor_value, _type_dict, _type_mro, own_annotations
from .errors import AnnotationValidationError
from .model import Annotation, _validate_key


def collect_annotations(target: Any, *, key: str | None = None) -> tuple[Annotation, ...]:
    """Collect annotations from one supplied live target without binding it.

    Classes are traversed in reversed C3 order. Other supported targets return
    only direct entries, except known static and class method descriptors also
    contribute their underlying function after descriptor entries.

    Args:
        target: A supplied live class or supported directly inspectable target.
        key: Optional exact built-in consumer key used to filter collected
            entries.

    Returns:
        An immutable, identity-deduplicated tuple in deterministic collection
        order.

    Raises:
        AnnotationValidationError: If ``key`` is invalid or direct metadata is
            malformed.
        UnsupportedAnnotationTargetError: If the target cannot be inspected
            statically by the attachment boundary.
    """

    if _is_class(target):
        return annotations_for_class(target, key=key)
    _validate_filter_key(key)
    return _filter(_target_annotations(target), key)


def annotations_for_class(cls: type, *, key: str | None = None) -> tuple[Annotation, ...]:
    """Collect direct class annotations in base-to-subclass reversed C3 order.

    Args:
        cls: A supplied live class whose MRO is inspected without dynamic hooks.
        key: Optional exact built-in consumer key used to filter collected
            entries.

    Returns:
        An immutable identity-deduplicated annotation tuple.

    Raises:
        AnnotationValidationError: If ``cls`` is not a class, ``key`` is invalid,
            or a direct attachment tuple is malformed.
        UnsupportedAnnotationTargetError: If a class cannot be inspected through
            the native static attachment boundary.
    """

    if not _is_class(cls):
        raise AnnotationValidationError("annotations_for_class() requires a class")
    _validate_filter_key(key)
    values = (
        annotation
        for base in reversed(_type_mro(cls))
        if base is not object
        for annotation in own_annotations(base)
    )
    return _filter(_dedupe(values), key)


def annotations_for_method(cls: type, method_name: str, *, key: str | None = None) -> tuple[Annotation, ...]:
    """Collect class and one statically selected method's annotations.

    The normal MRO selects exactly one member. Class declarations appear first,
    followed by direct descriptor declarations and then direct entries on the
    underlying function of known static and class method descriptors.

    Args:
        cls: A supplied live class.
        method_name: Exact built-in string name resolved by non-binding normal
            MRO lookup.
        key: Optional exact built-in consumer key used to filter collected
            entries.

    Returns:
        An immutable identity-deduplicated annotation tuple.

    Raises:
        AnnotationValidationError: If arguments are malformed, the method is
            absent, or any inspected metadata is malformed.
        UnsupportedAnnotationTargetError: If static inspection of the selected
            descriptor is unsafe.
    """

    if not _is_class(cls):
        raise AnnotationValidationError("annotations_for_method() requires a class")
    if type(method_name) is not str:
        raise AnnotationValidationError("method name must be a string")
    _validate_filter_key(key)
    for base in _type_mro(cls):
        namespace = _type_dict(base)
        if method_name in namespace:
            descriptor = namespace[method_name]
            break
    else:
        raise AnnotationValidationError("method is not declared on the supplied class")
    return _filter(_dedupe((*annotations_for_class(cls), *_target_annotations(descriptor))), key)


def _target_annotations(target: Any) -> tuple[Annotation, ...]:
    """Collect one direct target and known descriptor function without binding."""

    values: list[Annotation] = list(own_annotations(target))
    descriptor_type = _known_descriptor_type(target)
    if descriptor_type is not None:
        function = _native_descriptor_value(descriptor_type, "__func__", target)
        values.extend(own_annotations(function))
    return _dedupe(values)


def _filter(annotations: tuple[Annotation, ...], key: str | None) -> tuple[Annotation, ...]:
    """Apply the already-validated exact key filter after deduplication."""

    if key is None:
        return annotations
    return tuple(annotation for annotation in annotations if annotation.key == key)


def _dedupe(annotations: Iterable[Annotation]) -> tuple[Annotation, ...]:
    """Keep first occurrences by carrier identity, not value equality."""

    seen: set[int] = set()
    result: list[Annotation] = []
    for annotation in annotations:
        if id(annotation) not in seen:
            seen.add(id(annotation))
            result.append(annotation)
    return tuple(result)


def _validate_filter_key(key: str | None) -> None:
    """Validate an optional filter without assigning semantics to its key."""

    if key is not None:
        _validate_key(key)


def _known_descriptor_type(target: Any) -> type | None:
    """Return the built-in descriptor owner used for native unwrapping."""

    target_type = type(target)
    if issubclass(target_type, staticmethod):
        return staticmethod
    if issubclass(target_type, classmethod):
        return classmethod
    return None


__all__ = ["annotations_for_class", "annotations_for_method", "collect_annotations"]
