"""Static direct attachment for process-local annotation entries."""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Any

from .errors import AnnotationValidationError, UnsupportedAnnotationTargetError
from .model import Annotation

ANNOTATION_ATTR = "__dryml_annotations__"


def own_annotations(target: Any) -> tuple[Annotation, ...]:
    """Return annotations attached directly to one supported live target.

    Args:
        target: A function, class, known method descriptor, or safe custom
            descriptor. Inspection never binds descriptors or invokes dynamic
            attribute hooks.

    Returns:
        The direct immutable annotation tuple in attachment order, or an empty
        tuple when no direct annotations were attached.

    Raises:
        UnsupportedAnnotationTargetError: If static direct inspection is unsafe
            for ``target``.
        AnnotationValidationError: If the direct attachment attribute is not a
            tuple containing only :class:`Annotation` entries.
    """

    values = _target_dict(target).get(ANNOTATION_ATTR, ())
    if not isinstance(values, tuple) or not all(isinstance(item, Annotation) for item in values):
        raise AnnotationValidationError("annotation target contains malformed direct annotation metadata")
    return values


def attach_annotation(target: Any, annotation: Annotation) -> Any:
    """Append one annotation directly and return the unchanged target object.

    Args:
        target: A supported function, class, known method descriptor, or safe
            custom descriptor. Same-target concurrent attachment is unsupported;
            callers must finish setup before sharing the target.
        annotation: The immutable :class:`Annotation` carrier to attach.

    Returns:
        The exact supplied ``target`` object.

    Raises:
        AnnotationValidationError: If ``annotation`` is not an Annotation or
            existing direct metadata is malformed.
        UnsupportedAnnotationTargetError: If ``target`` cannot be inspected and
            mutated through native, non-binding object operations. No mutation
            occurs in this case.
    """

    if not isinstance(annotation, Annotation):
        raise AnnotationValidationError("annotation attachment requires an Annotation")
    existing = own_annotations(target)
    if _is_class(target):
        type.__setattr__(target, ANNOTATION_ATTR, existing + (annotation,))
    else:
        object.__setattr__(target, ANNOTATION_ATTR, existing + (annotation,))
    return target


def _target_dict(target: Any) -> Mapping[str, Any]:
    """Return a statically accessible direct dictionary or reject the target."""

    target_type = type(target)
    if issubclass(target_type, type):
        if _native_type_attr(target_type, "__setattr__") is not type.__setattr__:
            _unsupported(target)
        return type.__getattribute__(target, "__dict__")
    if _is_known_native_target_type(target_type):
        if _native_type_attr(target_type, "__setattr__") is not object.__setattr__:
            _unsupported(target)
        return object.__getattribute__(target, "__dict__")
    if (
        _native_type_attr(target_type, "__setattr__") is not object.__setattr__
        or not _has_static_descriptor_protocol(target)
        or not _has_real_instance_dict(target)
    ):
        _unsupported(target)
    return object.__getattribute__(target, "__dict__")


def _has_static_descriptor_protocol(target: Any) -> bool:
    """Return whether a custom target declares ``__get__`` without lookup hooks."""

    return any(
        "__get__" in _native_type_attr(base, "__dict__")
        for base in _native_type_attr(type(target), "__mro__")
    )


def _is_class(target: Any) -> bool:
    """Return whether ``target`` is a class without target attribute access."""

    return issubclass(type(target), type)


def _is_known_native_target_type(target_type: type) -> bool:
    """Return whether a type is a supported built-in target category."""

    return (
        target_type is types.FunctionType
        or issubclass(target_type, staticmethod)
        or issubclass(target_type, classmethod)
    )


def _has_real_instance_dict(target: Any) -> bool:
    """Return whether the target inherits Python's native instance dictionary."""

    for base in _native_type_attr(type(target), "__mro__"):
        descriptor = _native_type_attr(base, "__dict__").get("__dict__")
        if descriptor is not None:
            return isinstance(descriptor, types.GetSetDescriptorType)
    return False


def _unsupported(target: Any) -> None:
    """Raise the bounded unsupported-target error used by static attachment."""

    raise UnsupportedAnnotationTargetError(
        "annotation target does not support static direct metadata",
        context={"type": type(target).__name__},
    )


def _native_type_attr(cls: type, name: str) -> Any:
    """Read one class attribute without a custom metaclass lookup hook."""

    return type.__getattribute__(cls, name)


__all__ = ["ANNOTATION_ATTR", "attach_annotation", "own_annotations"]
