"""Static direct attachment for process-local annotation entries."""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Any

from .errors import AnnotationValidationError, UnsupportedAnnotationTargetError
from .model import Annotation

ANNOTATION_ATTR = "__dryml_annotations__"

_OBJECT_SETATTR = object.__dict__["__setattr__"]
_TYPE_DICT = type.__dict__["__dict__"]
_TYPE_MRO = type.__dict__["__mro__"]
_TYPE_NAME = type.__dict__["__name__"]
_TYPE_SETATTR = type.__dict__["__setattr__"]


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
        AnnotationValidationError: If the direct attachment attribute is not an
            exact built-in tuple containing only exact :class:`Annotation`
            entries.
    """

    values = _target_dict(target).get(ANNOTATION_ATTR, ())
    if type(values) is not tuple or any(type(item) is not Annotation for item in values):
        raise AnnotationValidationError("annotation target contains malformed direct annotation metadata")
    return values


def attach_annotation(target: Any, annotation: Annotation) -> Any:
    """Append one annotation directly and return the unchanged target object.

    Args:
        target: A supported function, class, known method descriptor, or safe
            custom descriptor. Same-target concurrent attachment is unsupported;
            callers must finish setup before sharing the target.
        annotation: The exact immutable :class:`Annotation` carrier to attach.

    Returns:
        The exact supplied ``target`` object.

    Raises:
        AnnotationValidationError: If ``annotation`` is not an exact
            :class:`Annotation` or existing direct metadata is malformed.
        UnsupportedAnnotationTargetError: If ``target`` cannot be inspected and
            mutated through native, non-binding object operations. No mutation
            occurs in this case.
    """

    if type(annotation) is not Annotation:
        raise AnnotationValidationError("annotation attachment requires an Annotation")
    existing = own_annotations(target)
    if _has_data_descriptor(type(target), ANNOTATION_ATTR):
        _unsupported(target)
    setter = type.__setattr__ if _is_class(target) else object.__setattr__
    try:
        setter(target, ANNOTATION_ATTR, existing + (annotation,))
    except (AttributeError, TypeError) as error:
        _unsupported(target, cause=error)
    return target


def _target_dict(target: Any) -> Mapping[str, Any]:
    """Return a statically accessible direct dictionary or reject the target."""

    target_type = type(target)
    if _is_class(target):
        if _static_type_attr(target_type, "__setattr__") is not _TYPE_SETATTR:
            _unsupported(target)
        return _type_dict(target)
    if _static_type_attr(target_type, "__setattr__") is not _OBJECT_SETATTR:
        _unsupported(target)
    if not _is_known_native_target_type(target_type) and not _has_static_descriptor_protocol(target_type):
        _unsupported(target)
    dictionary_descriptor = _instance_dict_descriptor(target_type)
    if dictionary_descriptor is None:
        _unsupported(target)
    return _descriptor_value(dictionary_descriptor, target, target_type)


def _has_static_descriptor_protocol(target_type: type) -> bool:
    """Return whether a custom target declares ``__get__`` without lookup hooks."""

    return any("__get__" in _type_dict(base) for base in _type_mro(target_type))


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


def _instance_dict_descriptor(target_type: type) -> Any | None:
    """Return the first native instance-dictionary descriptor in static MRO order."""

    for base in _type_mro(target_type):
        descriptor = _type_dict(base).get("__dict__")
        if descriptor is not None:
            return descriptor if isinstance(descriptor, types.GetSetDescriptorType) else None
    return None


def _has_data_descriptor(target_type: type, name: str) -> bool:
    """Return whether normal assignment would invoke a statically found descriptor."""

    try:
        descriptor = _static_type_attr(target_type, name)
    except AttributeError:
        return False
    descriptor_type = type(descriptor)
    return any(
        "__set__" in _type_dict(base) or "__delete__" in _type_dict(base)
        for base in _type_mro(descriptor_type)
    )


def _unsupported(target: Any, *, cause: Exception | None = None) -> None:
    """Raise the bounded unsupported-target error used by static attachment."""

    error = UnsupportedAnnotationTargetError(
        "annotation target does not support static direct metadata",
        context={"type": _type_name(type(target))},
    )
    if cause is not None:
        raise error from cause
    raise error


def _static_type_attr(cls: type, name: str) -> Any:
    """Return one raw class attribute through the native MRO."""

    for base in _type_mro(cls):
        namespace = _type_dict(base)
        if name in namespace:
            return namespace[name]
    raise AttributeError(name)


def _descriptor_value(descriptor: Any, target: Any, owner: type) -> Any:
    """Invoke one native storage descriptor without target-side lookup."""

    return type(descriptor).__get__(descriptor, target, owner)


def _native_descriptor_value(owner: type, name: str, target: Any) -> Any:
    """Read a built-in storage member while bypassing subclass overrides."""

    descriptor = _type_dict(owner)[name]
    return _descriptor_value(descriptor, target, type(target))


def _type_dict(cls: type) -> Mapping[str, Any]:
    """Return a class namespace without invoking its metaclass hooks."""

    return _descriptor_value(_TYPE_DICT, cls, type(cls))


def _type_mro(cls: type) -> tuple[type, ...]:
    """Return a class MRO without invoking its metaclass hooks."""

    return _descriptor_value(_TYPE_MRO, cls, type(cls))


def _type_name(cls: type) -> str:
    """Return a class name without invoking its metaclass hooks."""

    return _descriptor_value(_TYPE_NAME, cls, type(cls))


__all__ = ["ANNOTATION_ATTR", "attach_annotation", "own_annotations"]
