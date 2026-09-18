"""Static deterministic collection from live annotation targets."""

from __future__ import annotations

import types
from collections.abc import Iterable
from typing import Any

from .attachment import (
    _raw_annotations,
    _has_static_descriptor_protocol,
    _is_class,
    _native_descriptor_value,
    _type_dict,
    _type_mro,
    own_annotations,
)
from .errors import AnnotationValidationError
from .model import AnnotatedMember, Annotation, _validate_key


class _RawAnnotationBudget:
    """Private whole-collection ceiling checked before carrier validation."""

    def __init__(self, limit: int) -> None:
        """Create a budget with one exact positive attachment limit."""

        self.limit = limit
        self.count = 0
        self.seen: set[int] = set()

    def reserve(self, target: Any) -> None:
        """Account for one direct raw tuple without iterating its carriers."""

        identifier = id(target)
        if identifier in self.seen:
            return
        values = _raw_annotations(target)
        if self.count + len(values) > self.limit:
            raise AnnotationValidationError(
                "annotation collection exceeds raw attachment limit")
        self.seen.add(identifier)
        self.count += len(values)


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

    return _collect_annotations(target, key=key)


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

    return _annotations_for_class(cls, key=key)


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

    return _annotations_for_method(cls, method_name, key=key)


def annotations_for_members(
    cls: type,
    *,
    key: str | None = None,
    after: type | None = None,
) -> tuple[AnnotatedMember, ...]:
    """Collect annotated member declarations in base-to-subclass C3 order.

    A declaration is included when its direct descriptor annotations match
    ``key``. Later unannotated declarations with the same name are also
    included, preserving observable shadow evidence for consumers. Descriptor
    inspection is static: descriptors are not bound and dynamic attribute hooks
    are not invoked. Known ``staticmethod`` and ``classmethod`` descriptors
    contribute direct descriptor entries followed by entries on their underlying
    function; custom descriptors contribute only direct entries.

    Args:
        cls: A supplied live class whose declaration namespaces are inspected
            through the native static attachment boundary.
        key: Optional exact built-in consumer key used to filter each member's
            annotations before matching and shadow detection.
        after: Optional class boundary excluded with all of its base classes.
            Collection begins with the next subclass in base-to-subclass order.

    Returns:
        An immutable tuple of raw declaration evidence. Each member's annotation
        tuple is identity-deduplicated in direct collection order.

    Raises:
        AnnotationValidationError: If ``cls``, ``key``, or ``after`` is invalid,
            ``after`` is outside ``cls``'s MRO, or an inspected direct attachment
            tuple is malformed.
        UnsupportedAnnotationTargetError: If an inspectable descriptor member
            cannot be inspected through the static attachment boundary.

    Side Effects:
        None. The collector neither binds or invokes descriptors nor assigns
        consumer semantics to the returned declaration evidence.
    """

    if not _is_class(cls):
        raise AnnotationValidationError("annotations_for_members() requires a class")
    _validate_filter_key(key)
    mro = _type_mro(cls)
    if after is not None and (not _is_class(after) or after not in mro):
        raise AnnotationValidationError("after must be a class in the supplied class MRO")
    matching_names: set[str] = set()
    members: list[AnnotatedMember] = []
    reached_boundary = after is None
    for owner in reversed(mro):
        if owner is after:
            reached_boundary = True
            continue
        if not reached_boundary:
            continue
        if owner is object:
            continue
        for name, descriptor in _type_dict(owner).items():
            if not _is_member_target(descriptor):
                if name in matching_names:
                    members.append(AnnotatedMember(owner, name, descriptor, ()))
                continue
            annotations = _filter(_target_annotations(descriptor), key)
            if annotations:
                matching_names.add(name)
                members.append(AnnotatedMember(owner, name, descriptor, annotations))
            elif name in matching_names:
                members.append(AnnotatedMember(owner, name, descriptor, ()))
    return tuple(members)


def _target_annotations(target: Any) -> tuple[Annotation, ...]:
    """Collect one direct target and known descriptor function without binding."""

    values: list[Annotation] = []
    for source in _annotation_targets(target):
        values.extend(own_annotations(source))
    return _dedupe(values)


def _annotation_targets(target: Any) -> tuple[Any, ...]:
    """Return direct annotation carriers without reading their metadata."""

    descriptor_type = _known_descriptor_type(target)
    if descriptor_type is None:
        return (target, )
    return (target,
            _native_descriptor_value(descriptor_type, "__func__", target))


def _class_annotation_targets(cls: type) -> tuple[type, ...]:
    """
    Return class declaration carriers in the established reversed-C3 order.
    """

    if not _is_class(cls):
        raise AnnotationValidationError(
            "annotations_for_class() requires a class")
    return tuple(base for base in reversed(_type_mro(cls))
                 if base is not object)


def _method_annotation_targets(cls: type, method_name: str) -> tuple[Any, ...]:
    """
    Return class and selected-member carriers without binding descriptors.
    """

    if not _is_class(cls):
        raise AnnotationValidationError(
            "annotations_for_method() requires a class")
    if type(method_name) is not str:
        raise AnnotationValidationError("method name must be a string")
    for base in _type_mro(cls):
        namespace = _type_dict(base)
        if method_name in namespace:
            return (*_class_annotation_targets(cls),
                    *_annotation_targets(namespace[method_name]))
    raise AnnotationValidationError(
        "method is not declared on the supplied class")


def _reserve_annotation_targets(
    targets: Iterable[Any], budget: _RawAnnotationBudget,
) -> None:
    """Reserve every unique raw attachment tuple before carrier validation."""

    for target in targets:
        budget.reserve(target)


def _collect_annotations(
    target: Any,
    *,
    key: str | None = None,
    budget: _RawAnnotationBudget | None = None,
) -> tuple[Annotation, ...]:
    """Collect one target using an optional already-reserved private budget."""

    if _is_class(target):
        return _annotations_for_class(target, key=key, budget=budget)
    _validate_filter_key(key)
    targets = _annotation_targets(target)
    if budget is not None:
        _reserve_annotation_targets(targets, budget)
    return _filter(
        _dedupe(annotation for source in targets
                for annotation in own_annotations(source)), key)


def _annotations_for_class(
    cls: type,
    *,
    key: str | None = None,
    budget: _RawAnnotationBudget | None = None,
) -> tuple[Annotation, ...]:
    """
    Collect class entries with an optional capture-wide raw attachment budget.
    """

    targets = _class_annotation_targets(cls)
    _validate_filter_key(key)
    if budget is not None:
        _reserve_annotation_targets(targets, budget)
    return _filter(
        _dedupe(annotation for source in targets
                for annotation in own_annotations(source)), key)


def _annotations_for_method(
    cls: type,
    method_name: str,
    *,
    key: str | None = None,
    budget: _RawAnnotationBudget | None = None,
) -> tuple[Annotation, ...]:
    """Collect selected-method entries with an optional capture-wide budget."""

    if not _is_class(cls):
        raise AnnotationValidationError(
            "annotations_for_method() requires a class")
    if type(method_name) is not str:
        raise AnnotationValidationError("method name must be a string")
    _validate_filter_key(key)
    targets = _method_annotation_targets(cls, method_name)
    if budget is not None:
        _reserve_annotation_targets(targets, budget)
    return _filter(
        _dedupe(annotation for source in targets
                for annotation in own_annotations(source)), key)


def _is_member_target(target: Any) -> bool:
    """Return whether a declaration can carry direct annotation evidence."""

    target_type = type(target)
    return (
        _is_class(target)
        or target_type is types.FunctionType
        or _known_descriptor_type(target) is not None
        or (
            target_type not in (types.GetSetDescriptorType, types.MemberDescriptorType)
            and _has_static_descriptor_protocol(target_type)
        )
    )


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


__all__ = ["annotations_for_class", "annotations_for_members", "annotations_for_method", "collect_annotations"]
