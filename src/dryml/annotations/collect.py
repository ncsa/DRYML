"""Import-free deterministic collection from supplied live annotation targets."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from typing import Any

from .model import AnnotationFragment, UnresolvedAnnotationResult
from .storage import own_fragments


def collect_fragments(target: Any, *, namespace: str | None = None, kind: str | None = None) -> tuple[AnnotationFragment, ...] | UnresolvedAnnotationResult:
    """Collect declarations from only a supplied live target or Definition/CDef.

    A symbolic definition never causes a class import; it yields an explicit
    unresolved value instead.

    Args:
        target: Live object, Definition, or ConcreteDefinition to inspect.
        namespace: Optional namespace filter.
        kind: Optional requirement/default filter.

    Returns:
        Ordered fragments or an import-free unresolved result.
    """

    if _definition_like(target):
        return UnresolvedAnnotationResult("definition collection requires a supplied live target")
    if isinstance(target, type):
        return _filter(fragments_for_class(target), namespace, kind)
    return _filter(_targets_fragments(target), namespace, kind)


def fragments_for_method(cls: type, method_name: str, *, namespace: str | None = None, kind: str | None = None) -> tuple[AnnotationFragment, ...]:
    """Collect class C3 fragments and one normal-MRO selected method body.

    Args:
        cls: Supplied live class.
        method_name: Name found by static normal MRO lookup.
        namespace: Optional namespace filter.
        kind: Optional requirement/default filter.

    Returns:
        Class fragments followed by descriptor then underlying-function fragments.

    Raises:
        TypeError: If arguments do not name a live class and string member.
        AttributeError: If no normal-MRO implementation is present.
    """

    if not isinstance(cls, type) or not isinstance(method_name, str):
        raise TypeError("fragments_for_method() requires a class and method name")
    descriptor = inspect.getattr_static(cls, method_name)
    return _filter(_dedupe((*fragments_for_class(cls), *_targets_fragments(descriptor))), namespace, kind)


def fragments_for_definition_method(defn: Any, method_name: str, *, live_cls: type | None = None, namespace: str | None = None, kind: str | None = None) -> tuple[AnnotationFragment, ...] | UnresolvedAnnotationResult:
    """Collect definition method fragments without resolving symbolic classes.

    Args:
        defn: Definition or ConcreteDefinition being inspected.
        method_name: Name on the supplied or already-live class.
        live_cls: Explicit live class; no class lookup occurs when absent.
        namespace: Optional namespace filter.
        kind: Optional requirement/default filter.

    Returns:
        Method fragments or an unresolved result when no live class is supplied.
    """

    if not isinstance(method_name, str):
        raise TypeError("method_name must be a string")
    if not _definition_like(defn):
        raise TypeError("fragments_for_definition_method() requires a Definition or ConcreteDefinition")
    if not isinstance(live_cls, type):
        return UnresolvedAnnotationResult("definition method requires a supplied live_cls", method_name)
    return fragments_for_method(live_cls, method_name, namespace=namespace, kind=kind)


def fragments_for_class(cls: type) -> tuple[AnnotationFragment, ...]:
    """Return direct class fragments in base-to-subclass C3 order.

    Args:
        cls: Supplied live class.

    Returns:
        One identity-deduplicated tuple ordered through ``cls.__mro__``.
    """

    if not isinstance(cls, type):
        raise TypeError("fragments_for_class() requires a class")
    return _dedupe(fragment for base in reversed(cls.__mro__) if base is not object for fragment in own_fragments(base))


def _targets_fragments(target: Any) -> tuple[AnnotationFragment, ...]:
    values: list[AnnotationFragment] = list(own_fragments(target))
    function = getattr(target, "__func__", None)
    if function is not None and function is not target:
        values.extend(own_fragments(function))
    return _dedupe(values)


def _filter(fragments: Iterable[AnnotationFragment], namespace: str | None, kind: str | None) -> tuple[AnnotationFragment, ...]:
    return tuple(fragment for fragment in fragments if (namespace is None or fragment.namespace == namespace) and (kind is None or fragment.kind == kind))


def _dedupe(fragments: Iterable[AnnotationFragment]) -> tuple[AnnotationFragment, ...]:
    seen: set[int] = set()
    result: list[AnnotationFragment] = []
    for fragment in fragments:
        if id(fragment) not in seen:
            seen.add(id(fragment))
            result.append(fragment)
    return tuple(result)


def _definition_like(value: Any) -> bool:
    """Recognize destination definition values without importing their modules."""

    return type(value).__module__ == "dryml.core.definition" and type(value).__name__ in {"Definition", "ConcreteDefinition"}


__all__ = ["collect_fragments", "fragments_for_class", "fragments_for_definition_method", "fragments_for_method"]
