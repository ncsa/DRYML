"""Collection helpers for DRYML annotation fragments."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from .decorators import FRAGMENT_ATTR
from .model import AnnotationFragment
from dryml.core2.methods import bound_method_parts, descriptor_function


def fragments_for(target: Any, *, namespace: str | None = None, kind: str | None = None, include_inherited: bool = True) -> tuple[AnnotationFragment, ...]:
    """Collect fragments for a class, function, method, or arbitrary target.

    Args:
        target: Live Python target to inspect without invoking it.
        namespace: Optional namespace filter such as ``"environment"``.
        kind: Optional fragment-kind filter such as ``"requirement"``.
        include_inherited: Whether class and method targets include inherited
            class fragments and concrete method implementation fragments.

    Returns:
        Matching fragments in deterministic collection order.
    """

    if isinstance(target, type):
        fragments = fragments_for_class(target, namespace=namespace, kind=kind) if include_inherited else own_fragments(target)
    elif bound_method_parts(target) is not None:
        method_target = _method_target_from_bound_method(target)
        if include_inherited and method_target is not None:
            owner, method_name = method_target
            fragments = fragments_for_method(owner, method_name, namespace=namespace, kind=kind)
        else:
            fragments = own_fragments(descriptor_function(target))
    elif inspect.isfunction(target):
        owner = _owner_class_for_function(target)
        if include_inherited and owner is not None and getattr(target, "__name__", None):
            fragments = fragments_for_method(owner, target.__name__, namespace=namespace, kind=kind)
        else:
            fragments = own_fragments(target)
    else:
        unwrapped = descriptor_function(target)
        if unwrapped is not target:
            owner = _owner_class_for_function(unwrapped)
            method_name = getattr(unwrapped, "__name__", None)
            if include_inherited and owner is not None and method_name:
                fragments = fragments_for_method(owner, method_name)
            else:
                fragments = _dedupe_fragments_preserve_order(
                    (*own_fragments(target), *own_fragments(unwrapped))
                )
        else:
            fragments = own_fragments(target)
    return _filter_fragments(fragments, namespace=namespace, kind=kind)


def own_fragments(target: Any, *, namespace: str | None = None, kind: str | None = None) -> tuple[AnnotationFragment, ...]:
    """Return fragments directly attached to *target*.

    This helper performs no owner inference, MRO traversal, or descriptor
    unwrapping. It is the lowest-level public collection API for reading exactly
    the fragments stored on one live object.

    Args:
        target: Object whose sidecar fragment storage should be read.
        namespace: Optional namespace filter.
        kind: Optional fragment-kind filter.

    Returns:
        A stable tuple of directly attached annotation fragments.
    """

    return _filter_fragments(_own_fragments(target), namespace=namespace, kind=kind)


def fragments_for_class(cls: type, *, namespace: str | None = None, kind: str | None = None) -> tuple[AnnotationFragment, ...]:
    """Return class fragments in deterministic base-to-subclass MRO order.

    Args:
        cls: Class whose class-level annotation fragments should be collected.
        namespace: Optional namespace filter.
        kind: Optional fragment-kind filter.

    Returns:
        Fragments attached directly to base classes and then ``cls``.
    """

    if not isinstance(cls, type):
        raise TypeError("fragments_for_class() requires a class.")

    fragments: list[AnnotationFragment] = []
    bases: list[type] = []
    for base in cls.__mro__[1:]:
        if base is object:
            continue
        insert_at = len(bases)
        for index, existing in enumerate(bases):
            if issubclass(existing, base):
                insert_at = index
                break
        bases.insert(insert_at, base)
    bases.append(cls)
    for base in bases:
        fragments.extend(_own_fragments(base))
    return _filter_fragments(tuple(fragments), namespace=namespace, kind=kind)


def fragments_for_method(
    cls: type,
    method_name: str,
    *,
    namespace: str | None = None,
    kind: str | None = None,
    include_class_fragments: bool = True,
    include_method_fragments: bool = True,
) -> tuple[AnnotationFragment, ...]:
    """Collect class and concrete method implementation fragments.

    Args:
        cls: Class on which Python method lookup should be performed.
        method_name: Attribute name for the method-like class attribute.
        namespace: Optional namespace filter.
        kind: Optional fragment-kind filter.
        include_class_fragments: Whether to include class fragments through MRO.
        include_method_fragments: Whether to include fragments attached to the
            concrete method implementation descriptor or underlying function.

    Returns:
        Class fragments first, followed by fragments for the concrete method
        implementation selected by :func:`inspect.getattr_static`.

    Raises:
        TypeError: If ``cls`` is not a class or ``method_name`` is not a string.
        AttributeError: If ``method_name`` is not present through normal Python
            class attribute lookup.

    Notes:
        Inherited method implementations include the inherited implementation's
        fragments. Overridden subclass methods do not include base method
        fragments by default; put requirements on the class when they are part
        of the inherited class contract.
    """

    if not isinstance(cls, type):
        raise TypeError("fragments_for_method() requires a class.")
    if not isinstance(method_name, str):
        raise TypeError("method_name must be a string.")

    fragments: list[AnnotationFragment] = []
    if include_class_fragments:
        fragments.extend(fragments_for_class(cls))
    if include_method_fragments:
        raw_attr = _get_static_method_attribute(cls, method_name)
        for target in _descriptor_fragment_targets(raw_attr):
            fragments.extend(_own_fragments(target))
    return _filter_fragments(_dedupe_fragments_preserve_order(fragments), namespace=namespace, kind=kind)


def fragments_for_definition_method(
    defn: Any,
    method_name: str,
    *,
    namespace: str | None = None,
    kind: str | None = None,
    include_class_fragments: bool = True,
    include_method_fragments: bool = True,
) -> tuple[AnnotationFragment, ...]:
    """Collect method requirements for a Definition/CDef-like subject.

    Args:
        defn: Object exposing a live class through ``.cls`` or ``.definition``.
        method_name: Attribute name to collect on the subject class.
        namespace: Optional namespace filter.
        kind: Optional fragment-kind filter.
        include_class_fragments: Whether class MRO fragments should be included.
        include_method_fragments: Whether method implementation fragments should
            be included.

    Returns:
        The same fragment sequence as :func:`fragments_for_method` for the
        resolved subject class.

    Raises:
        TypeError: If a live class cannot be resolved without building an
            object.
        AttributeError: If the resolved class has no matching method.
    """

    cls = _class_from_definition_like(defn)
    return fragments_for_method(
        cls,
        method_name,
        namespace=namespace,
        kind=kind,
        include_class_fragments=include_class_fragments,
        include_method_fragments=include_method_fragments,
    )


def fragments_for_callable(fn: Any, *, namespace: str | None = None, kind: str | None = None) -> tuple[AnnotationFragment, ...]:
    """Return fragments for a callable, including owning class fragments for methods."""

    return fragments_for(fn, namespace=namespace, kind=kind)


def collect_fragments(targets: Iterable[Any] | Any, *, provider_fragments: Iterable[AnnotationFragment] = (), namespace: str | None = None, kind: str | None = None) -> tuple[AnnotationFragment, ...]:
    """Collect target and provider-like synthetic fragments in deterministic order.

    Target fragments are collected first. Provider fragments are appended after
    target fragments, then namespace/kind filtering is applied.
    """

    fragments: list[AnnotationFragment] = []
    for target in _iter_targets(targets):
        fragments.extend(fragments_for(target, namespace=namespace, kind=kind))
    fragments.extend(provider_fragments)
    return _filter_fragments(tuple(fragments), namespace=namespace, kind=kind)


def _own_fragments(target: Any) -> tuple[AnnotationFragment, ...]:
    return tuple(getattr(target, "__dict__", {}).get(FRAGMENT_ATTR, ()))


def _filter_fragments(fragments: Iterable[AnnotationFragment], *, namespace: str | None, kind: str | None) -> tuple[AnnotationFragment, ...]:
    return tuple(fragment for fragment in fragments if (namespace is None or fragment.namespace == namespace) and (kind is None or fragment.kind == kind))


def _dedupe_fragments_preserve_order(fragments: Iterable[AnnotationFragment]) -> tuple[AnnotationFragment, ...]:
    seen: set[int] = set()
    result: list[AnnotationFragment] = []
    for fragment in fragments:
        key = id(fragment)
        if key in seen:
            continue
        seen.add(key)
        result.append(fragment)
    return tuple(result)


def _descriptor_fragment_targets(raw_attr: Any) -> tuple[Any, ...]:
    """Return raw descriptor/function candidates that may carry fragments."""

    targets = [raw_attr]
    unwrapped = descriptor_function(raw_attr)
    if unwrapped is not raw_attr:
        targets.append(unwrapped)
    return _dedupe_targets_preserve_order(targets)


def _dedupe_targets_preserve_order(targets: Iterable[Any]) -> tuple[Any, ...]:
    seen: set[int] = set()
    result: list[Any] = []
    for target in targets:
        key = id(target)
        if key in seen:
            continue
        seen.add(key)
        result.append(target)
    return tuple(result)


def _get_static_method_attribute(cls: type, method_name: str) -> Any:
    return inspect.getattr_static(cls, method_name)


def _method_target_from_bound_method(target: Any) -> tuple[type, str] | None:
    parts = bound_method_parts(target)
    if parts is None:
        return None
    owner_obj, func = parts
    if owner_obj is None:
        return None
    cls = owner_obj if isinstance(owner_obj, type) else type(owner_obj)
    method_name = getattr(func, "__name__", None)
    if not isinstance(method_name, str):
        return None
    return cls, method_name


def _class_from_definition_like(defn: Any) -> type:
    candidate = _definition_cls_candidate(defn)
    if isinstance(candidate, type):
        return candidate
    raise TypeError(f"Could not resolve a live class from {type(defn).__name__} without building an object.")


def _definition_cls_candidate(defn: Any) -> Any:
    if isinstance(defn, type):
        return defn
    if hasattr(defn, "cls"):
        return getattr(defn, "cls")
    if hasattr(defn, "definition"):
        definition = getattr(defn, "definition")
        if hasattr(definition, "cls"):
            return getattr(definition, "cls")
    return None


def _iter_targets(targets: Iterable[Any] | Any) -> tuple[Any, ...]:
    if isinstance(targets, (str, bytes, Mapping)) or not isinstance(targets, Iterable):
        return (targets,)
    return tuple(targets)


def _owner_class_for_function(fn: Any) -> type | None:
    qualname = getattr(fn, "__qualname__", "")
    if "." not in qualname:
        return None
    module = sys.modules.get(getattr(fn, "__module__", ""))
    if module is None:
        return None
    current: Any = module
    for part in qualname.rsplit(".", 1)[0].replace(".<locals>", "").split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, type) else None


__all__ = [
    "collect_fragments",
    "fragments_for",
    "fragments_for_callable",
    "fragments_for_class",
    "fragments_for_definition_method",
    "fragments_for_method",
    "own_fragments",
]
