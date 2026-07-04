"""Collection helpers for DRYML annotation fragments."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterable
from typing import Any

from .decorators import FRAGMENT_ATTR
from .model import AnnotationFragment


def fragments_for(target: Any, *, namespace: str | None = None, kind: str | None = None, include_inherited: bool = True) -> tuple[AnnotationFragment, ...]:
    """Collect fragments for a class, function, method, or arbitrary target."""

    if isinstance(target, type):
        fragments = fragments_for_class(target, namespace=namespace, kind=kind) if include_inherited else _own_fragments(target)
    elif inspect.ismethod(target):
        owner = target.__self__ if isinstance(target.__self__, type) else type(target.__self__)
        fragments = fragments_for_class(owner, namespace=namespace, kind=kind) + _own_fragments(target.__func__)
    elif inspect.isfunction(target):
        owner = _owner_class_for_function(target)
        fragments = (fragments_for_class(owner, namespace=namespace, kind=kind) if include_inherited and owner is not None else ()) + _own_fragments(target)
    else:
        fragments = _own_fragments(target)
    return _filter(fragments, namespace=namespace, kind=kind)


def fragments_for_class(cls: type, *, namespace: str | None = None, kind: str | None = None) -> tuple[AnnotationFragment, ...]:
    """Return class fragments in deterministic base-to-subclass MRO order."""

    fragments: list[AnnotationFragment] = []
    bases = [base for base in cls.__mro__[1:] if base is not object] + [cls]
    for base in bases:
        fragments.extend(_own_fragments(base))
    return _filter(tuple(fragments), namespace=namespace, kind=kind)


def fragments_for_callable(fn: Any, *, namespace: str | None = None, kind: str | None = None) -> tuple[AnnotationFragment, ...]:
    """Return fragments for a callable, including owning class fragments for methods."""

    return fragments_for(fn, namespace=namespace, kind=kind)


def collect_fragments(targets: Iterable[Any], *, provider_fragments: Iterable[AnnotationFragment] = (), namespace: str | None = None, kind: str | None = None) -> tuple[AnnotationFragment, ...]:
    """Collect target and provider-like synthetic fragments in deterministic order."""

    fragments: list[AnnotationFragment] = []
    for target in targets:
        fragments.extend(fragments_for(target, namespace=namespace, kind=kind))
    fragments.extend(provider_fragments)
    return _filter(tuple(fragments), namespace=namespace, kind=kind)


def _own_fragments(target: Any) -> tuple[AnnotationFragment, ...]:
    return tuple(getattr(target, "__dict__", {}).get(FRAGMENT_ATTR, ()))


def _filter(fragments: Iterable[AnnotationFragment], *, namespace: str | None, kind: str | None) -> tuple[AnnotationFragment, ...]:
    return tuple(fragment for fragment in fragments if (namespace is None or fragment.namespace == namespace) and (kind is None or fragment.kind == kind))


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


__all__ = ["collect_fragments", "fragments_for", "fragments_for_callable", "fragments_for_class"]
