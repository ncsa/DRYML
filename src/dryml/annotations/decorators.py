"""Low-level decorators that attach annotation fragments without wrapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .model import AnnotationFragment, SourceTrace, source_from_target, validate_namespace

FRAGMENT_ATTR = "__dryml_annotation_fragments__"


def require(*, namespace: str, fragment: Mapping[str, Any], source: SourceTrace | None = None, priority: int = 0, merge_policy: str | None = None):
    """Attach a hard requirement fragment to a function, method, or class.

    Example: ``@dryml.annotations.require(namespace="world", fragment={...})``.
    The target object is returned unchanged; no wrapper is introduced.
    """

    validate_namespace(namespace)
    return _decorator(namespace=namespace, kind="requirement", fragment=fragment, source=source, priority=priority, merge_policy=merge_policy)


def default(*, namespace: str, fragment: Mapping[str, Any], source: SourceTrace | None = None, priority: int = 0, merge_policy: str | None = None):
    """Attach a soft default fragment to a function, method, or class.

    Example: ``@dryml.annotations.default(namespace="runtime", fragment={...})``.
    The target object is returned unchanged; runtime activation is never entered.
    """

    validate_namespace(namespace)
    return _decorator(namespace=namespace, kind="default", fragment=fragment, source=source, priority=priority, merge_policy=merge_policy)


def attach_fragment(target: Any, fragment: AnnotationFragment) -> Any:
    """Attach *fragment* to *target* without mutating inherited fragments."""

    own = tuple(getattr(target, "__dict__", {}).get(FRAGMENT_ATTR, ()))
    setattr(target, FRAGMENT_ATTR, own + (fragment,))
    return target


def _decorator(*, namespace: str, kind: str, fragment: Mapping[str, Any], source: SourceTrace | None, priority: int, merge_policy: str | None):
    def decorate(target: Any) -> Any:
        if isinstance(source, SourceTrace):
            trace = source
        elif source is not None:
            trace = source_from_target(target, namespace=namespace, label=str(source))
        else:
            trace = source_from_target(target, namespace=namespace, label=f"@dryml.annotations.{kind}")
        annotation = AnnotationFragment(namespace, kind, fragment, trace, priority=priority, merge_policy=merge_policy)
        return attach_fragment(target, annotation)

    return decorate


__all__ = ["FRAGMENT_ATTR", "attach_fragment", "default", "require"]
