"""Identity-preserving requirement and default declaration decorators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, TypeVar

from .model import AnnotationFragment, SourceTrace, source_from_target, target_from_live, validate_namespace
from .storage import attach_fragment

T = TypeVar("T")


def require(*, namespace: str, fragment: Mapping[str, Any], source: SourceTrace | None = None, priority: int = 0, merge_policy: str | None = None) -> Callable[[T], T]:
    """Return an identity-preserving hard-requirement decorator.

    Args:
        namespace: Closed declaration namespace.
        fragment: Bounded JSON or typed-family envelope payload.
        source: Optional identifying source trace.
        priority: Merge priority; higher values apply later.
        merge_policy: Optional closed merge policy.

    Returns:
        A decorator that stores direct metadata and returns its exact target.
    """

    validate_namespace(namespace)
    return _decorator(namespace, "requirement", fragment, source, priority, merge_policy)


def default(*, namespace: str, fragment: Mapping[str, Any], source: SourceTrace | None = None, priority: int = 0, merge_policy: str | None = None) -> Callable[[T], T]:
    """Return an identity-preserving overrideable-default decorator.

    Args:
        namespace: Closed declaration namespace.
        fragment: Bounded JSON or typed-family envelope payload.
        source: Optional identifying source trace.
        priority: Merge priority; higher values apply later.
        merge_policy: Optional closed merge policy.

    Returns:
        A decorator that stores direct metadata and returns its exact target.
    """

    validate_namespace(namespace)
    return _decorator(namespace, "default", fragment, source, priority, merge_policy)


def _decorator(namespace: str, kind: str, fragment: Mapping[str, Any], source: SourceTrace | None, priority: int, merge_policy: str | None) -> Callable[[T], T]:
    def decorate(target: T) -> T:
        trace = source if source is not None else source_from_target(target, namespace=namespace, label=f"@dryml.annotations.{kind}")
        annotation = AnnotationFragment(target_from_live(target), namespace, kind, fragment, trace, priority, merge_policy)
        return attach_fragment(target, annotation)
    return decorate


__all__ = ["default", "require"]
