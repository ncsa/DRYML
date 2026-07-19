"""Shared dependency-light helpers for framework cache views."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheViewIssue:
    """One structured backend-view support issue."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CacheViewSupport:
    """Dependency support result that does not import a framework."""

    status: str
    issues: tuple[CacheViewIssue, ...] = ()


class CacheViewUnavailableError(RuntimeError):
    """Raised when iterating a view whose optional dependency is unavailable."""

    def __init__(self, support: CacheViewSupport):
        self.support = support
        message = support.issues[0].message if support.issues else "cache view is unavailable"
        super().__init__(message)


def framework_support(
    name: str,
    label: str,
    extra: str,
    *,
    available: bool | None = None,
) -> CacheViewSupport:
    """Report optional framework availability without importing it."""

    if available is None:
        available = importlib.util.find_spec(name) is not None
    if not available:
        return CacheViewSupport(
            "unsupported",
            (
                CacheViewIssue(
                    "optional_dependency_missing",
                    f"{label} cache views require the dryml[{extra}] extra",
                ),
            ),
        )
    return CacheViewSupport("ok")


def iter_cache_representation(root, kind):
    """Open a supported cache representation lazily."""

    if kind == "dryml.numpy_sequence":
        from dryml.artifacts.representations import iter_numpy_sequence

        return iter_numpy_sequence(root)
    if kind == "dryml.parquet":
        from dryml.artifacts.representations import iter_parquet_sequence

        return iter_parquet_sequence(root)
    raise RuntimeError(f"cache view does not support representation {kind!r}")


__all__ = [
    "CacheViewIssue",
    "CacheViewSupport",
    "CacheViewUnavailableError",
    "framework_support",
    "iter_cache_representation",
]
