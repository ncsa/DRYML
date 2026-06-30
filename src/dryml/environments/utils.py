"""Small shared utilities for DRYML environment metadata."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from .errors import EnvironmentRequirementError


_NAME_RUN_RE = re.compile(r"[-_.]+")


def normalize_distribution_name(name: str) -> str:
    """Normalize a Python distribution name using PEP 503-style rules."""

    return _NAME_RUN_RE.sub("-", str(name).strip().lower())


def coerce_tuple(values: Iterable[Any] | Any | None) -> tuple[Any, ...]:
    """Coerce ``None``, a scalar, or an iterable to a tuple."""

    if values is None:
        return ()
    if isinstance(values, tuple):
        return values
    if isinstance(values, str):
        return (values,)
    try:
        return tuple(values)
    except TypeError:
        return (values,)


def normalize_requirement_string(requirement: str) -> str:
    """Validate and conservatively normalize a PEP 508 requirement string."""

    text = str(requirement).strip()
    try:
        req = Requirement(text)
    except InvalidRequirement as exc:
        raise EnvironmentRequirementError(
            f"invalid package requirement {requirement!r}",
            context={"requirement": requirement, "error": str(exc)},
        ) from exc

    parts = [normalize_distribution_name(req.name)]
    if req.extras:
        parts.append("[" + ",".join(sorted(req.extras)) + "]")
    if req.url:
        parts.append(f" @ {req.url}")
    if str(req.specifier):
        parts.append(str(req.specifier))
    if req.marker:
        parts.append(f"; {req.marker}")
    return "".join(parts)


def requirement_sort_key(requirement: str) -> tuple[str, str]:
    """Return a deterministic sorting key for a requirement string."""

    req = Requirement(requirement)
    return normalize_distribution_name(req.name), requirement


def coerce_specifier(value: str | None) -> SpecifierSet | None:
    """Return a packaging ``SpecifierSet`` or raise a structured error."""

    if value in (None, ""):
        return None
    try:
        return SpecifierSet(str(value))
    except InvalidSpecifier as exc:
        raise EnvironmentRequirementError(
            f"invalid version specifier {value!r}",
            context={"specifier": value, "error": str(exc)},
        ) from exc


def merge_env(base: Mapping[str, str] | None, overrides: Mapping[str, str] | None) -> dict[str, str]:
    """Merge environment variables for probe subprocesses."""

    env = dict(base or os.environ)
    env.update({str(key): str(value) for key, value in (overrides or {}).items()})
    return env


__all__ = [
    "normalize_distribution_name",
    "normalize_requirement_string",
    "requirement_sort_key",
    "coerce_specifier",
    "coerce_tuple",
    "merge_env",
]
