"""Small shared utilities for DRYML environment metadata."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from .errors import EnvironmentRequirementError, EnvironmentSpecError


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


def dryml_source_root() -> str:
    """Return the source root containing the importable ``dryml`` package."""

    return str(Path(__file__).resolve().parents[2])


def build_probe_env(
    *,
    base: Mapping[str, str] | None,
    overrides: Mapping[str, str] | None,
    pythonpath_policy: str,
    extra_pythonpath: tuple[str, ...] = (),
) -> dict[str, str]:
    """Build a subprocess environment while enforcing PYTHONPATH policy.

    ``none`` removes ``PYTHONPATH``. ``explicit`` uses only
    ``extra_pythonpath``. ``inherit`` preserves the base value and appends any
    explicit paths. ``dryml-source`` uses the current DRYML source root plus any
    explicit paths so a probed interpreter can import this checkout without
    inheriting unrelated orchestrator paths.
    """

    policy = str(pythonpath_policy).strip().lower().replace("_", "-")
    if policy not in {"none", "explicit", "inherit", "dryml-source"}:
        raise EnvironmentSpecError(
            f"unknown Python path probe policy {pythonpath_policy!r}",
            context={"pythonpath_policy": pythonpath_policy},
        )

    base_env = os.environ if base is None else base
    env = dict(base_env)
    env.update({str(key): str(value) for key, value in (overrides or {}).items() if str(key) != "PYTHONPATH"})
    inherited = base_env.get("PYTHONPATH")
    paths = tuple(str(path) for path in extra_pythonpath if str(path))

    if policy == "none":
        env.pop("PYTHONPATH", None)
    elif policy == "explicit":
        if paths:
            env["PYTHONPATH"] = os.pathsep.join(paths)
        else:
            env.pop("PYTHONPATH", None)
    elif policy == "inherit":
        combined = tuple(path for path in ((inherited,) if inherited else ()) + paths if path)
        if combined:
            env["PYTHONPATH"] = os.pathsep.join(combined)
        else:
            env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = os.pathsep.join((dryml_source_root(), *paths))
    return env


__all__ = [
    "normalize_distribution_name",
    "normalize_requirement_string",
    "requirement_sort_key",
    "coerce_specifier",
    "coerce_tuple",
    "merge_env",
    "build_probe_env",
    "dryml_source_root",
]
