"""Lightweight current-process environment inspection."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import sys
from pathlib import Path

from .records import (
    DrymlRuntimeRecord,
    EnvironmentRecord,
    PackageRecord,
    PlatformRecord,
    PythonRecord,
)
from .schema import (
    COMPATIBILITY_REPORT_SCHEMA_VERSION,
    ENVIRONMENT_FRAGMENT_SCHEMA_VERSION,
    ENVIRONMENT_LOCK_REF_SCHEMA_VERSION,
    ENVIRONMENT_RECORD_SCHEMA_VERSION,
    ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION,
    ENVIRONMENT_SPEC_SCHEMA_VERSION,
)
from .utils import normalize_distribution_name


def _distribution_location(dist: metadata.Distribution) -> str | None:
    try:
        if dist.files:
            first = next(iter(dist.files), None)
            if first is not None:
                located = dist.locate_file(first)
                return str(Path(located).parent)
    except (OSError, TypeError, ValueError):
        return None
    return None


def _distribution_installer(dist: metadata.Distribution) -> str | None:
    try:
        text = dist.read_text("INSTALLER")
    except (OSError, TypeError, ValueError):
        return None
    return text.strip() if text else None


def _environment_kind() -> str:
    if os.environ.get("CONDA_PREFIX"):
        return "conda"
    if os.environ.get("VIRTUAL_ENV") or sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return "venv"
    return "system"


def _dryml_version() -> str | None:
    try:
        return metadata.version("dryml")
    except metadata.PackageNotFoundError:
        return None


def inspect_current() -> EnvironmentRecord:
    """Inspect the current Python environment without importing package runtimes.

    Installed distributions are read through :mod:`importlib.metadata`; heavy
    modules such as TensorFlow, Torch, JAX, Ray, or Slurm integrations are never
    imported by this function.
    """

    distributions: dict[str, PackageRecord] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") or getattr(dist, "name", None) or "unknown"
        normalized = normalize_distribution_name(name)
        distributions[normalized] = PackageRecord(
            name=name,
            normalized_name=normalized,
            version=getattr(dist, "version", None),
            metadata_name=dist.metadata.get("Name"),
            location=_distribution_location(dist),
            installer=_distribution_installer(dist),
        )

    details = {}
    for env_name in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"):
        value = os.environ.get(env_name)
        if value:
            details[env_name.lower()] = value

    dryml = DrymlRuntimeRecord(
        version=_dryml_version(),
        schema_versions={
            "environment_record": ENVIRONMENT_RECORD_SCHEMA_VERSION,
            "environment_requirement": ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION,
            "environment_spec": ENVIRONMENT_SPEC_SCHEMA_VERSION,
            "environment_lock_ref": ENVIRONMENT_LOCK_REF_SCHEMA_VERSION,
            "compatibility_report": COMPATIBILITY_REPORT_SCHEMA_VERSION,
            "environment_fragment": ENVIRONMENT_FRAGMENT_SCHEMA_VERSION,
        },
        features=("dryml.environments.v1",),
    )

    return EnvironmentRecord(
        python=PythonRecord(
            version=platform.python_version(),
            implementation=platform.python_implementation(),
            executable=sys.executable,
            prefix=sys.prefix,
            base_prefix=getattr(sys, "base_prefix", sys.prefix),
        ),
        platform=PlatformRecord(
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
            platform=platform.platform(),
        ),
        distributions=distributions,
        dryml=dryml,
        kind=_environment_kind(),
        details=details,
    )


__all__ = ["inspect_current"]
