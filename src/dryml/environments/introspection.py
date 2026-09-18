"""Lightweight current-process environment inspection."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import site
import sys
import sysconfig
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


def _distribution_paths() -> tuple[str, ...]:
    """Select installation roots, retaining active site-path precedence."""

    roots = [sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    roots.extend(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        roots.append(site.getusersitepackages())
    roots = list(dict.fromkeys(os.path.abspath(path) for path in roots))
    registered = {os.path.normcase(path) for path in roots}
    ordered = [
        os.path.abspath(path) for path in sys.path
        if isinstance(path, str)
        and os.path.normcase(os.path.abspath(path)) in registered
    ]
    return tuple(dict.fromkeys((*ordered, *roots)))


def inspect_current() -> EnvironmentRecord:
    """Inspect the current Python environment without importing package runtimes.

    Installed distributions are read through :mod:`importlib.metadata` from
    interpreter installation roots, including enabled user/system site roots.
    Transient application and vendored search paths do not change this
    inventory. Heavy modules such as TensorFlow, Torch, JAX, Ray, or Slurm
    integrations are never imported by this function.

    Returns:
        An immutable environment record with interpreter, platform, installed
        package, and DRYML capability evidence.

    Raises:
        EnvironmentSerializationError: If observed metadata cannot be encoded
            within the environment record's value and size limits.

    Side Effects:
        Reads local installed-distribution metadata and interpreter state.
        It does not import inspected package runtimes or modify the
        environment.
    """

    distributions: dict[str, PackageRecord] = {}
    for dist in metadata.distributions(path=_distribution_paths()):
        name = dist.metadata.get("Name") or getattr(dist, "name", None) or "unknown"
        normalized = normalize_distribution_name(name)
        if normalized in distributions:
            continue
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
        },
        features=("dryml.environments.v1.1",),
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
            os_name=os.name,
            sys_platform=sys.platform,
            implementation_name=getattr(sys.implementation, "name", platform.python_implementation().lower()),
            implementation_version=platform.python_version(),
            platform_python_implementation=platform.python_implementation(),
        ),
        distributions=distributions,
        dryml=dryml,
        kind=_environment_kind(),
        details=details,
    )


__all__ = ["inspect_current"]
