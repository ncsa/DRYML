"""Resolve and compare detached exact environment selections for Execute.

This module owns transient selection facts only. Its values are not environment
records, selector schemas, or persisted configuration. Execute transports them
only while admitting a selected worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from types import MappingProxyType

from .compatibility import (
    CompatibilityIssue,
    CompatibilityReport,
    report_from_issues,
)
from .errors import EnvironmentSpecError
from .introspection import inspect_current
from .records import EnvironmentRecord
from .specs import (
    CondaEnvironmentSpec,
    ContainerEnvironmentSpec,
    CurrentEnvironmentSpec,
    EnvironmentSpec,
    PythonExecutableSpec,
)
from .utils import _minimal_launch_environment, build_probe_env

_CONDA_INVENTORY_TIMEOUT = 30.0
_CONDA_INVENTORY_OUTPUT_LIMIT = 1 << 20
_CONDA_INVENTORY_ENTRY_LIMIT = 4096


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedEnvironmentSelection:
    """Detached launch and identity evidence for an exact environment selector.

    Args:
        spec: Original immutable selector requested by the caller.
        command: Exact worker launch command preserving supplied spelling.
        expected_executable: Expected fresh worker executable at admission.
        expected_prefix: Expected fresh worker environment prefix at admission.
        expected_base_prefix: Expected fresh worker base prefix at admission.
        software_digest: Stable Python, DRYML, and distribution evidence.
        record: Coordinator observation used to freeze the selection.
        resolved_prefix: Conda prefix resolved from a name, when applicable.
        launch_env: Frozen selector environment and PYTHONPATH controls.
            Backend configuration may override these at worker launch.

    The value is private operation state. It neither provisions an environment
    nor authorizes a worker. :func:`compare_selection` must check fresh worker
    evidence before Execute transfers a payload. Its representation omits
    selector, environment, record, and path values.
    """

    spec: EnvironmentSpec
    command: tuple[str, ...]
    expected_executable: str | None
    expected_prefix: str | None
    expected_base_prefix: str | None
    software_digest: str
    record: EnvironmentRecord
    resolved_prefix: str | None = None
    launch_env: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __repr__(self) -> str:
        """Return a fixed representation that does not disclose launch data."""

        return "ResolvedEnvironmentSelection()"


def resolve_environment_spec(
    spec: EnvironmentSpec,
    *,
    conda_inventory: Callable[[], Iterable[str]] | None = None,
) -> ResolvedEnvironmentSelection:
    """Resolve one selector once into launch and expected-identity evidence.

    Args:
        spec: Current, direct Python, or Conda selector to pin exactly.
        conda_inventory: Optional finite prefix provider used only for a
            named Conda selector.  It must not create or activate environments.

    Returns:
        Detached command and observed identity facts for the selected target.

    Raises:
        TypeError: If ``spec`` is not an EnvironmentSpec value.
        EnvironmentSpecError: If the selector is unsupported, unavailable, or
            or a Conda name has zero or multiple existing prefix matches.

    Side Effects:
        Observes only the supplied target. Named Conda reads a finite
        inventory; no candidate discovery, fallback, package installation, or
        provisioning occurs.
    """

    if not isinstance(
        spec,
        (
            CurrentEnvironmentSpec,
            PythonExecutableSpec,
            CondaEnvironmentSpec,
            ContainerEnvironmentSpec,
        ),
    ):
        raise TypeError("environment_spec must be an EnvironmentSpec")
    if isinstance(spec, ContainerEnvironmentSpec):
        raise EnvironmentSpecError(
            "container environment selection is unsupported"
        )
    resolved_prefix: str | None = None
    selected = spec
    if isinstance(spec, CondaEnvironmentSpec) and spec.name is not None:
        # Names are converted to a prefix before the one exact launch probe.
        inventory = (
            _bounded_prefixes(conda_inventory())
            if conda_inventory is not None
            else _conda_prefixes(spec)
        )
        matches = tuple(
            prefix for prefix in inventory if Path(prefix).name == spec.name
        )
        if len(matches) != 1:
            raise EnvironmentSpecError(
                "Conda name must resolve to exactly one existing prefix"
            )
        resolved_prefix = matches[0]
        selected = CondaEnvironmentSpec(
            prefix=resolved_prefix,
            conda_executable=spec.conda_executable,
            launch_mode=spec.launch_mode,
            env=spec.env,
            pythonpath_policy=spec.pythonpath_policy,
            extra_pythonpath=spec.extra_pythonpath,
            metadata=spec.metadata,
        )
    record = _probe_record(selected)
    if isinstance(spec, CurrentEnvironmentSpec):
        command = (_absolute_path(record.python.executable),)
    elif isinstance(selected, PythonExecutableSpec):
        command = (_absolute_path(selected.executable),)
    elif isinstance(selected, CondaEnvironmentSpec):
        command = (
            (_absolute_path(selected.direct_python_executable()),)
            if selected.launch_mode == "direct"
            else (
                _absolute_executable(selected.conda_executable),
                "run",
                "-p",
                _absolute_path(selected.prefix or ""),
                "--no-capture-output",
                "--",
                "python",
            )
        )
    else:  # pragma: no cover - closed selector validation above
        raise EnvironmentSpecError("environment selector is unsupported")
    launch_env = _selector_launch_env(selected)
    return ResolvedEnvironmentSelection(
        spec=spec,
        command=tuple(command),
        expected_executable=(
            command[0]
            if len(command) == 1
            else _absolute_path(record.python.executable)
        ),
        expected_prefix=record.python.prefix,
        expected_base_prefix=record.python.base_prefix,
        software_digest=software_digest(record),
        record=record,
        resolved_prefix=resolved_prefix
        or (
            selected.prefix
            if isinstance(selected, CondaEnvironmentSpec)
            else None
        ),
        launch_env=MappingProxyType(launch_env),
    )


def compare_selection(
    selection: ResolvedEnvironmentSelection,
    observed: EnvironmentRecord | None,
) -> CompatibilityReport:
    """Compare fresh worker evidence with one resolved selection.

    Args:
        selection: Detached selection at operation entry.
        observed: Fresh worker record, or ``None`` when absent.

    Returns:
        A strict report with exact identity and software mismatches. It remains
        independent of caller-supplied EnvironmentRequirement constraints.

    Side Effects:
        None. This is point-in-time evidence; external mutation after a
        successful comparison is intentionally not monitored.
    """

    if observed is None:
        return report_from_issues(
            (
                CompatibilityIssue(
                    "selection_evidence_unavailable",
                    "error",
                    "fresh worker environment evidence is unavailable",
                ),
            ),
            policy="strict",
        )
    issues: list[CompatibilityIssue] = []
    for name, expected, actual in (
        (
            "executable",
            selection.expected_executable,
            observed.python.executable,
        ),
        ("prefix", selection.expected_prefix, observed.python.prefix),
        (
            "base_prefix",
            selection.expected_base_prefix,
            observed.python.base_prefix,
        ),
    ):
        if expected != actual:
            issues.append(
                CompatibilityIssue(
                    "selection_identity_mismatch",
                    "error",
                    f"worker {name} does not match the selected environment",
                )
            )
    if selection.software_digest != software_digest(observed):
        issues.append(
            CompatibilityIssue(
                "selection_software_mismatch",
                "error",
                "worker software evidence does not match the selected "
                "environment",
            )
        )
    return report_from_issues(tuple(issues), policy="strict")


def software_digest(record: EnvironmentRecord) -> str:
    """Return stable Python/DRYML evidence without advisory metadata.

    Args:
        record: Immutable environment observation to normalize.

    Returns:
        A SHA-256 digest over Python implementation/version and DRYML release.
        Protocol facts and normalized distributions also contribute.

    Side Effects:
        None. Timestamps, labels, package locations, and installer data do not
        contribute. Neither do details or the DRYML git revision.
    """

    dryml = record.dryml
    payload = {
        "python": {
            "implementation": record.python.implementation,
            "version": record.python.version,
        },
        "dryml": (
            None
            if dryml is None
            else {
                "version": dryml.version,
                "execution_protocol": dryml.execution_protocol,
                "schema_versions": dict(dryml.schema_versions),
                "features": list(dryml.features),
            }
        ),
        "distributions": {
            name: {
                "normalized_name": package.normalized_name,
                "version": package.version,
            }
            for name, package in record.distributions.items()
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _probe_record(spec: EnvironmentSpec) -> EnvironmentRecord:
    """Observe one exact selector without enumerating alternate candidates."""

    if isinstance(spec, CurrentEnvironmentSpec):
        return inspect_current()
    from . import _load_explicit_export
    from .probe import _probe_with_base_env

    # Importing ``probe.py`` assigns the submodule to the package attribute.
    # Restore the public callable lazy export before returning to callers.
    _load_explicit_export("probe")

    try:
        return _probe_with_base_env(
            spec, base_env=_selector_probe_base(spec)
        ).require_ok()
    except Exception:
        # Probe result context can contain child stdout/stderr and environment
        # paths; the public selection error deliberately retains none of it.
        raise EnvironmentSpecError(
            "selected environment could not be observed"
        ) from None


def _conda_prefixes(spec: CondaEnvironmentSpec) -> tuple[str, ...]:
    """Read a bounded Conda inventory without creating an environment."""

    try:
        returncode, output = _bounded_stdout(
            [
                _absolute_executable(spec.conda_executable),
                "info",
                "--envs",
                "--json",
            ],
            timeout=_CONDA_INVENTORY_TIMEOUT,
            limit=_CONDA_INVENTORY_OUTPUT_LIMIT,
            env=_selector_probe_env(spec),
        )
        payload = json.loads(output)
        prefixes = payload["envs"]
    except (OSError, ValueError, KeyError, TypeError, TimeoutError):
        raise EnvironmentSpecError(
            "Conda environment inventory is unavailable"
        ) from None
    if (
        returncode != 0
        or not isinstance(prefixes, list)
        or not all(isinstance(prefix, str) and prefix for prefix in prefixes)
    ):
        raise EnvironmentSpecError(
            "Conda environment inventory is unavailable"
        )
    return _bounded_prefixes(prefixes)


def _absolute_path(path: str) -> str:
    """Return an absolute launch spelling without resolving symlinks."""

    return os.path.abspath(path)


def _absolute_executable(path: str) -> str:
    """Return an absolute executable spelling without resolving symlinks."""

    return _absolute_path(shutil.which(path) or path)


def _selector_launch_env(spec: EnvironmentSpec) -> dict[str, str]:
    """Freeze only selector-owned launch controls at operation entry."""

    if not isinstance(spec, (PythonExecutableSpec, CondaEnvironmentSpec)):
        return {}
    prepared = build_probe_env(
        base=os.environ,
        overrides=spec.env,
        pythonpath_policy=spec.pythonpath_policy,
        extra_pythonpath=spec.extra_pythonpath,
    )
    result = dict(spec.env)
    if "PYTHONPATH" in prepared:
        result["PYTHONPATH"] = prepared["PYTHONPATH"]
    return result


def _selector_probe_base(spec: EnvironmentSpec) -> dict[str, str]:
    """Return the exact selector's minimal base plus approved inheritance."""

    environment = _minimal_launch_environment()
    if (
        isinstance(spec, (PythonExecutableSpec, CondaEnvironmentSpec))
        and str(spec.pythonpath_policy).strip().lower().replace("_", "-")
        == "inherit"
        and (pythonpath := os.environ.get("PYTHONPATH"))
    ):
        environment["PYTHONPATH"] = pythonpath
    return environment


def _selector_probe_env(spec: CondaEnvironmentSpec) -> dict[str, str]:
    """Build the minimal inventory environment under selector path policy."""

    return build_probe_env(
        base=_selector_probe_base(spec),
        overrides=spec.env,
        pythonpath_policy=spec.pythonpath_policy,
        extra_pythonpath=spec.extra_pythonpath,
    )


def _bounded_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    """Copy one Conda inventory under fixed count and elapsed-time ceilings."""

    deadline = time.monotonic() + _CONDA_INVENTORY_TIMEOUT
    result: list[str] = []
    iterator = iter(prefixes)
    for _ in range(_CONDA_INVENTORY_ENTRY_LIMIT + 1):
        if time.monotonic() >= deadline:
            raise EnvironmentSpecError(
                "Conda environment inventory is unavailable"
            )
        try:
            prefix = next(iterator)
        except StopIteration:
            return tuple(result)
        if not isinstance(prefix, str) or not prefix:
            raise EnvironmentSpecError(
                "Conda environment inventory is unavailable"
            )
        result.append(prefix)
    raise EnvironmentSpecError("Conda environment inventory is unavailable")


def _bounded_stdout(
    command: list[str], *, timeout: float, limit: int, env: Mapping[str, str]
) -> tuple[int, str]:
    """Run an inventory command while retaining only bounded output."""

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    assert process.stdout is not None
    retained = bytearray()
    overflow = False

    def drain() -> None:
        """Drain output so a verbose inventory cannot block its process."""
        nonlocal overflow
        while chunk := process.stdout.read(8192):
            remaining = limit + 1 - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow = True

    reader = Thread(target=drain, name="dryml-conda-inventory", daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise TimeoutError from None
    finally:
        reader.join(timeout=timeout)
        process.stdout.close()
    if reader.is_alive() or overflow or len(retained) > limit:
        raise ValueError(
            "Conda environment inventory exceeds the output limit"
        )
    return returncode, retained.decode("utf-8")


__all__ = [
    "ResolvedEnvironmentSelection",
    "compare_selection",
    "resolve_environment_spec",
    "software_digest",
]
