"""Bounded, non-reserving environment candidate discovery for Execute."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from dryml.environments import ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION, EnvironmentRecord
from dryml.environments.specs import CondaEnvironmentSpec, CurrentEnvironmentSpec, EnvironmentSpec, PythonExecutableSpec
from dryml.environments.utils import build_probe_env
from dryml.formats import semantic_id

from ._process import minimal_environment, run_bounded
from .config import BackendConfig
from .models import EnvironmentCandidate, ExecutionIssue

_EXCLUDED_DIRECTORIES = {".git", ".hg", ".svn", "node_modules", "__pycache__"}
_PROBE_KIND = "dryml.environment_probe_result"


@dataclass(frozen=True, slots=True)
class CandidateInventory:
    """Hold ordered, deduplicated selectors, bounded issues, and completion state."""

    specs: tuple[EnvironmentSpec, ...]
    issues: tuple[ExecutionIssue, ...]
    complete: bool


def identity(spec: EnvironmentSpec, *, interpreter: Path | None = None) -> tuple[str, str]:
    """Return an opaque complete launch identity for a selector.

    Current-environment selectors include the captured interpreter. Other selector
    semantic IDs already cover their executable, Conda mode, environment, and
    Python-path declarations. The hash prevents diagnostic keys from exposing
    private local paths or environment values.
    """
    payload = {"spec": spec.semantic_id}
    if isinstance(spec, CurrentEnvironmentSpec):
        payload["interpreter"] = "" if interpreter is None else str(interpreter)
    return (type(spec).__name__, semantic_id("execcandidate", "dryml.execute.candidate.v1", "launch", payload))


def identity_text(spec: EnvironmentSpec, *, interpreter: Path | None = None) -> str:
    """Return the opaque snapshot key for one complete launch identity."""
    return identity(spec, interpreter=interpreter)[1]


def discover_candidates(
    config: BackendConfig,
    *,
    cwd: Path,
    interpreter: Path,
    conda_inventory: Callable[[], Iterable[CondaEnvironmentSpec]] | None = None,
    deadline: float | None = None,
    cancelled: Event | None = None,
) -> CandidateInventory:
    """Enumerate allowed selectors under one total discovery deadline.

    Args:
        config: Immutable Execute discovery and subprocess policy.
        cwd: Coordinator-captured working directory used only for conventional
            project venv locations.
        interpreter: Coordinator-captured interpreter for the current selector.
        conda_inventory: Optional test/backend inventory provider; it is consumed
            lazily and never creates environments.
        deadline: Optional absolute monotonic deadline. It may shorten, but never
            lengthens, ``config.discovery_timeout``.
        cancelled: Optional cancellation signal checked between every bounded unit.

    Returns:
        Ordered selectors, bounded safe diagnostics, and false ``complete`` when
        a deadline, cancellation, candidate cap, or issue cap stopped discovery.

    Side Effects:
        Reads configured roots and may invoke Conda's JSON inventory command. It
        neither scans home directories nor installs, activates, or imports a
        workload environment.
    """
    now = time.monotonic()
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise TypeError("deadline must be a finite monotonic timestamp")
    stop_at = min(now + config.discovery_timeout, float(deadline) if deadline is not None else math.inf)
    candidates: list[EnvironmentSpec] = []
    issues: list[ExecutionIssue] = []
    seen: set[tuple[str, str]] = set()
    examined = 0
    complete = True

    def issue(code: str, message: str) -> None:
        nonlocal complete
        if len(issues) < config.diagnostic_issue_limit:
            issues.append(ExecutionIssue(code, message))
        else:
            complete = False

    def stopped() -> bool:
        nonlocal complete
        if cancelled is not None and cancelled.is_set():
            issue("discovery_cancelled", "environment discovery was cancelled")
            complete = False
            return True
        if time.monotonic() >= stop_at:
            issue("discovery_timeout", "environment discovery exceeded its total deadline")
            complete = False
            return True
        return False

    def add(spec: EnvironmentSpec) -> bool:
        nonlocal examined, complete
        if stopped():
            return False
        examined += 1
        if examined > config.discovery_candidate_limit:
            issue("discovery_candidate_limit", "environment candidate examination limit reached")
            complete = False
            return False
        key = identity(spec, interpreter=interpreter)
        if key not in seen:
            seen.add(key)
            candidates.append(spec)
        return True

    if config.automatic_environment_discovery and not add(CurrentEnvironmentSpec()):
        return CandidateInventory(tuple(candidates), tuple(issues), complete)
    for spec in config.environment_candidates:
        if not add(spec):
            return CandidateInventory(tuple(candidates), tuple(issues), complete)
    if config.automatic_environment_discovery:
        inventory = conda_inventory() if conda_inventory is not None else _conda_specs(config, stop_at, issue)
        try:
            for spec in inventory:
                if not isinstance(spec, CondaEnvironmentSpec):
                    examined += 1
                    issue("conda_inventory_malformed", "Conda inventory contained a non-Conda selector")
                    if examined >= config.discovery_candidate_limit:
                        issue("discovery_candidate_limit", "environment candidate examination limit reached")
                        complete = False
                        return CandidateInventory(tuple(candidates), tuple(issues), complete)
                    continue
                if not add(spec):
                    return CandidateInventory(tuple(candidates), tuple(issues), complete)
        except Exception:
            issue("conda_inventory_unavailable", "Conda inventory could not be read")
        for name in (".venv", "venv"):
            if stopped():
                return CandidateInventory(tuple(candidates), tuple(issues), complete)
            executable = _venv_python(cwd / name)
            if executable is not None and not add(PythonExecutableSpec(executable=str(executable))):
                return CandidateInventory(tuple(candidates), tuple(issues), complete)
    for configured_root in config.environment_search_roots:
        root = configured_root if configured_root.is_absolute() else cwd / configured_root
        for executable in _root_pythons(root, config.environment_search_depth, config.discovery_directory_entry_limit, stop_at, issue):
            if not add(PythonExecutableSpec(executable=str(executable))):
                return CandidateInventory(tuple(candidates), tuple(issues), complete)
    if any(item.code in {"conda_inventory_incomplete", "discovery_directory_limit"} for item in issues):
        complete = False
    if stopped():
        complete = False
    return CandidateInventory(tuple(candidates), tuple(issues), complete)


def probe_candidate(
    spec: EnvironmentSpec,
    *,
    interpreter: Path,
    timeout: float,
    output_limit: int,
    deadline: float | None = None,
    cancelled: Event | None = None,
    termination_timeout: float = 5.0,
    read_chunk_bytes: int = 8192,
    poll_interval: float = 0.005,
) -> EnvironmentCandidate:
    """Probe one selector through bounded owner-process controls.

    The environment owner defines probe records and launch environment semantics.
    Execute validates the exact owner envelope and separately marks a candidate
    unknown until a future backend proves its Python/dill/Execute eligibility.
    """
    key = identity_text(spec, interpreter=interpreter)
    try:
        command = [str(interpreter), "-m", "dryml.environments.probe_worker", "--json"] if isinstance(spec, CurrentEnvironmentSpec) else spec.probe_command() if isinstance(spec, (PythonExecutableSpec, CondaEnvironmentSpec)) else None
        env = _probe_environment(spec)
    except Exception:
        return EnvironmentCandidate(key, spec, None, None, False, (ExecutionIssue("candidate_unlaunchable", "environment selector has no usable probe command"),))
    if command is None:
        return EnvironmentCandidate(key, spec, None, None, None, (ExecutionIssue("candidate_unsupported", "environment selector is not launchable by Execute"),))
    try:
        result = run_bounded(command, timeout=timeout, deadline=deadline, output_limit=output_limit, cancelled=cancelled, env=env, termination_timeout=termination_timeout, read_chunk_bytes=read_chunk_bytes, poll_interval=poll_interval)
    except OSError:
        return EnvironmentCandidate(key, spec, None, None, False, (ExecutionIssue("candidate_unlaunchable", "environment probe process could not start"),))
    if result.cancelled:
        return EnvironmentCandidate(key, spec, None, None, None, (ExecutionIssue("probe_cancelled", "environment probe was cancelled"),))
    if result.timed_out:
        return EnvironmentCandidate(key, spec, None, None, None, (ExecutionIssue("probe_timeout", "environment probe exceeded its bounded deadline"),))
    if not result.cleanup_complete:
        return EnvironmentCandidate(key, spec, None, None, None, (ExecutionIssue("probe_cleanup_incomplete", "environment probe cleanup could not be confirmed"),))
    if not result.stdout_complete or not result.stderr_complete or result.stdout_truncated or result.stderr_truncated:
        return EnvironmentCandidate(key, spec, None, None, None, (ExecutionIssue("probe_output_incomplete", "environment probe output was incomplete"),))
    if result.returncode != 0:
        return EnvironmentCandidate(key, spec, None, None, False, (ExecutionIssue("candidate_unlaunchable", "environment probe process failed"),))
    try:
        payload = json.loads(result.stdout, object_pairs_hook=_unique_object)
        if not isinstance(payload, Mapping) or payload.get("kind") != _PROBE_KIND or payload.get("schema_version") != ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION or payload.get("ok") is not True or set(payload) != {"kind", "schema_version", "ok", "record"}:
            raise ValueError
        record = EnvironmentRecord.from_data(payload["record"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return EnvironmentCandidate(key, spec, None, None, None, (ExecutionIssue("probe_malformed", "environment probe returned malformed owner evidence"),))
    eligibility = _execute_eligibility(record)
    if eligibility is not None:
        return EnvironmentCandidate(key, spec, record, None, None, (eligibility,))
    return EnvironmentCandidate(key, spec, record, None, True, ())


def _probe_environment(spec: EnvironmentSpec) -> dict[str, str]:
    """Apply the selector owner's environment and Python-path policy to a probe."""
    if isinstance(spec, (PythonExecutableSpec, CondaEnvironmentSpec)):
        return build_probe_env(base=minimal_environment(), overrides=spec.env, pythonpath_policy=spec.pythonpath_policy, extra_pythonpath=spec.extra_pythonpath)
    return minimal_environment()


def _execute_eligibility(record: EnvironmentRecord) -> ExecutionIssue | None:
    """Keep environment inspection distinct from future backend launch proof."""
    if record.python.implementation != "CPython" or not record.python.version:
        return ExecutionIssue("candidate_python_unproven", "candidate Python identity is not supported")
    if "dill" not in record.distributions:
        return ExecutionIssue("candidate_dill_unproven", "candidate lacks required serializer evidence")
    features = () if record.dryml is None else record.dryml.features
    if "dryml.execute.v0.3" not in features:
        return ExecutionIssue("candidate_execute_unproven", "candidate lacks Execute bootstrap feature evidence")
    return None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Decode one JSON object while rejecting duplicate protocol fields."""
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _venv_python(prefix: Path) -> Path | None:
    """Return an existing conventional venv executable without resolving links."""
    executable = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return executable if executable.is_file() else None


def _conda_specs(config: BackendConfig, deadline: float, issue: Callable[[str, str], None]) -> Iterable[CondaEnvironmentSpec]:
    """Read Conda JSON inventory within the caller's remaining total deadline."""
    try:
        result = run_bounded([config.conda_executable, "info", "--envs", "--json"], deadline=deadline, output_limit=config.owner_envelope_limit_bytes, termination_timeout=config.termination_timeout, read_chunk_bytes=config.process_read_chunk_bytes, poll_interval=config.process_poll_interval)
    except OSError:
        issue("conda_inventory_unavailable", "Conda inventory command could not be started")
        return ()
    if result.timed_out:
        issue("conda_inventory_timeout", "Conda inventory exceeded its bounded deadline")
        return ()
    if not result.cleanup_complete or not result.stdout_complete or not result.stderr_complete or result.stdout_truncated or result.stderr_truncated:
        issue("conda_inventory_incomplete", "Conda inventory output was incomplete")
        return ()
    if result.returncode != 0:
        issue("conda_inventory_unavailable", "Conda inventory command could not be run")
        return ()
    try:
        payload = json.loads(result.stdout, object_pairs_hook=_unique_object)
        prefixes = payload["envs"]
        if not isinstance(prefixes, list) or not all(isinstance(prefix, str) and prefix for prefix in prefixes):
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError, KeyError):
        issue("conda_inventory_malformed", "Conda inventory returned malformed JSON")
        return ()
    return (CondaEnvironmentSpec(prefix=prefix, conda_executable=config.conda_executable, launch_mode=config.conda_launch_mode) for prefix in prefixes)


def _root_pythons(root: Path, depth: int, entry_limit: int, deadline: float, issue: Callable[[str, str], None]) -> Iterable[Path]:
    """Yield executable files under a root with bounded per-directory work."""
    if not root.is_dir():
        issue("discovery_root_unavailable", "configured environment root is inaccessible or not a directory")
        return ()
    stack = [(root, 0)]
    while stack:
        if time.monotonic() >= deadline:
            issue("discovery_timeout", "environment discovery exceeded its total deadline")
            return ()
        directory, level = stack.pop()
        try:
            with os.scandir(directory) as entries:
                batch = []
                for entry in entries:
                    if len(batch) >= entry_limit:
                        issue("discovery_directory_limit", "configured environment directory entry limit reached")
                        break
                    batch.append(entry)
        except OSError:
            issue("discovery_path_error", "configured environment root could not be read")
            continue
        for entry in sorted(batch, key=lambda item: item.name, reverse=True):
            if entry.name in _EXCLUDED_DIRECTORIES:
                continue
            path = Path(entry.path)
            try:
                if entry.is_file() and entry.name in {"python", "python.exe"}:
                    yield path
                elif level < depth and entry.is_dir():
                    stack.append((path, level + 1))
            except OSError:
                issue("discovery_path_error", "configured environment path could not be read")


__all__ = ["CandidateInventory", "discover_candidates", "identity", "identity_text", "probe_candidate"]
