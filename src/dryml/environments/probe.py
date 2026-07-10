"""Probe current, Python-executable, and Conda environments."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Any

from .compatibility import CompatibilityIssue, CompatibilityReport, report_from_issues
from .errors import EnvironmentProbeError, EnvironmentSpecError
from .introspection import inspect_current
from .records import EnvironmentRecord
from .schema import ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION
from .specs import (
    CondaEnvironmentSpec,
    ContainerEnvironmentSpec,
    CurrentEnvironmentSpec,
    EnvironmentSpec,
    ProbeWorkerCommand,
    PythonExecutableSpec,
    spec_from_data,
)
from .utils import build_probe_env

_MAX_CAPTURED_OUTPUT_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class EnvironmentProbeResult:
    """Result of probing an environment spec."""

    spec: EnvironmentSpec
    ok: bool
    record: EnvironmentRecord | None = None
    report: CompatibilityReport | None = None
    stdout: str | None = None
    stderr: str | None = None
    returncode: int | None = None
    schema_version: int = ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION

    def require_ok(self) -> EnvironmentRecord:
        """Return the environment record or raise a structured probe error."""

        if self.ok and self.record is not None:
            return self.record
        message = self.report.explain() if self.report is not None else "environment probe failed"
        raise EnvironmentProbeError(
            message,
            context={"stdout": self.stdout, "stderr": self.stderr, "returncode": self.returncode},
        )

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible probe-result data."""

        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_data(),
            "ok": self.ok,
            "record": None if self.record is None else self.record.to_data(),
            "report": None if self.report is None else self.report.to_data(),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "EnvironmentProbeResult":
        """Build a probe result from serialized data."""

        if not isinstance(data, Mapping):
            raise EnvironmentProbeError("environment probe result must be a mapping")
        if data.get("schema_version", ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION) != ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION:
            raise EnvironmentProbeError("unsupported environment probe result schema")
        if not isinstance(data.get("ok"), bool):
            raise EnvironmentProbeError("environment probe result ok must be a boolean")
        if data["ok"] and not isinstance(data.get("record"), Mapping):
            raise EnvironmentProbeError("successful environment probe result requires a record")
        try:
            return cls(
                spec=spec_from_data(data["spec"]),
                ok=bool(data["ok"]),
                record=(
                    None
                    if data.get("record") is None
                    else EnvironmentRecord.from_data(data["record"])
                ),
                report=(
                    None
                    if data.get("report") is None
                    else CompatibilityReport.from_data(data["report"])
                ),
                stdout=data.get("stdout"),
                stderr=data.get("stderr"),
                returncode=data.get("returncode"),
                schema_version=data.get("schema_version", ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION),
            )
        except Exception as exc:
            raise EnvironmentProbeError(f"environment probe result could not be decoded: {type(exc).__name__}") from exc


def _failure_result(
    spec: EnvironmentSpec,
    code: str,
    message: str,
    *,
    stdout: str | None = None,
    stderr: str | None = None,
    returncode: int | None = None,
) -> EnvironmentProbeResult:
    issue = CompatibilityIssue(code, "error", message)
    return EnvironmentProbeResult(
        spec=spec,
        ok=False,
        report=report_from_issues((issue,)),
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )


def probe(spec: EnvironmentSpec | None = None, *, timeout: float | None = 30.0) -> EnvironmentProbeResult:
    """Probe an environment spec and return a structured result."""

    spec = spec or CurrentEnvironmentSpec()
    if isinstance(spec, CurrentEnvironmentSpec):
        if timeout is None:
            return EnvironmentProbeResult(spec=spec, ok=True, record=inspect_current())
        return _probe_command(spec, [sys.executable, *ProbeWorkerCommand], timeout=timeout)
    if isinstance(spec, PythonExecutableSpec):
        return _probe_command(
            spec,
            spec.probe_command(),
            timeout=timeout,
            env=build_probe_env(
                base=None,
                overrides=spec.env,
                pythonpath_policy=spec.pythonpath_policy,
                extra_pythonpath=spec.extra_pythonpath,
            ),
        )
    if isinstance(spec, CondaEnvironmentSpec):
        try:
            cmd = spec.probe_command()
        except EnvironmentSpecError as exc:
            return _failure_result(spec, "unsupported_environment_spec", str(exc))
        return _probe_command(
            spec,
            cmd,
            timeout=timeout,
            env=build_probe_env(
                base=None,
                overrides=spec.env,
                pythonpath_policy=spec.pythonpath_policy,
                extra_pythonpath=spec.extra_pythonpath,
            ),
        )
    if isinstance(spec, ContainerEnvironmentSpec):
        return _failure_result(spec, "unsupported_environment_spec", "container probing is not implemented")
    return _failure_result(spec, "unsupported_environment_spec", f"unsupported environment spec {type(spec).__name__}")


def _probe_command(
    spec: EnvironmentSpec,
    command: list[str],
    *,
    timeout: float | None,
    env: dict[str, str] | None = None,
) -> EnvironmentProbeResult:
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            completed = subprocess.run(command, stdout=stdout_file, stderr=stderr_file, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            stdout, stdout_truncated = _read_captured_output(stdout_file)
            stderr, stderr_truncated = _read_captured_output(stderr_file)
            return _capture_diagnostic(
                _failure_result(spec, "probe_timeout", f"environment probe timed out after {timeout} seconds", stdout=stdout, stderr=stderr),
                stdout_truncated or stderr_truncated,
            )
        except OSError as exc:
            return _failure_result(spec, "probe_failed", f"environment probe could not start: {exc}")
        stdout, stdout_truncated = _read_captured_output(stdout_file)
        stderr, stderr_truncated = _read_captured_output(stderr_file)
    truncated = stdout_truncated or stderr_truncated
    if completed.returncode != 0:
        return _capture_diagnostic(_failure_result(spec, "probe_failed", f"environment probe exited with status {completed.returncode}", stdout=stdout, stderr=stderr, returncode=completed.returncode), truncated)
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return _capture_diagnostic(_failure_result(spec, "probe_failed", f"environment probe returned malformed JSON: {exc}", stdout=stdout, stderr=stderr, returncode=completed.returncode), truncated)
    if not isinstance(payload, Mapping):
        return _capture_diagnostic(_failure_result(spec, "probe_failed", "environment probe returned a non-mapping protocol payload", stdout=stdout, stderr=stderr, returncode=completed.returncode), truncated)
    if payload.get("kind") != "dryml.environment_probe_result":
        return _capture_diagnostic(_failure_result(spec, "probe_failed", "environment probe returned an unsupported protocol payload", stdout=stdout, stderr=stderr, returncode=completed.returncode), truncated)
    if payload.get("schema_version") != ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION:
        return _capture_diagnostic(_failure_result(spec, "probe_failed", "environment probe returned an unsupported protocol schema", stdout=stdout, stderr=stderr, returncode=completed.returncode), truncated)
    if not isinstance(payload.get("ok"), bool):
        return _capture_diagnostic(_failure_result(spec, "probe_failed", "environment probe returned an invalid protocol status", stdout=stdout, stderr=stderr, returncode=completed.returncode), truncated)
    if not payload["ok"]:
        return _capture_diagnostic(_failure_result(spec, "probe_failed", "environment probe worker reported failure", stdout=stdout, stderr=stderr, returncode=completed.returncode), truncated)
    try:
        record = EnvironmentRecord.from_data(payload["record"])
    except Exception as exc:
        return _capture_diagnostic(_failure_result(spec, "probe_failed", f"environment probe record could not be decoded: {exc}", stdout=stdout, stderr=stderr, returncode=completed.returncode), truncated)
    return _capture_diagnostic(EnvironmentProbeResult(spec=spec, ok=True, record=record, stdout=stdout, stderr=stderr, returncode=completed.returncode), truncated)


def _read_captured_output(handle) -> tuple[str, bool]:
    """Decode at most the bounded diagnostic prefix from a probe stream."""

    handle.seek(0)
    value = handle.read(_MAX_CAPTURED_OUTPUT_BYTES + 1)
    return value[:_MAX_CAPTURED_OUTPUT_BYTES].decode("utf-8", errors="replace"), len(value) > _MAX_CAPTURED_OUTPUT_BYTES


def _capture_diagnostic(result: EnvironmentProbeResult, truncated: bool) -> EnvironmentProbeResult:
    """Mark captured probe output truncation without retaining excess bytes."""

    if not truncated:
        return result
    issue = CompatibilityIssue("probe_output_truncated", "warning", "environment probe output was truncated")
    issues = () if result.report is None else result.report.issues
    return replace(result, report=report_from_issues((*issues, issue)))


def probe_current() -> EnvironmentProbeResult:
    """Probe the current process environment."""

    return probe(CurrentEnvironmentSpec())


def probe_python(executable: str, *, timeout: float | None = 30.0) -> EnvironmentProbeResult:
    """Probe an environment through a Python executable path."""

    return probe(PythonExecutableSpec(executable=executable), timeout=timeout)


def probe_conda(
    *,
    prefix: str | None = None,
    name: str | None = None,
    launch_mode: str = "direct",
    timeout: float | None = 30.0,
) -> EnvironmentProbeResult:
    """Probe a Conda environment by prefix or name."""

    return probe(
        CondaEnvironmentSpec(prefix=prefix, name=name, launch_mode=launch_mode),
        timeout=timeout,
    )


__all__ = [
    "EnvironmentProbeResult",
    "probe",
    "probe_current",
    "probe_python",
    "probe_conda",
]
