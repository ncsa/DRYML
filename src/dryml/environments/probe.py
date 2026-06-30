"""Probe current, Python-executable, and Conda environments."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
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
    PythonExecutableSpec,
    spec_from_data,
)
from .utils import build_probe_env


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
        return EnvironmentProbeResult(spec=spec, ok=True, record=inspect_current())
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
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return _failure_result(
            spec,
            "probe_timeout",
            f"environment probe timed out after {timeout} seconds",
            stdout=exc.stdout if isinstance(exc.stdout, str) else None,
            stderr=exc.stderr if isinstance(exc.stderr, str) else None,
        )
    except OSError as exc:
        return _failure_result(spec, "probe_failed", f"environment probe could not start: {exc}")

    if completed.returncode != 0:
        return _failure_result(
            spec,
            "probe_failed",
            f"environment probe exited with status {completed.returncode}",
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return _failure_result(
            spec,
            "probe_failed",
            f"environment probe returned malformed JSON: {exc}",
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
    if not payload.get("ok"):
        issue = payload.get("error") or "environment probe worker reported failure"
        return _failure_result(
            spec,
            "probe_failed",
            str(issue),
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
    try:
        record = EnvironmentRecord.from_data(payload["record"])
    except (KeyError, TypeError, ValueError) as exc:
        return _failure_result(
            spec,
            "probe_failed",
            f"environment probe record could not be decoded: {exc}",
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
    return EnvironmentProbeResult(
        spec=spec,
        ok=True,
        record=record,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


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
