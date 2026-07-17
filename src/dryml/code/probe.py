"""Lightweight code probe request, result, and execution helpers.

Code probes run the existing :mod:`dryml.code` analyzers under
``RuntimeMode.PROBE``. They may import a target module to resolve an import path;
that import can execute module-level Python code, so probe execution captures
stdout/stderr and never intentionally executes target function bodies, dynamic
tracing, workload allocation, package solving, or world synthesis.
"""

from __future__ import annotations

import io
import json
import math
import subprocess
import sys
from collections.abc import Iterable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from dryml import environments, runtime
from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, analyze, available_analyzers
from dryml.code.facts import DiagnosticFact, json_compatible
from dryml.code.targets import CodeTarget, CodeTargetSpec, normalize_target
from dryml.environments.errors import EnvironmentSpecError
from dryml.environments.records import EnvironmentRecord
from dryml.environments.specs import (
    CondaEnvironmentSpec,
    ContainerEnvironmentSpec,
    CurrentEnvironmentSpec,
    EnvironmentSpec,
    PythonExecutableSpec,
)
from dryml.environments.probe import _diagnostic_output, _run_bounded_command
from dryml.environments.utils import build_probe_env


PROBE_SCHEMA_VERSION = 1
DEFAULT_PROBE_ALGORITHMS = ("callables", "source", "symbol_capture", "direct_annotations")
PROBE_WORKER_ARGS = ("-m", "dryml.code.probe_worker", "--json")
PROBE_RESULT_KIND = "dryml.code_probe_result"


class _InvalidTimeoutError(ValueError):
    """Raised when a serialized probe timeout is not a usable deadline."""


@dataclass(frozen=True, slots=True)
class CodeProbeRequest:
    """Serializable request for lightweight code analysis.

    Args:
        target: Serializable code target spec. Other target inputs are
            normalized to a spec during construction.
        algorithms: Analyzer names to run. Defaults to the built-in lightweight
            analyzer set.
        include_environment_record: Whether to include an observed
            ``EnvironmentRecord`` from the process running the probe.
        args: Reserved JSON-compatible positional metadata. The probe does not
            call the target with these arguments.
        kwargs: Reserved JSON-compatible keyword metadata. The probe does not
            call the target with these arguments.
        runtime_mode: ``"probe"`` is the supported runtime mode.
        policy: Probe policy name. ``"lightweight"`` is the supported policy.
        timeout_s: Optional parent-side subprocess timeout in seconds.
        metadata: JSON-compatible caller metadata passed to analyzers.
    """

    target: CodeTargetSpec
    algorithms: tuple[str, ...] = DEFAULT_PROBE_ALGORITHMS
    include_environment_record: bool = True
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    runtime_mode: str = "probe"
    policy: str = "lightweight"
    timeout_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.target) is not CodeTargetSpec:
            object.__setattr__(self, "target", normalize_target(self.target, metadata=self.metadata).spec)
        object.__setattr__(self, "algorithms", _coerce_algorithms(self.algorithms))
        object.__setattr__(self, "args", tuple(json_compatible(self.args)))
        object.__setattr__(self, "kwargs", dict(json_compatible(self.kwargs)))
        object.__setattr__(self, "runtime_mode", str(self.runtime_mode).strip().lower())
        object.__setattr__(self, "policy", str(self.policy).strip().lower())
        object.__setattr__(self, "metadata", dict(json_compatible(self.metadata)))
        if self.timeout_s is not None:
            object.__setattr__(self, "timeout_s", _validated_timeout(self.timeout_s))

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible request data using schema version 1."""

        return {
            "schema_version": PROBE_SCHEMA_VERSION,
            "target": self.target.to_data(),
            "algorithms": list(self.algorithms),
            "include_environment_record": self.include_environment_record,
            "args": list(json_compatible(self.args)),
            "kwargs": json_compatible(self.kwargs),
            "runtime_mode": self.runtime_mode,
            "policy": self.policy,
            "timeout_s": self.timeout_s,
            "metadata": json_compatible(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CodeProbeRequest":
        """Build a request from JSON-compatible mapping data."""

        if not isinstance(data, Mapping):
            raise TypeError("CodeProbeRequest data must be a mapping")
        _validate_schema_version(data)
        if "target" not in data:
            raise ValueError("CodeProbeRequest data is missing required 'target'")
        return cls(
            target=CodeTargetSpec.from_data(data["target"]),
            algorithms=_coerce_algorithms(data.get("algorithms")),
            include_environment_record=bool(data.get("include_environment_record", True)),
            args=tuple(data.get("args") or ()),
            kwargs=data.get("kwargs") or {},
            runtime_mode=data.get("runtime_mode", "probe"),
            policy=data.get("policy", "lightweight"),
            timeout_s=data.get("timeout_s"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True, slots=True)
class CodeProbeResult:
    """Result of a lightweight code probe.

    Args:
        ok: Whether no error-severity diagnostics were emitted. Construction
            normalizes this value from diagnostics so serialized failures remain
            consistent.
        analysis: Optional reusable code-analysis result.
        environment_record: Optional observed environment record from the probe
            process.
        diagnostics: Structured request, analysis, environment, or worker
            diagnostics.
        stdout: Captured user-code stdout from import/analysis.
        stderr: Captured user-code stderr from import/analysis or worker stderr
            when subprocess protocol handling fails.
    """

    ok: bool
    analysis: CodeAnalysisResult | None
    environment_record: EnvironmentRecord | None
    diagnostics: tuple[DiagnosticFact, ...] = ()
    stdout: str | None = None
    stderr: str | None = None

    def __post_init__(self) -> None:
        diagnostics = tuple(self.diagnostics or ())
        all_diagnostics = diagnostics + (() if self.analysis is None else self.analysis.diagnostics)
        if probe_ok(all_diagnostics) and self.analysis is None:
            raise ValueError("successful CodeProbeResult requires analysis")
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "ok", probe_ok(all_diagnostics))

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible result data using schema version 1."""

        return {
            "kind": PROBE_RESULT_KIND,
            "schema_version": PROBE_SCHEMA_VERSION,
            "ok": self.ok,
            "analysis": None if self.analysis is None else self.analysis.to_data(),
            "environment_record": None if self.environment_record is None else self.environment_record.to_data(),
            "diagnostics": [diagnostic.to_data() for diagnostic in self.diagnostics],
            "stdout": self.stdout,
            "stderr": self.stderr,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CodeProbeResult":
        """Build a probe result from JSON-compatible mapping data."""

        if not isinstance(data, Mapping):
            raise TypeError("CodeProbeResult data must be a mapping")
        if data.get("kind") != PROBE_RESULT_KIND:
            raise ValueError("unsupported code probe result kind")
        _validate_schema_version(data)
        if not isinstance(data.get("ok"), bool):
            raise TypeError("CodeProbeResult ok must be a boolean")
        diagnostics = tuple(DiagnosticFact.from_data(item) for item in data.get("diagnostics") or ())
        analysis = None if data.get("analysis") is None else CodeAnalysisResult.from_data(data["analysis"])
        all_diagnostics = diagnostics + (() if analysis is None else analysis.diagnostics)
        if data["ok"] != probe_ok(all_diagnostics):
            raise ValueError("CodeProbeResult ok does not match diagnostics")
        if data["ok"] and analysis is None:
            raise ValueError("successful CodeProbeResult requires analysis")
        return cls(
            ok=data["ok"],
            analysis=analysis,
            environment_record=(
                None
                if data.get("environment_record") is None
                else EnvironmentRecord.from_data(data["environment_record"])
            ),
            diagnostics=diagnostics,
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
        )


def diagnostic(code: str, message: str, *, severity: str = "error", data: Mapping[str, Any] | None = None) -> DiagnosticFact:
    """Construct a code-probe diagnostic fact."""

    return DiagnosticFact(
        severity=severity,
        code=code,
        message=message,
        source={"component": "dryml.code.probe"},
        data=data or {},
    )


def probe_ok(diagnostics: Iterable[DiagnosticFact]) -> bool:
    """Return true when *diagnostics* contains no error-severity item."""

    return not any(item.severity == "error" for item in diagnostics)


def normalize_probe_request(request: CodeProbeRequest | Mapping[str, Any]) -> CodeProbeRequest:
    """Normalize a request object or mapping and validate supported probe policy."""

    normalized = request if isinstance(request, CodeProbeRequest) else CodeProbeRequest.from_data(request)
    if normalized.runtime_mode != "probe":
        raise ValueError(f"unsupported code probe runtime_mode {normalized.runtime_mode!r}")
    if normalized.policy != "lightweight":
        raise ValueError(f"unsupported code probe policy {normalized.policy!r}")
    return normalized


def run_probe_request(
    request: CodeProbeRequest | Mapping[str, Any],
    *,
    environment: Any | None = None,
    require_stable_import_path: bool = False,
) -> CodeProbeResult:
    """Execute a normalized code probe in the current process.

    Args:
        request: Probe request object or serialized request mapping.
        environment: Optional current-process environment metadata. This
            parameter does not create or solve environments.
        require_stable_import_path: Reject targets without an import path. Probe
            workers enable this because they cannot reconstruct source specs.

    Returns:
        A ``CodeProbeResult`` containing code facts, diagnostics, optional
        environment record, and captured user output.
    """

    del environment
    try:
        normalized = normalize_probe_request(request)
    except Exception as exc:
        return CodeProbeResult(
            ok=False,
            analysis=None,
            environment_record=None,
            diagnostics=(diagnostic("code_probe.invalid_request", "Invalid code probe request.", data={"error": repr(exc)}),),
        )

    if require_stable_import_path and not _target_can_run_in_subprocess(normalized.target):
        return _non_serializable_target_result(normalized.target)
    return _run_probe_request_normalized(normalized, analysis_target=normalized.target)


def _run_probe_request_normalized(normalized: CodeProbeRequest, *, analysis_target: Any) -> CodeProbeResult:
    diagnostics: list[DiagnosticFact] = _request_diagnostics(normalized)
    if diagnostics:
        return CodeProbeResult(ok=False, analysis=None, environment_record=None, diagnostics=tuple(diagnostics))

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    analysis: CodeAnalysisResult | None = None
    environment_record: EnvironmentRecord | None = None

    try:
        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            with runtime.enter_runtime(runtime.RuntimeMode.PROBE) as state:
                if state.allocation is not runtime.NoAllocation:
                    diagnostics.append(diagnostic(
                        "code_probe.unexpected_error",
                        "Probe runtime unexpectedly has a workload allocation.",
                        data={"allocation": repr(state.allocation)},
                    ))
                if normalized.include_environment_record:
                    try:
                        environment_record = environments.inspect_current()
                    except Exception as exc:
                        diagnostics.append(diagnostic(
                            "code_probe.environment_record_error",
                            "Environment record collection failed.",
                            data={"error": repr(exc)},
                        ))
                context = CodeAnalysisContext(
                    algorithms=normalized.algorithms,
                    allow_import=True,
                    allow_source=True,
                    allow_dynamic_execution=False,
                    include_annotations=True,
                    diagnostics_policy="collect",
                    metadata={"probe_policy": normalized.policy, **dict(normalized.metadata)},
                )
                analysis = analyze(analysis_target, algorithms=normalized.algorithms, context=context)
    except Exception as exc:
        diagnostics.append(diagnostic("code_probe.unexpected_error", "Code probe failed unexpectedly.", data={"error": repr(exc)}))

    if analysis is not None:
        diagnostics.extend(_probe_diagnostics_from_analysis(analysis))
        diagnostics.extend(analysis.diagnostics)

    return CodeProbeResult(
        ok=probe_ok(diagnostics),
        analysis=analysis,
        environment_record=environment_record,
        diagnostics=tuple(diagnostics),
        stdout=captured_stdout.getvalue() or None,
        stderr=captured_stderr.getvalue() or None,
    )


def probe_target(
    target: Any,
    *,
    environment: EnvironmentSpec | None = None,
    algorithms: Iterable[str] | None = None,
    include_environment_record: bool = True,
    timeout: float | None = None,
    policy: str = "lightweight",
    metadata: Mapping[str, Any] | None = None,
) -> CodeProbeResult:
    """Normalize and probe a target in the current or requested Python environment.

    Args:
        target: Live Python target, import-path string, or ``CodeTargetSpec`` to
            inspect with the registered non-invoking analyzers.
        environment: Execution location. ``None`` or
            ``CurrentEnvironmentSpec`` runs inline unless ``timeout`` requests
            an isolated current-Python worker. ``PythonExecutableSpec`` and
            ``CondaEnvironmentSpec`` run a probe worker in that environment.
        algorithms: Analyzer names to run, or the lightweight default set when
            omitted.
        include_environment_record: Include an ``EnvironmentRecord`` observed
            in the process that performs analysis.
        timeout: Finite positive subprocess deadline in seconds. Supplying one
            requires a target with a stable worker-reconstructible import path.
        policy: Probe policy name passed to the analyzers. ``"lightweight"`` is
            the supported policy.
        metadata: JSON-compatible caller metadata passed through target
            normalization and analyzer context.

    Returns:
        A ``CodeProbeResult`` containing analysis facts, optional environment
        data, captured output, and structured diagnostics. Invalid targets or
        timeouts, unsupported locations, non-reconstructible worker targets,
        import/analysis failures, worker startup failures, timeouts, and
        protocol errors normally produce ``ok=False`` rather than raising.

    Import-path targets may execute module-level code while being imported by
    Python in the current process or selected probe worker. Probe mode uses
    ``RuntimeMode.PROBE`` with no workload allocation and does not execute target
    function bodies, instantiate classes, or enable dynamic tracing. Probes run
    trusted imports and analyzers; process isolation is a lifecycle boundary,
    not a sandbox. Subprocess execution cannot carry live bound-method receiver
    state, and source-spec reconstruction is not implemented. Unsupported
    environment specs return structured diagnostics rather than attempting
    package solving, container execution, or world synthesis.
    """

    try:
        timeout = _validated_timeout(timeout) if timeout is not None else None
    except (TypeError, ValueError):
        return CodeProbeResult(
            ok=False,
            analysis=None,
            environment_record=None,
            diagnostics=(diagnostic(
                "code_probe.invalid_timeout",
                "Code probe timeout must be a finite positive number of seconds.",
            ),),
        )

    code_target: CodeTarget | None = None
    try:
        if type(target) is CodeTargetSpec:
            target_spec = target
            target_kind = target.kind
        elif type(target) is str:
            target_spec = CodeTargetSpec.from_import_path(target)
            target_kind = target_spec.kind
        else:
            code_target = normalize_target(target, metadata=metadata or {})
            target_spec = code_target.spec
            target_kind = code_target.spec.kind
    except Exception as exc:
        return CodeProbeResult(
            ok=False,
            analysis=None,
            environment_record=None,
            diagnostics=(diagnostic("code_probe.target_normalization_error", "Code target normalization failed.", data={"error": repr(exc)}),),
        )
    if target_kind == "unknown":
        return CodeProbeResult(
            ok=False,
            analysis=None,
            environment_record=None,
            diagnostics=(diagnostic(
                "code_probe.target_normalization_error",
                "Unsupported code probe target type.",
                data={"target_type": type(target).__name__},
            ),),
        )

    request = CodeProbeRequest(
        target=target_spec,
        algorithms=tuple(algorithms or DEFAULT_PROBE_ALGORITHMS),
        include_environment_record=include_environment_record,
        runtime_mode="probe",
        policy=policy,
        timeout_s=timeout,
        metadata=metadata or {},
    )
    if environment is None or isinstance(environment, CurrentEnvironmentSpec):
        if timeout is not None:
            if target_spec.kind == "bound_method":
                return _bound_method_worker_result(target_spec)
            if _target_can_run_in_subprocess(target_spec):
                env = build_probe_env(base=None, overrides=None, pythonpath_policy="inherit")
                return probe_target_in_subprocess(request, [sys.executable, *PROBE_WORKER_ARGS], timeout=timeout, env=env)
            if target_spec.source_spec is not None:
                return _non_serializable_target_result(target_spec)
            return CodeProbeResult(
                ok=False,
                analysis=None,
                environment_record=None,
                diagnostics=(diagnostic(
                    "code_probe.timeout",
                    "Current-process timeout is unsupported for live non-serializable code probe targets.",
                    data={"provider": "current_process", "target": _diagnostic_target(target_spec)},
                ),),
            )
        if code_target is not None:
            return _run_probe_request_normalized(request, analysis_target=code_target)
        return run_probe_request(request, environment=environment)
    if isinstance(environment, PythonExecutableSpec):
        if target_spec.kind == "bound_method":
            return _bound_method_worker_result(target_spec)
        if not _target_can_run_in_subprocess(target_spec):
            return _non_serializable_target_result(target_spec)
        return probe_target_in_subprocess(request, _python_command(environment), timeout=timeout, env=_python_env(environment))
    if isinstance(environment, CondaEnvironmentSpec):
        if target_spec.kind == "bound_method":
            return _bound_method_worker_result(target_spec)
        if not _target_can_run_in_subprocess(target_spec):
            return _non_serializable_target_result(target_spec)
        try:
            command = _conda_command(environment)
        except EnvironmentSpecError as exc:
            return _unsupported_environment_result(environment, str(exc))
        return probe_target_in_subprocess(request, command, timeout=timeout, env=_python_env(environment))
    if isinstance(environment, ContainerEnvironmentSpec):
        return _unsupported_environment_result(environment, "container code probing is not implemented")
    return _unsupported_environment_result(environment, f"unsupported environment spec {type(environment).__name__}")


def probe_target_in_subprocess(
    request: CodeProbeRequest,
    command: list[str],
    *,
    timeout: float | None,
    env: Mapping[str, str] | None = None,
) -> CodeProbeResult:
    """Launch a probe worker command and decode its JSON result."""

    try:
        timeout = _validated_timeout(timeout) if timeout is not None else None
    except (TypeError, ValueError):
        return CodeProbeResult(
            ok=False,
            analysis=None,
            environment_record=None,
            diagnostics=(diagnostic(
                "code_probe.invalid_timeout",
                "Code probe timeout must be a finite positive number of seconds.",
            ),),
        )
    payload = json.dumps(request.to_data(), sort_keys=True).encode("utf-8")
    try:
        returncode, protocol, protocol_truncated, stderr_bytes, stderr_truncated, timed_out = _run_bounded_command(
            command,
            timeout=timeout,
            env=dict(env) if env is not None else None,
            input_data=payload,
        )
    except OSError as exc:
        return CodeProbeResult(
            ok=False,
            analysis=None,
            environment_record=None,
            diagnostics=(diagnostic("code_probe.unsupported_environment", "Code probe worker could not be started.", data={"error": repr(exc)}),),
        )
    stdout, stdout_truncated = _diagnostic_output(protocol)
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if timed_out:
        return CodeProbeResult(
            ok=False,
            analysis=None,
            environment_record=None,
            diagnostics=(diagnostic(
                "code_probe.timeout",
                f"Code probe timed out after {timeout} seconds.",
                data={"provider": "subprocess", "target": _diagnostic_target(request.target)},
            ),),
            stdout=stdout,
            stderr=stderr,
        )
    if protocol_truncated:
        return CodeProbeResult(
            ok=False,
            analysis=None,
            environment_record=None,
            diagnostics=(diagnostic("code_probe.worker_protocol_error", "Code probe worker output exceeded the bounded protocol limit."),),
            stdout=stdout,
            stderr=stderr,
        )

    try:
        result = CodeProbeResult.from_data(json.loads(protocol.decode("utf-8")))
    except Exception as exc:
        return CodeProbeResult(
            ok=False,
            analysis=None,
            environment_record=None,
            diagnostics=(diagnostic(
                "code_probe.worker_protocol_error",
                "Code probe worker did not emit valid result JSON.",
                data={"error": repr(exc), "returncode": returncode},
            ),),
            stdout=stdout,
            stderr=stderr,
        )
    if returncode != 0:
        extra = diagnostic(
            "code_probe.worker_protocol_error",
            f"Code probe worker exited with status {returncode}.",
            data={"returncode": returncode},
        )
        return CodeProbeResult(
            ok=False,
            analysis=result.analysis,
            environment_record=result.environment_record,
            diagnostics=result.diagnostics + (extra,),
            stdout=result.stdout,
            stderr=result.stderr or stderr or None,
        )
    return result


def request_from_data(data: Mapping[str, Any]) -> CodeProbeRequest:
    """Deserialize a code probe request."""

    return CodeProbeRequest.from_data(data)


def result_from_data(data: Mapping[str, Any]) -> CodeProbeResult:
    """Deserialize a code probe result."""

    return CodeProbeResult.from_data(data)


def _coerce_algorithms(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_PROBE_ALGORITHMS
    if isinstance(value, str):
        return (value,)
    algorithms = tuple(str(item) for item in value)
    return algorithms or DEFAULT_PROBE_ALGORITHMS


def _validated_timeout(timeout: Any) -> float:
    """Return a finite positive worker deadline or raise ``ValueError``."""

    value = float(timeout)
    if not math.isfinite(value) or value <= 0:
        raise _InvalidTimeoutError("timeout must be finite and positive")
    return value


def _validate_schema_version(data: Mapping[str, Any]) -> None:
    version = data.get("schema_version", PROBE_SCHEMA_VERSION)
    if version != PROBE_SCHEMA_VERSION:
        raise ValueError(f"unsupported code probe schema_version {version!r}")


def _target_can_run_in_subprocess(target: CodeTargetSpec) -> bool:
    """Return whether *target* has the stable import path required by workers."""

    if target.kind == "bound_method" or not isinstance(target.import_path, str):
        return False
    module_name, separator, qualname = target.import_path.partition(":")
    if not separator or module_name == "__main__" or not module_name or not qualname:
        return False
    return (
        all(part.isidentifier() for part in module_name.split("."))
        and all(part.isidentifier() for part in qualname.split("."))
    )


def _non_serializable_target_result(target: CodeTargetSpec) -> CodeProbeResult:
    if target.source_spec is not None:
        return CodeProbeResult(
            ok=False,
            analysis=None,
            environment_record=None,
            diagnostics=(diagnostic(
                "code_probe.source_spec_reconstruction_unavailable",
                "Source-spec reconstruction is not implemented for subprocess code probes.",
                data={"target": target.to_data()},
            ),),
        )
    return CodeProbeResult(
        ok=False,
        analysis=None,
        environment_record=None,
        diagnostics=(diagnostic(
            "code_probe.non_serializable_target",
            "Target cannot be probed in a subprocess without a stable import path.",
            data={"target": target.to_data()},
        ),),
    )


def _bound_method_worker_result(target: CodeTargetSpec) -> CodeProbeResult:
    """Reject worker routing that would silently drop a live receiver instance."""

    return CodeProbeResult(
        ok=False,
        analysis=None,
        environment_record=None,
        diagnostics=(diagnostic(
            "code_probe.bound_method_receiver_unavailable",
            "A live bound-method receiver cannot be reconstructed in a subprocess code probe.",
            data={"target": target.to_data()},
        ),),
    )


def _request_diagnostics(request: CodeProbeRequest) -> list[DiagnosticFact]:
    diagnostics: list[DiagnosticFact] = []
    if request.runtime_mode != "probe":
        diagnostics.append(diagnostic(
            "code_probe.unsupported_runtime_mode",
            f"Unsupported code probe runtime mode {request.runtime_mode!r}.",
            data={"runtime_mode": request.runtime_mode},
        ))
    available = set(available_analyzers())
    for name in request.algorithms:
        if name not in available:
            diagnostics.append(diagnostic(
                "code_probe.unknown_algorithm",
                f"Unknown code probe analyzer {name!r}.",
                data={"algorithm": name},
            ))
    return diagnostics


def _probe_diagnostics_from_analysis(analysis: CodeAnalysisResult) -> tuple[DiagnosticFact, ...]:
    diagnostics: list[DiagnosticFact] = []
    if any(item.code == "dryml.code.import_failed" for item in analysis.diagnostics):
        diagnostics.append(diagnostic(
            "code_probe.import_error",
            "Code probe target import failed.",
            data={"provider": "current_process", "target": _diagnostic_target(analysis.target)},
        ))
    if any(item.code == "dryml.code.algorithm_failed" for item in analysis.diagnostics):
        diagnostics.append(diagnostic(
            "code_probe.analysis_error",
            "Code probe analysis failed.",
            data={"target": analysis.target.to_data()},
        ))
    return tuple(diagnostics)


def _diagnostic_target(target: CodeTargetSpec) -> dict[str, str]:
    """Return bounded target identity without source, metadata, or live values."""

    data = {"kind": target.kind[:64]}
    for name in ("import_path", "method_name", "subject_ref"):
        value = getattr(target, name)
        if isinstance(value, str):
            data[name] = value[:512]
    return data


def _python_command(spec: PythonExecutableSpec) -> list[str]:
    return [spec.executable, *PROBE_WORKER_ARGS]


def _conda_command(spec: CondaEnvironmentSpec) -> list[str]:
    if spec.launch_mode == "direct":
        return [spec.direct_python_executable(), *PROBE_WORKER_ARGS]
    if not spec.prefix and not spec.name:
        raise EnvironmentSpecError("conda-run launch requires prefix or name")
    command = [spec.conda_executable, "run"]
    if spec.prefix:
        command.extend(["-p", spec.prefix])
    else:
        command.extend(["-n", spec.name or ""])
    command.extend(["--no-capture-output", "--", "python", *PROBE_WORKER_ARGS])
    return command


def _python_env(spec: PythonExecutableSpec | CondaEnvironmentSpec) -> dict[str, str]:
    return build_probe_env(
        base=None,
        overrides=spec.env,
        pythonpath_policy=spec.pythonpath_policy,
        extra_pythonpath=spec.extra_pythonpath,
    )


def _unsupported_environment_result(environment: Any, message: str) -> CodeProbeResult:
    data = environment.to_data() if hasattr(environment, "to_data") else {"type": type(environment).__name__}
    return CodeProbeResult(
        ok=False,
        analysis=None,
        environment_record=None,
        diagnostics=(diagnostic("code_probe.unsupported_environment", message, data={"environment": data}),),
    )


__all__ = [
    "CodeProbeRequest",
    "CodeProbeResult",
    "DEFAULT_PROBE_ALGORITHMS",
    "PROBE_SCHEMA_VERSION",
    "diagnostic",
    "normalize_probe_request",
    "probe_ok",
    "probe_target",
    "probe_target_in_subprocess",
    "request_from_data",
    "result_from_data",
    "run_probe_request",
]
