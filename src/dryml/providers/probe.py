"""Subprocess runner for target-environment provider probes."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from dryml.environments import CondaEnvironmentSpec, ContainerEnvironmentSpec, CurrentEnvironmentSpec, PythonExecutableSpec, build_probe_env
from dryml.environments.specs import EnvironmentSpec
from dryml.operations import attach_operation_id
from dryml.runtime import RuntimeMode, attach_runtime_id, make_runtime_spec

from .identity import ProviderRef
from .registry import ProviderRegistry
from .reports import ProbeReport, ProviderIssue
from .requests import OperationInspectionRequest, ProbePolicy, ProviderRequest
from .probe_worker import PROVIDER_PROBE_REQUEST_SCHEMA, PROVIDER_PROBE_RESPONSE_SCHEMA


PROVIDER_WORKER_COMMAND = ("-m", "dryml.providers.probe_worker", "--json")


def run_probe(
    request: ProviderRequest,
    *,
    environment: EnvironmentSpec | None = None,
    providers: Iterable[str | ProviderRef] | None = None,
    registry: ProviderRegistry | None = None,
    timeout: float | None = None,
) -> ProbeReport:
    """Run a provider probe in a subprocess and return a structured report."""

    environment = environment or CurrentEnvironmentSpec()
    runtime_spec = attach_runtime_id(make_runtime_spec(mode=RuntimeMode.PROBE, device_visibility={"policy": request.probe_policy.device_visibility}))
    environment_spec = _environment_spec_data(environment)
    request = replace(request, environment_spec=environment_spec, runtime_spec=runtime_spec)
    try:
        provider_refs = _resolve_provider_refs(request, providers=providers, registry=registry)
    except Exception as exc:
        return _runner_failure(request, "provider_resolution_failed", str(exc), exception=exc, environment_spec=environment_spec, runtime_spec=runtime_spec)
    if isinstance(environment, ContainerEnvironmentSpec):
        return _runner_failure(request, "unsupported_environment_spec", "container provider probes are not implemented", environment_spec=environment_spec, runtime_spec=runtime_spec)
    try:
        command, env = _command_for_environment(environment)
    except Exception as exc:
        return _runner_failure(request, "unsupported_environment_spec", str(exc), exception=exc, environment_spec=environment_spec, runtime_spec=runtime_spec)
    envelope = {
        "schema": PROVIDER_PROBE_REQUEST_SCHEMA,
        "schema_version": 1,
        "request": request.to_data(),
        "providers": [ref.to_data() for ref in provider_refs],
        "runtime_spec": runtime_spec,
        "probe_policy": request.probe_policy.to_data(),
    }
    timeout = request.probe_policy.timeout if timeout is None else timeout
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(envelope, sort_keys=True, separators=(",", ":")),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return _runner_failure(request, "probe_timeout", f"provider probe timed out after {timeout} seconds", stdout=_text(exc.stdout), stderr=_text(exc.stderr), environment_spec=environment_spec, runtime_spec=runtime_spec)
    except OSError as exc:
        return _runner_failure(request, "probe_failed", f"provider probe could not start: {exc}", exception=exc, environment_spec=environment_spec, runtime_spec=runtime_spec)
    if completed.returncode != 0:
        parsed = _parse_worker_response(completed.stdout, request, stderr=completed.stderr, returncode=completed.returncode)
        if parsed is not None:
            return parsed
        return _runner_failure(request, "probe_failed", f"provider probe exited with status {completed.returncode}", stdout=completed.stdout, stderr=completed.stderr, returncode=completed.returncode, environment_spec=environment_spec, runtime_spec=runtime_spec)
    parsed = _parse_worker_response(completed.stdout, request, stderr=completed.stderr, returncode=completed.returncode)
    if parsed is None:
        return _runner_failure(request, "malformed_worker_output", "provider probe returned malformed JSON", stdout=completed.stdout, stderr=completed.stderr, returncode=completed.returncode, environment_spec=environment_spec, runtime_spec=runtime_spec)
    return parsed


def probe_operation(
    operation_spec: Mapping[str, Any],
    *,
    environment: EnvironmentSpec | None = None,
    providers: Iterable[str | ProviderRef] = (),
    registry: ProviderRegistry | None = None,
    timeout: float | None = 30.0,
    probe_policy: ProbePolicy | None = None,
    provider_options: Mapping[str, Any] | None = None,
) -> ProbeReport:
    """Convenience wrapper for operation-inspection probes."""

    attached = attach_operation_id(operation_spec)
    environment = environment or CurrentEnvironmentSpec()
    runtime_spec = attach_runtime_id(make_runtime_spec(mode=RuntimeMode.PROBE, device_visibility={"policy": "none"}))
    environment_spec = dict(environment.to_data())
    environment_spec["id"] = environment.id
    policy = probe_policy or ProbePolicy(timeout=timeout)
    request = OperationInspectionRequest(
        operation_spec=attached,
        environment_spec=environment_spec,
        runtime_spec=runtime_spec,
        provider_names=tuple(ref if isinstance(ref, str) else ref.name for ref in providers),
        provider_options=provider_options or {},
        probe_policy=policy,
    )
    return run_probe(request, environment=environment, providers=providers, registry=registry, timeout=timeout)


def _resolve_provider_refs(request: ProviderRequest, *, providers: Iterable[str | ProviderRef] | None, registry: ProviderRegistry | None) -> tuple[ProviderRef, ...]:
    items = tuple(providers or request.provider_names)
    refs: list[ProviderRef] = []
    for item in items:
        if isinstance(item, ProviderRef):
            refs.append(item)
        else:
            if registry is None:
                raise ValueError("registry is required when providers are named by string")
            refs.append(registry.get_ref(str(item)))
    if not refs:
        raise ValueError("at least one provider is required")
    return tuple(refs)


def _environment_spec_data(environment: EnvironmentSpec) -> dict[str, Any]:
    data = dict(environment.to_data())
    data["id"] = environment.id
    return data


def _command_for_environment(environment: EnvironmentSpec) -> tuple[list[str], dict[str, str] | None]:
    if isinstance(environment, CurrentEnvironmentSpec):
        return [sys.executable, *PROVIDER_WORKER_COMMAND], build_probe_env(base=None, overrides=None, pythonpath_policy="inherit")
    if isinstance(environment, PythonExecutableSpec):
        return [environment.executable, *PROVIDER_WORKER_COMMAND], build_probe_env(base=None, overrides=environment.env, pythonpath_policy=environment.pythonpath_policy, extra_pythonpath=environment.extra_pythonpath)
    if isinstance(environment, CondaEnvironmentSpec):
        if environment.launch_mode == "direct":
            command = [environment.direct_python_executable(), *PROVIDER_WORKER_COMMAND]
        else:
            command = [environment.conda_executable, "run"]
            if environment.prefix:
                command.extend(["-p", environment.prefix])
            else:
                command.extend(["-n", environment.name or ""])
            command.extend(["--no-capture-output", "--", "python", *PROVIDER_WORKER_COMMAND])
        return command, build_probe_env(base=None, overrides=environment.env, pythonpath_policy=environment.pythonpath_policy, extra_pythonpath=environment.extra_pythonpath)
    raise TypeError(f"unsupported environment spec {type(environment).__name__}")


def _parse_worker_response(stdout: str, request: ProviderRequest, *, stderr: str, returncode: int) -> ProbeReport | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping) or payload.get("schema") != PROVIDER_PROBE_RESPONSE_SCHEMA:
        return None
    try:
        report = ProbeReport.from_data(payload.get("probe_report") or {})
    except Exception as exc:
        return _runner_failure(request, "invalid_worker_report", f"provider probe report could not be decoded: {exc}", stdout=stdout, stderr=stderr, returncode=returncode)
    return report


def _runner_failure(request: ProviderRequest | None, code: str, message: str, *, exception: BaseException | None = None, stdout: str | None = None, stderr: str | None = None, returncode: int | None = None, environment_spec: Mapping[str, Any] | None = None, runtime_spec: Mapping[str, Any] | None = None) -> ProbeReport:
    metadata = {"returncode": returncode}
    issue = ProviderIssue(code, "error", message, exception_type=type(exception).__name__ if exception is not None else None, metadata=metadata)
    return ProbeReport(
        request=None if request is None else request.to_data(),
        operation_id=getattr(request, "operation_id", None),
        environment_spec=environment_spec or getattr(request, "environment_spec", None),
        environment_spec_id=(environment_spec or getattr(request, "environment_spec", {}) or {}).get("id") if isinstance(environment_spec or getattr(request, "environment_spec", None), Mapping) else None,
        runtime_spec=runtime_spec or getattr(request, "runtime_spec", None),
        runtime_id=(runtime_spec or getattr(request, "runtime_spec", {}) or {}).get("id") if isinstance(runtime_spec or getattr(request, "runtime_spec", None), Mapping) else None,
        probe_policy=getattr(request, "probe_policy", ProbePolicy()).to_data() if request is not None else ProbePolicy().to_data(),
        status="failed",
        diagnostics=(issue,),
        metadata={"stdout": stdout, "stderr": stderr, "returncode": returncode},
    )


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


__all__ = ["PROVIDER_WORKER_COMMAND", "probe_operation", "run_probe"]
