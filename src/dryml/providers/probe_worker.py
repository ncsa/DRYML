"""JSON stdin/stdout worker for target-environment provider probes."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from dryml.runtime import FrameworkBootstrapPolicy, NoAllocation, RuntimeContextSpec, RuntimeMode, activate, active_runtime, active_runtime_bootstrap, make_runtime_spec, attach_runtime_id


from .errors import ProviderError, ProviderProtocolError
from .identity import ProviderRef
from .registry import load_provider_ref
from .reports import OperationInspectionReport, ProbeReport, ProviderIssue, ProviderReport, report_from_data
from .requests import OperationInspectionRequest, ProbePolicy, request_from_data


PROVIDER_PROBE_REQUEST_SCHEMA = "dryml.provider_probe_request.v1"
PROVIDER_PROBE_RESPONSE_SCHEMA = "dryml.provider_probe_response.v1"
PROVIDER_PROBE_SCHEMA_VERSION = 1


def main(argv: list[str] | None = None) -> int:
    """Run the provider probe worker command."""

    parser = argparse.ArgumentParser(prog="python -m dryml.providers.probe_worker")
    parser.add_argument("--json", action="store_true", help="read one JSON request from stdin and write one JSON response")
    args = parser.parse_args(argv)
    if not args.json:
        parser.print_help(sys.stderr)
        return 2
    raw = sys.stdin.read()
    response = handle_json_request(raw)
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0 if response.get("ok") else 1


def handle_json_request(raw: str) -> dict[str, Any]:
    """Handle one worker JSON protocol request."""

    try:
        envelope = json.loads(raw)
        report = run_worker_request(envelope)
        return {"schema": PROVIDER_PROBE_RESPONSE_SCHEMA, "schema_version": PROVIDER_PROBE_SCHEMA_VERSION, "ok": report.status != "failed", "probe_report": report.to_data()}
    except Exception as exc:
        report = _protocol_failure(str(exc), exception=exc)
        return {"schema": PROVIDER_PROBE_RESPONSE_SCHEMA, "schema_version": PROVIDER_PROBE_SCHEMA_VERSION, "ok": False, "probe_report": report.to_data()}


def run_worker_request(envelope: Mapping[str, Any]) -> ProbeReport:
    """Validate and execute a worker protocol envelope."""

    request, providers, runtime_spec_data, probe_policy = _parse_envelope(envelope)
    runtime_payload = _runtime_payload(runtime_spec_data, probe_policy)
    runtime_spec = RuntimeContextSpec.from_data(runtime_payload)
    runtime_envelope = attach_runtime_id(make_runtime_spec(**runtime_spec.to_data()))
    provider_reports: list[ProviderReport] = []
    diagnostics: list[ProviderIssue] = []
    with activate(mode=RuntimeMode.PROBE, allocation=NoAllocation, spec=runtime_spec, policy=FrameworkBootstrapPolicy(strict_preimport=probe_policy.strict_preimport), restore_environ=False):
        for ref in providers:
            identity = ref.fallback_identity()
            try:
                provider = load_provider_ref(ref)
                identity = provider.identity
                provider_reports.append(_with_context_ids(_run_provider(provider, request), request, runtime_envelope))
            except Exception as exc:
                report_type = _report_type_for_request(request)
                provider_reports.append(_with_context_ids(report_type.failed(identity, request, f"provider failed: {exc}", exception=exc), request, runtime_envelope))
        runtime = active_runtime()
        bootstrap = active_runtime_bootstrap()
        diagnostics.append(ProviderIssue("runtime_mode", "info", "probe worker runtime mode", actual=runtime.mode.value))
        diagnostics.append(ProviderIssue("runtime_allocation", "info", "probe worker allocation", actual=repr(runtime.allocation)))
        if bootstrap is not None:
            diagnostics.append(ProviderIssue("device_visibility", "info", "probe worker device visibility", actual=bootstrap.env_updates))
    status = "ok" if all(report.status in {"ok", "unsupported"} for report in provider_reports) else "failed"
    return ProbeReport(
        request=request.to_data(),
        reports=tuple(provider_reports),
        operation_id=getattr(request, "operation_id", None),
        environment_spec=request.environment_spec,
        environment_spec_id=(request.environment_spec or {}).get("id") if isinstance(request.environment_spec, Mapping) else None,
        runtime_spec=runtime_envelope,
        runtime_id=runtime_envelope.get("id"),
        probe_policy=probe_policy.to_data(),
        status=status,
        diagnostics=tuple(diagnostics),
    )


def _parse_envelope(envelope: Mapping[str, Any]):
    if not isinstance(envelope, Mapping):
        raise ProviderProtocolError("provider probe request envelope must be a mapping")
    if envelope.get("schema") != PROVIDER_PROBE_REQUEST_SCHEMA or envelope.get("schema_version") != PROVIDER_PROBE_SCHEMA_VERSION:
        raise ProviderProtocolError("provider probe request schema mismatch", context={"schema": envelope.get("schema"), "schema_version": envelope.get("schema_version")})
    unknown = set(envelope) - {"schema", "schema_version", "request", "providers", "runtime_spec", "probe_policy"}
    if unknown:
        raise ProviderProtocolError("provider probe request has unknown fields", context={"fields": sorted(unknown)})
    request = request_from_data(envelope.get("request") or {})
    providers = tuple(ProviderRef.from_data(item) for item in envelope.get("providers") or ())
    if not providers:
        raise ProviderProtocolError("provider probe request requires at least one provider")
    probe_policy = ProbePolicy.from_data(envelope.get("probe_policy") or (request.probe_policy.to_data() if request.probe_policy else None))
    runtime_spec = envelope.get("runtime_spec")
    if runtime_spec is not None and not isinstance(runtime_spec, Mapping):
        raise ProviderProtocolError("runtime_spec must be a mapping")
    return request, providers, runtime_spec, probe_policy


def _runtime_payload(runtime_spec_data: Mapping[str, Any] | None, probe_policy: ProbePolicy) -> dict[str, Any]:
    if runtime_spec_data is None:
        return {"mode": RuntimeMode.PROBE.value, "device_visibility": {"policy": probe_policy.device_visibility}, "frameworks": {}, "limits": {}, "env": {}, "metadata": {}}
    payload = runtime_spec_data.get("payload", runtime_spec_data)
    if not isinstance(payload, Mapping):
        raise ProviderProtocolError("runtime spec payload must be a mapping")
    data = dict(payload)
    data["mode"] = RuntimeMode.PROBE.value
    visibility = dict(data.get("device_visibility") or {})
    visibility.setdefault("policy", probe_policy.device_visibility)
    data["device_visibility"] = visibility
    data.setdefault("frameworks", {})
    data.setdefault("limits", {})
    data.setdefault("env", {})
    data.setdefault("metadata", {})
    return data


def _run_provider(provider: Any, request: Any) -> ProviderReport:
    if isinstance(request, OperationInspectionRequest):
        report = provider.inspect_operation(request)
    else:
        method_name = {
            "representation_inspection": "inspect_representations",
            "adapter_planning": "plan_adapters",
            "compatibility_check": "check_compatibility",
            "lowering": "lower_operation",
        }.get(request.request_kind)
        report = getattr(provider, method_name)(request)
    if isinstance(report, Mapping):
        report = report_from_data(report)
    if not isinstance(report, ProviderReport):
        raise ProviderProtocolError("provider returned non-report payload", context={"type": type(report).__name__})
    return report


def _with_context_ids(report: ProviderReport, request: Any, runtime_envelope: Mapping[str, Any]) -> ProviderReport:
    environment_spec = getattr(request, "environment_spec", None)
    environment_spec_id = environment_spec.get("id") if isinstance(environment_spec, Mapping) else None
    return replace(
        report,
        operation_id=report.operation_id or getattr(request, "operation_id", None),
        environment_spec_id=report.environment_spec_id or environment_spec_id,
        runtime_id=report.runtime_id or runtime_envelope.get("id"),
    )


def _report_type_for_request(request: Any):
    from .reports import AdapterPlanningReport, CompatibilityCheckReport, LoweringReport, RepresentationInspectionReport

    return {
        "operation_inspection": OperationInspectionReport,
        "representation_inspection": RepresentationInspectionReport,
        "adapter_planning": AdapterPlanningReport,
        "compatibility_check": CompatibilityCheckReport,
        "lowering": LoweringReport,
    }.get(request.request_kind, OperationInspectionReport)


def _protocol_failure(message: str, *, exception: BaseException | None = None) -> ProbeReport:
    return ProbeReport(
        status="failed",
        diagnostics=(ProviderIssue("worker_protocol_error", "error", message, exception_type=type(exception).__name__ if exception is not None else None),),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROVIDER_PROBE_REQUEST_SCHEMA", "PROVIDER_PROBE_RESPONSE_SCHEMA", "handle_json_request", "main", "run_worker_request"]
