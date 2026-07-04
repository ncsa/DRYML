"""Provider reports and aggregate probe reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from dryml.annotations import AnnotationFragment, SourceTrace
from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready

from .errors import ProviderReportError, ProviderValidationError
from .identity import ProviderIdentity


REPORT_SCHEMA_VERSION = 1
PROBE_REPORT_SCHEMA = "dryml.provider_probe_report.v1"
REPORT_STATUSES = frozenset({"ok", "unsupported", "failed", "skipped", "degraded"})


@dataclass(frozen=True, slots=True)
class ProviderIssue:
    """Structured diagnostic entry produced by providers or probe orchestration."""

    code: str
    severity: str
    message: str
    provider: str | None = None
    path: str | None = None
    expected: Any = None
    actual: Any = None
    exception_type: str | None = None
    traceback_summary: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "error"}:
            raise ProviderValidationError("invalid provider issue severity", context={"severity": self.severity})
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, "issue.metadata"))

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready issue data."""

        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "provider": self.provider,
            "path": self.path,
            "expected": json_ready(self.expected),
            "actual": json_ready(self.actual),
            "exception_type": self.exception_type,
            "traceback_summary": self.traceback_summary,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProviderIssue":
        """Build an issue from JSON-ready data."""

        if not isinstance(data, Mapping):
            raise ProviderReportError("provider issue must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"code", "severity", "message", "provider", "path", "expected", "actual", "exception_type", "traceback_summary", "metadata"}
        if unknown:
            raise ProviderReportError("provider issue has unknown fields", context={"fields": sorted(unknown)})
        return cls(
            code=str(data.get("code")),
            severity=str(data.get("severity")),
            message=str(data.get("message")),
            provider=data.get("provider"),
            path=data.get("path"),
            expected=data.get("expected"),
            actual=data.get("actual"),
            exception_type=data.get("exception_type"),
            traceback_summary=data.get("traceback_summary"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True, slots=True)
class ProviderReport:
    """Base provider report with common JSON-ready metadata."""

    report_kind: ClassVar[str] = "provider_report"
    provider_identity: ProviderIdentity = field(default_factory=lambda: ProviderIdentity("unknown"))
    status: str = "unsupported"
    request_key: str | None = None
    operation_id: str | None = None
    environment_id: str | None = None
    environment_spec_id: str | None = None
    runtime_id: str | None = None
    fragments: tuple[AnnotationFragment, ...] = ()
    issues: tuple[ProviderIssue, ...] = ()
    stdout: str | None = None
    stderr: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in REPORT_STATUSES:
            raise ProviderReportError("unknown provider report status", context={"status": self.status})
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ProviderReportError("unsupported provider report schema version", context={"schema_version": self.schema_version})
        object.__setattr__(self, "fragments", tuple(_coerce_fragment(fragment) for fragment in self.fragments))
        object.__setattr__(self, "issues", tuple(_coerce_issue(issue) for issue in self.issues))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, "report.metadata"))

    @classmethod
    def unsupported(cls, provider_identity: ProviderIdentity, request: Any, *, message: str | None = None) -> "ProviderReport":
        """Return a structured unsupported report for an optional provider method."""

        return cls(
            provider_identity=provider_identity,
            status="unsupported",
            request_key=getattr(request, "key", getattr(request, "request_key", None)),
            operation_id=getattr(request, "operation_id", None),
            issues=(ProviderIssue("unsupported", "info", message or f"{cls.report_kind} is not supported", provider=provider_identity.name),),
        )

    @classmethod
    def failed(cls, provider_identity: ProviderIdentity, request: Any | None, message: str, *, code: str = "provider_failed", exception: BaseException | None = None, metadata: Mapping[str, Any] | None = None) -> "ProviderReport":
        """Return a structured failed report."""

        issue = ProviderIssue(
            code,
            "error",
            message,
            provider=provider_identity.name,
            exception_type=type(exception).__name__ if exception is not None else None,
            metadata=metadata or {},
        )
        return cls(
            provider_identity=provider_identity,
            status="failed",
            request_key=getattr(request, "key", getattr(request, "request_key", None)) if request is not None else None,
            operation_id=getattr(request, "operation_id", None) if request is not None else None,
            issues=(issue,),
        )

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready report data."""

        return {
            "schema_version": self.schema_version,
            "report_kind": self.report_kind,
            "status": self.status,
            "provider_identity": self.provider_identity.to_data(),
            "request_key": self.request_key,
            "operation_id": self.operation_id,
            "environment_id": self.environment_id,
            "environment_spec_id": self.environment_spec_id,
            "runtime_id": self.runtime_id,
            "annotation_fragments": [fragment.to_data() for fragment in self.fragments],
            "issues": [issue.to_data() for issue in self.issues],
            "stdout": self.stdout,
            "stderr": self.stderr,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProviderReport":
        """Build this report type from JSON-ready data."""

        _validate_report_data(data, cls.report_kind)
        return cls(**_report_kwargs(data))

    def annotation_fragments(self, *, source_kind: str = "provider", probe_report_id: str | None = None) -> tuple[AnnotationFragment, ...]:
        """Return fragments with provider or cached-probe source traces."""

        return tuple(_with_provider_source(fragment, self, source_kind=source_kind, probe_report_id=probe_report_id) for fragment in self.fragments)


@dataclass(frozen=True, slots=True)
class OperationInspectionReport(ProviderReport):
    """Provider report for operation inspection."""

    report_kind: ClassVar[str] = "operation_inspection"


@dataclass(frozen=True, slots=True)
class RepresentationInspectionReport(ProviderReport):
    """Structural report placeholder for representation inspection."""

    report_kind: ClassVar[str] = "representation_inspection"


@dataclass(frozen=True, slots=True)
class AdapterPlanningReport(ProviderReport):
    """Structural report placeholder for adapter planning."""

    report_kind: ClassVar[str] = "adapter_planning"


@dataclass(frozen=True, slots=True)
class CompatibilityCheckReport(ProviderReport):
    """Structural report placeholder for compatibility checks."""

    report_kind: ClassVar[str] = "compatibility_check"


@dataclass(frozen=True, slots=True)
class LoweringReport(ProviderReport):
    """Structural report placeholder for lowering support."""

    report_kind: ClassVar[str] = "lowering"


REPORT_TYPES = {
    OperationInspectionReport.report_kind: OperationInspectionReport,
    RepresentationInspectionReport.report_kind: RepresentationInspectionReport,
    AdapterPlanningReport.report_kind: AdapterPlanningReport,
    CompatibilityCheckReport.report_kind: CompatibilityCheckReport,
    LoweringReport.report_kind: LoweringReport,
}


@dataclass(frozen=True, slots=True)
class ProbeReport:
    """Aggregate report returned by a provider probe worker or runner."""

    request: Mapping[str, Any] | None = None
    reports: tuple[ProviderReport, ...] = ()
    operation_id: str | None = None
    environment_spec: Mapping[str, Any] | None = None
    environment_spec_id: str | None = None
    environment_record_id: str | None = None
    runtime_spec: Mapping[str, Any] | None = None
    runtime_id: str | None = None
    probe_policy: Mapping[str, Any] = field(default_factory=dict)
    status: str = "ok"
    diagnostics: tuple[ProviderIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = PROBE_REPORT_SCHEMA
    schema_version: int = REPORT_SCHEMA_VERSION
    report_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in REPORT_STATUSES:
            raise ProviderReportError("unknown probe report status", context={"status": self.status})
        if self.schema != PROBE_REPORT_SCHEMA or self.schema_version != REPORT_SCHEMA_VERSION:
            raise ProviderReportError("unsupported probe report schema", context={"schema": self.schema, "schema_version": self.schema_version})
        object.__setattr__(self, "request", None if self.request is None else _freeze_json_mapping(self.request, "probe_report.request"))
        object.__setattr__(self, "environment_spec", None if self.environment_spec is None else _freeze_json_mapping(self.environment_spec, "probe_report.environment_spec"))
        object.__setattr__(self, "runtime_spec", None if self.runtime_spec is None else _freeze_json_mapping(self.runtime_spec, "probe_report.runtime_spec"))
        object.__setattr__(self, "probe_policy", _freeze_json_mapping(self.probe_policy, "probe_report.probe_policy"))
        object.__setattr__(self, "reports", tuple(report_from_data(report) if isinstance(report, Mapping) else report for report in self.reports))
        object.__setattr__(self, "diagnostics", tuple(_coerce_issue(issue) for issue in self.diagnostics))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, "probe_report.metadata"))

    @property
    def ok(self) -> bool:
        """Return whether the aggregate report succeeded without errors."""

        return self.status == "ok" and not any(issue.severity == "error" for issue in self.diagnostics)

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready aggregate report payload data."""

        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "request": None if self.request is None else json_ready(self.request),
            "operation_id": self.operation_id,
            "environment_spec": None if self.environment_spec is None else json_ready(self.environment_spec),
            "environment_spec_id": self.environment_spec_id,
            "environment_record_id": self.environment_record_id,
            "runtime_spec": None if self.runtime_spec is None else json_ready(self.runtime_spec),
            "runtime_id": self.runtime_id,
            "probe_policy": json_ready(self.probe_policy),
            "reports": [report.to_data() for report in self.reports],
            "annotation_fragments": [fragment.to_data() for fragment in self.annotation_fragments(cached=False)],
            "status": self.status,
            "diagnostics": [issue.to_data() for issue in self.diagnostics],
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProbeReport":
        """Build an aggregate report from JSON-ready payload data."""

        if not isinstance(data, Mapping):
            raise ProviderReportError("probe report must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"schema", "schema_version", "request", "operation_id", "environment_spec", "environment_spec_id", "environment_record_id", "runtime_spec", "runtime_id", "probe_policy", "reports", "annotation_fragments", "status", "diagnostics", "metadata", "report_id"}
        if unknown:
            raise ProviderReportError("probe report has unknown fields", context={"fields": sorted(unknown)})
        return cls(
            request=data.get("request"),
            reports=tuple(report_from_data(report) for report in data.get("reports") or ()),
            operation_id=data.get("operation_id"),
            environment_spec=data.get("environment_spec"),
            environment_spec_id=data.get("environment_spec_id"),
            environment_record_id=data.get("environment_record_id"),
            runtime_spec=data.get("runtime_spec"),
            runtime_id=data.get("runtime_id"),
            probe_policy=data.get("probe_policy") or {},
            status=data.get("status", "ok"),
            diagnostics=tuple(ProviderIssue.from_data(issue) for issue in data.get("diagnostics") or ()),
            metadata=data.get("metadata") or {},
            schema=data.get("schema", PROBE_REPORT_SCHEMA),
            schema_version=data.get("schema_version", REPORT_SCHEMA_VERSION),
            report_id=data.get("report_id"),
        )

    def annotation_fragments(self, *, cached: bool = False, report_id: str | None = None) -> tuple[AnnotationFragment, ...]:
        """Return merged provider fragments for annotation resolution."""

        source_kind = "cached_probe" if cached else "provider"
        probe_report_id = report_id or self.report_id
        fragments: list[AnnotationFragment] = []
        for report in self.reports:
            fragments.extend(report.annotation_fragments(source_kind=source_kind, probe_report_id=probe_report_id))
        return tuple(fragments)


def report_from_data(data: Mapping[str, Any]) -> ProviderReport:
    """Deserialize a concrete provider report from JSON-ready data."""

    if not isinstance(data, Mapping):
        raise ProviderReportError("provider report must be a mapping", context={"type": type(data).__name__})
    report_kind = data.get("report_kind")
    try:
        report_type = REPORT_TYPES[report_kind]
    except KeyError as exc:
        raise ProviderReportError("unknown provider report kind", context={"report_kind": report_kind}) from exc
    return report_type.from_data(data)


def as_provider_fragments(report: ProbeReport, *, cached: bool = False) -> tuple[AnnotationFragment, ...]:
    """Return provider fragments from a probe report."""

    return report.annotation_fragments(cached=cached)


def _report_kwargs(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provider_identity": ProviderIdentity.from_data(data.get("provider_identity") or {}),
        "status": data.get("status", "unsupported"),
        "request_key": data.get("request_key"),
        "operation_id": data.get("operation_id"),
        "environment_id": data.get("environment_id"),
        "environment_spec_id": data.get("environment_spec_id"),
        "runtime_id": data.get("runtime_id"),
        "fragments": tuple(AnnotationFragment.from_data(fragment) for fragment in data.get("annotation_fragments") or ()),
        "issues": tuple(ProviderIssue.from_data(issue) for issue in data.get("issues") or ()),
        "stdout": data.get("stdout"),
        "stderr": data.get("stderr"),
        "metadata": data.get("metadata") or {},
        "schema_version": data.get("schema_version", REPORT_SCHEMA_VERSION),
    }


def _validate_report_data(data: Mapping[str, Any], report_kind: str) -> None:
    if not isinstance(data, Mapping):
        raise ProviderReportError("provider report must be a mapping", context={"type": type(data).__name__})
    allowed = {"schema_version", "report_kind", "status", "provider_identity", "request_key", "operation_id", "environment_id", "environment_spec_id", "runtime_id", "annotation_fragments", "issues", "stdout", "stderr", "metadata"}
    unknown = set(data) - allowed
    if unknown:
        raise ProviderReportError("provider report has unknown fields", context={"fields": sorted(unknown)})
    if data.get("report_kind") != report_kind:
        raise ProviderReportError("provider report kind mismatch", context={"expected": report_kind, "observed": data.get("report_kind")})


def _with_provider_source(fragment: AnnotationFragment, report: ProviderReport, *, source_kind: str, probe_report_id: str | None) -> AnnotationFragment:
    label = report.provider_identity.name if report.provider_identity.version is None else f"{report.provider_identity.name}/{report.provider_identity.version}"
    source_metadata = dict(json_ready(fragment.source.metadata))
    source_metadata.update(
        {
            "provider_id": report.provider_identity.id,
            "provider_name": report.provider_identity.name,
            "provider_version": report.provider_identity.version,
            "operation_id": report.operation_id,
            "environment_id": report.environment_id,
            "environment_spec_id": report.environment_spec_id,
            "runtime_id": report.runtime_id,
            "request_key": report.request_key,
            "request_kind": report.report_kind,
            "probe_report_id": probe_report_id,
        }
    )
    source = SourceTrace(kind=source_kind, label=label, namespace=fragment.namespace, metadata=source_metadata)
    return AnnotationFragment(fragment.namespace, fragment.kind, fragment.fragment, source, fragment.priority, fragment.merge_policy, fragment.schema_version)


def _coerce_fragment(value: AnnotationFragment | Mapping[str, Any]) -> AnnotationFragment:
    return value if isinstance(value, AnnotationFragment) else AnnotationFragment.from_data(value)


def _coerce_issue(value: ProviderIssue | Mapping[str, Any]) -> ProviderIssue:
    return value if isinstance(value, ProviderIssue) else ProviderIssue.from_data(value)


def _freeze_json_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderReportError("provider report payload must be a mapping", context={"path": path, "type": type(value).__name__})
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise ProviderReportError("provider report payload is not JSON-ready", context={"path": path, **exc.context}) from exc
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = [
    "AdapterPlanningReport",
    "CompatibilityCheckReport",
    "LoweringReport",
    "OperationInspectionReport",
    "PROBE_REPORT_SCHEMA",
    "ProbeReport",
    "ProviderIssue",
    "ProviderReport",
    "RepresentationInspectionReport",
    "as_provider_fragments",
    "report_from_data",
]
