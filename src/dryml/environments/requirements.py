"""Environment requirements and compatibility checking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from .compatibility import CompatibilityIssue, CompatibilityReport, coerce_policy, report_from_issues
from .records import EnvironmentRecord
from .schema import ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION
from .serialization import deep_freeze_json, json_ready
from .ids import content_id
from .utils import (
    coerce_specifier,
    coerce_tuple,
    normalize_distribution_name,
    normalize_requirement_string,
    requirement_sort_key,
)


@dataclass(frozen=True, slots=True)
class EnvironmentRequirement:
    """Portable software constraints for an acceptable Python environment."""

    python: str | None = None
    requirements: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    dryml_protocol: str | None = None
    schema_versions: Mapping[str, str] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized_requirements = tuple(
            sorted(
                (normalize_requirement_string(req) for req in coerce_tuple(self.requirements)),
                key=requirement_sort_key,
            )
        )
        object.__setattr__(self, "requirements", normalized_requirements)
        object.__setattr__(
            self,
            "excludes",
            tuple(sorted(normalize_distribution_name(item) for item in coerce_tuple(self.excludes))),
        )
        object.__setattr__(
            self,
            "capabilities",
            tuple(sorted(str(item) for item in coerce_tuple(self.capabilities))),
        )
        object.__setattr__(self, "tags", tuple(sorted(str(item) for item in coerce_tuple(self.tags))))
        object.__setattr__(self, "schema_versions", deep_freeze_json(self.schema_versions))
        object.__setattr__(self, "details", deep_freeze_json(self.details))
        coerce_specifier(self.python)
        if self.dryml_protocol:
            coerce_specifier(self.dryml_protocol)
        for spec in self.schema_versions.values():
            coerce_specifier(spec)

    @property
    def id(self) -> str:
        """Stable content-addressed ID for this requirement."""

        return content_id("envreq", self.schema_version, self.to_data())

    def check(self, record: EnvironmentRecord, *, policy: str = "compatible") -> CompatibilityReport:
        """Check this requirement against an observed environment record.

        Policy is applied only when converting collected issues into the final
        report status. The raw issue list remains machine-readable.
        """

        coerced_policy = coerce_policy(policy)
        if coerced_policy == "ignore":
            return report_from_issues((), policy=coerced_policy)

        issues: list[CompatibilityIssue] = []
        issues.extend(self._check_python(record))
        issues.extend(self._check_packages(record))
        issues.extend(self._check_excludes(record))
        issues.extend(self._check_capabilities(record))
        issues.extend(self._check_dryml(record, strict=coerced_policy == "strict"))
        issues.extend(self._check_tags(record, strict=coerced_policy == "strict"))
        return report_from_issues(tuple(issues), policy=coerced_policy)

    def explain_sources(self) -> str:
        """Return fragment/source information when this requirement was composed."""

        sources = self.details.get("sources", ())
        if not sources:
            return "Environment requirement has no recorded fragment sources."
        return "Environment requirement sources:\n" + "\n".join(f"- {source}" for source in sources)

    def _check_python(self, record: EnvironmentRecord) -> tuple[CompatibilityIssue, ...]:
        if not self.python:
            return ()
        specifier = coerce_specifier(self.python)
        observed = record.python.version
        if not observed:
            return (CompatibilityIssue("python_missing", "error", "Python version is missing"),)
        try:
            version = Version(observed)
        except InvalidVersion:
            return (
                CompatibilityIssue(
                    "python_missing",
                    "unknown",
                    f"Python version {observed!r} could not be parsed",
                    expected=self.python,
                    observed=observed,
                ),
            )
        if specifier and version not in specifier:
            return (
                CompatibilityIssue(
                    "python_version_mismatch",
                    "error",
                    f"Python {observed} does not satisfy {self.python}",
                    requirement_path="python",
                    observed_path="python.version",
                    expected=self.python,
                    observed=observed,
                ),
            )
        return ()

    def _check_packages(self, record: EnvironmentRecord) -> tuple[CompatibilityIssue, ...]:
        issues: list[CompatibilityIssue] = []
        marker_env = marker_environment_from_record(record)
        for req_text in self.requirements:
            req = Requirement(req_text)
            if req.marker is not None:
                unknowns = _unknown_marker_variables(req.marker, marker_env)
                if unknowns:
                    issues.append(
                        CompatibilityIssue(
                            "marker_environment_unknown",
                            "unknown",
                            f"environment marker for {req_text!r} references unknown fields: {', '.join(unknowns)}",
                            requirement_path=f"requirements.{normalize_distribution_name(req.name)}.marker",
                            expected=str(req.marker),
                            observed={name: marker_env.get(name) for name in unknowns},
                        )
                    )
                    continue
                if not req.marker.evaluate({key: "" if value is None else value for key, value in marker_env.items()}):
                    continue
            name = normalize_distribution_name(req.name)
            package = record.distributions.get(name)
            if package is None:
                issues.append(
                    CompatibilityIssue(
                        "package_missing",
                        "error",
                        f"required distribution {req_text!r} is not installed",
                        requirement_path=f"requirements.{name}",
                        observed_path=f"distributions.{name}",
                        expected=req_text,
                    )
                )
                continue
            if str(req.specifier):
                if not package.version:
                    issues.append(
                        CompatibilityIssue(
                            "package_version_unknown",
                            "unknown",
                            f"installed distribution {name!r} has no known version",
                            requirement_path=f"requirements.{name}",
                            observed_path=f"distributions.{name}.version",
                            expected=str(req.specifier),
                            observed=package.version,
                        )
                    )
                    continue
                try:
                    version = Version(package.version)
                except InvalidVersion:
                    issues.append(
                        CompatibilityIssue(
                            "package_version_unknown",
                            "unknown",
                            f"installed distribution {name!r} version {package.version!r} could not be parsed",
                            requirement_path=f"requirements.{name}",
                            observed_path=f"distributions.{name}.version",
                            expected=str(req.specifier),
                            observed=package.version,
                        )
                    )
                    continue
                if version not in req.specifier:
                    issues.append(
                        CompatibilityIssue(
                            "package_version_mismatch",
                            "error",
                            f"distribution {name!r} version {package.version} does not satisfy {req.specifier}",
                            requirement_path=f"requirements.{name}",
                            observed_path=f"distributions.{name}.version",
                            expected=str(req.specifier),
                            observed=package.version,
                        )
                    )
        return tuple(issues)

    def _check_excludes(self, record: EnvironmentRecord) -> tuple[CompatibilityIssue, ...]:
        issues = []
        for name in self.excludes:
            package = record.distributions.get(name)
            if package is not None:
                issues.append(
                    CompatibilityIssue(
                        "package_excluded_present",
                        "error",
                        f"excluded distribution {name!r} is installed as {package.version}",
                        requirement_path=f"excludes.{name}",
                        observed_path=f"distributions.{name}",
                        expected="absent",
                        observed=package.version,
                    )
                )
        return tuple(issues)

    def _check_capabilities(self, record: EnvironmentRecord) -> tuple[CompatibilityIssue, ...]:
        features = set(record.dryml.features if record.dryml is not None else ())
        return tuple(
            CompatibilityIssue(
                "capability_missing",
                "error",
                f"required capability {capability!r} is not available",
                requirement_path=f"capabilities.{capability}",
                observed_path="dryml.features",
                expected=capability,
            )
            for capability in self.capabilities
            if capability not in features
        )

    def _check_dryml(self, record: EnvironmentRecord, *, strict: bool) -> tuple[CompatibilityIssue, ...]:
        issues: list[CompatibilityIssue] = []
        if record.dryml is None:
            if strict or self.dryml_protocol or self.schema_versions:
                issues.append(
                    CompatibilityIssue(
                        "dryml_runtime_missing",
                        "error" if strict else "unknown",
                        "DRYML runtime metadata is missing",
                        observed_path="dryml",
                    )
                )
            return tuple(issues)
        if self.dryml_protocol:
            if record.dryml.execution_protocol is None:
                issues.append(
                    CompatibilityIssue(
                        "dryml_protocol_mismatch",
                        "error" if strict else "unknown",
                        "DRYML execution protocol metadata is missing",
                        requirement_path="dryml_protocol",
                        observed_path="dryml.execution_protocol",
                        expected=self.dryml_protocol,
                    )
                )
            else:
                specifier = coerce_specifier(self.dryml_protocol)
                try:
                    observed = Version(str(record.dryml.execution_protocol))
                except InvalidVersion:
                    issues.append(
                        CompatibilityIssue(
                            "dryml_protocol_mismatch",
                            "unknown",
                            f"DRYML protocol {record.dryml.execution_protocol!r} could not be parsed",
                            requirement_path="dryml_protocol",
                            observed_path="dryml.execution_protocol",
                            expected=self.dryml_protocol,
                            observed=record.dryml.execution_protocol,
                        )
                    )
                    observed = None
                if observed is not None and specifier and observed not in specifier:
                    issues.append(
                        CompatibilityIssue(
                            "dryml_protocol_mismatch",
                            "error",
                            f"DRYML protocol {observed} does not satisfy {self.dryml_protocol}",
                            requirement_path="dryml_protocol",
                            observed_path="dryml.execution_protocol",
                            expected=self.dryml_protocol,
                            observed=str(observed),
                        )
                    )
        for schema_name, schema_spec in self.schema_versions.items():
            if schema_name not in record.dryml.schema_versions:
                issues.append(
                    CompatibilityIssue(
                        "schema_missing",
                        "error",
                        f"schema version {schema_name!r} is missing",
                        requirement_path=f"schema_versions.{schema_name}",
                        observed_path=f"dryml.schema_versions.{schema_name}",
                        expected=schema_spec,
                    )
                )
                continue
            observed = record.dryml.schema_versions[schema_name]
            specifier = coerce_specifier(schema_spec)
            try:
                observed_version = Version(str(observed))
            except InvalidVersion:
                issues.append(
                    CompatibilityIssue(
                        "schema_version_mismatch",
                        "unknown",
                        f"schema {schema_name!r} version {observed!r} could not be parsed",
                        requirement_path=f"schema_versions.{schema_name}",
                        observed_path=f"dryml.schema_versions.{schema_name}",
                        expected=schema_spec,
                        observed=observed,
                    )
                )
                continue
            if specifier and observed_version not in specifier:
                issues.append(
                    CompatibilityIssue(
                        "schema_version_mismatch",
                        "error",
                        f"schema {schema_name!r} version {observed} does not satisfy {schema_spec}",
                        requirement_path=f"schema_versions.{schema_name}",
                        observed_path=f"dryml.schema_versions.{schema_name}",
                        expected=schema_spec,
                        observed=observed,
                    )
                )
        return tuple(issues)

    def _check_tags(self, record: EnvironmentRecord, *, strict: bool) -> tuple[CompatibilityIssue, ...]:
        tags = set(record.tags)
        return tuple(
            CompatibilityIssue(
                "tag_missing",
                "error" if strict else "warning",
                f"environment tag {tag!r} is not present",
                requirement_path=f"tags.{tag}",
                observed_path="tags",
                expected=tag,
            )
            for tag in self.tags
            if tag not in tags
        )

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible requirement data."""

        return {
            "schema_version": self.schema_version,
            "python": self.python,
            "requirements": list(self.requirements),
            "excludes": list(self.excludes),
            "capabilities": list(self.capabilities),
            "tags": list(self.tags),
            "dryml_protocol": self.dryml_protocol,
            "schema_versions": json_ready(self.schema_versions),
            "details": json_ready(self.details),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "EnvironmentRequirement":
        """Build a requirement from serialized data."""

        return cls(
            python=data.get("python"),
            requirements=tuple(data.get("requirements", ())),
            excludes=tuple(data.get("excludes", ())),
            capabilities=tuple(data.get("capabilities", ())),
            tags=tuple(data.get("tags", ())),
            dryml_protocol=data.get("dryml_protocol"),
            schema_versions=data.get("schema_versions", {}),
            details=data.get("details", {}),
            schema_version=data.get("schema_version", ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION),
        )


def marker_environment_from_record(record: EnvironmentRecord) -> dict[str, str | None]:
    """Derive PEP 508 marker variables from an EnvironmentRecord.

    Values come from the record and conservative derivations from record fields;
    the orchestrator process platform is never used for remote marker checks.
    """

    system = record.platform.system or None
    implementation = record.python.implementation or None
    return {
        "implementation_name": record.platform.implementation_name or _implementation_name(implementation),
        "implementation_version": record.platform.implementation_version or record.python.version,
        "os_name": record.platform.os_name or _os_name_from_system(system),
        "platform_machine": record.platform.machine or None,
        "platform_release": record.platform.release or None,
        "platform_system": system,
        "platform_version": record.platform.version or None,
        "platform_python_implementation": record.platform.platform_python_implementation or implementation,
        "python_full_version": record.python.version or None,
        "python_version": _python_major_minor(record.python.version),
        "sys_platform": record.platform.sys_platform or _sys_platform_from_system(system),
        "extra": "",
    }


def _python_major_minor(version: str | None) -> str | None:
    if not version:
        return None
    parts = str(version).split(".")
    if len(parts) < 2:
        return None
    return ".".join(parts[:2])


def _implementation_name(value: str | None) -> str | None:
    if not value:
        return None
    return str(value).lower()


def _os_name_from_system(system: str | None) -> str | None:
    if not system:
        return None
    key = system.lower()
    if key.startswith(("linux", "darwin", "freebsd", "openbsd", "netbsd")):
        return "posix"
    if key.startswith("windows"):
        return "nt"
    return None


def _sys_platform_from_system(system: str | None) -> str | None:
    if not system:
        return None
    key = system.lower()
    if key.startswith("linux"):
        return "linux"
    if key.startswith("darwin"):
        return "darwin"
    if key.startswith("windows"):
        return "win32"
    return None


def _unknown_marker_variables(marker: Any, marker_env: Mapping[str, str | None]) -> tuple[str, ...]:
    return tuple(sorted(name for name in _marker_variables(marker) if marker_env.get(name) is None))


def _marker_variables(marker: Any) -> set[str]:
    variables: set[str] = set()

    def visit(value: Any) -> None:
        if value.__class__.__name__ == "Variable":
            variables.add(str(getattr(value, "value", value)))
            return
        if isinstance(value, list | tuple):
            for item in value:
                visit(item)

    visit(getattr(marker, "_markers", ()))
    return variables


__all__ = ["EnvironmentRequirement", "marker_environment_from_record"]
