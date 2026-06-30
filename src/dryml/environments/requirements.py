"""Environment requirements and compatibility checking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from .compatibility import CompatibilityIssue, CompatibilityReport, coerce_policy, report_from_issues
from .records import EnvironmentRecord
from .schema import ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION
from .serialization import freeze_mapping
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
        object.__setattr__(self, "schema_versions", freeze_mapping(self.schema_versions))
        object.__setattr__(self, "details", freeze_mapping(self.details))
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
        marker_env = default_environment()
        marker_env["python_version"] = ".".join(record.python.version.split(".")[:2])
        marker_env["python_full_version"] = record.python.version
        for req_text in self.requirements:
            req = Requirement(req_text)
            if req.marker is not None and not req.marker.evaluate(marker_env):
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
                observed = Version(str(record.dryml.execution_protocol))
                if specifier and observed not in specifier:
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
            if specifier and Version(str(observed)) not in specifier:
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
            "schema_versions": dict(self.schema_versions),
            "details": dict(self.details),
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


__all__ = ["EnvironmentRequirement"]
