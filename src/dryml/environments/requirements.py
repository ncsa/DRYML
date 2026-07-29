"""Environment requirements and compatibility checking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .compatibility import CompatibilityIssue, CompatibilityReport, coerce_policy, report_from_issues
from .errors import EnvironmentRequirementError
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

    def merge(self, other: "EnvironmentRequirement", *, sources: tuple[str, ...] = ()) -> "EnvironmentRequirement":
        """Return the semantic intersection of two compatible requirements.

        Package constraints with the same PEP 508 marker, extras, and
        direct-reference identity are intersected. Requirements with different
        markers remain separate, but contradictory constraints are rejected
        unless their markers are proven disjoint.
        """

        if not isinstance(other, EnvironmentRequirement):
            raise EnvironmentRequirementError("environment requirement merge requires an EnvironmentRequirement")
        details_sources = tuple(self.details.get("sources", ())) + tuple(other.details.get("sources", ())) + tuple(sources)
        requirements = _merge_package_requirements(self.requirements, other.requirements)
        excludes = tuple(sorted(set(self.excludes) | set(other.excludes)))
        required_names = {normalize_distribution_name(Requirement(item).name) for item in requirements}
        overlap = required_names & set(excludes)
        if overlap:
            raise EnvironmentRequirementError(
                "required distributions cannot also be excluded",
                context={"path": f"requirements.{sorted(overlap)[0]}"},
            )
        return EnvironmentRequirement(
            python=_intersect_specifiers(self.python, other.python, path="python"),
            requirements=requirements,
            excludes=excludes,
            capabilities=tuple(sorted(set(self.capabilities) | set(other.capabilities))),
            tags=tuple(sorted(set(self.tags) | set(other.tags))),
            dryml_protocol=_intersect_specifiers(self.dryml_protocol, other.dryml_protocol, path="dryml_protocol"),
            schema_versions=_merge_schema_versions(self.schema_versions, other.schema_versions),
            details={"sources": details_sources} if details_sources else {},
            schema_version=max(self.schema_version, other.schema_version),
        )

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


def _intersect_specifiers(left: str | None, right: str | None, *, path: str) -> str | None:
    """Return a canonical satisfiable specifier intersection."""

    if left in (None, ""):
        return right or None
    if right in (None, ""):
        return left or None
    combined = SpecifierSet(f"{left},{right}")
    if _specifier_has_obvious_conflict(combined):
        raise EnvironmentRequirementError("conflicting environment version constraints", context={"path": path, "left": left, "right": right})
    return str(combined) or None


def _merge_package_requirements(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    grouped: dict[tuple[str, tuple[str, ...], str | None, str | None], list[Requirement]] = {}
    for text in left + right:
        req = Requirement(text)
        key = (normalize_distribution_name(req.name), tuple(sorted(req.extras)), req.url, None if req.marker is None else str(req.marker))
        grouped.setdefault(key, []).append(req)
    merged: list[str] = []
    combined_groups: list[tuple[tuple[str, tuple[str, ...], str | None], str | None, SpecifierSet]] = []
    for (name, extras, url, marker), requirements in grouped.items():
        combined = SpecifierSet(",".join(str(item.specifier) for item in requirements if str(item.specifier)))
        if _specifier_has_obvious_conflict(combined):
            raise EnvironmentRequirementError(
                "conflicting package requirement specifiers",
                context={"path": f"requirements.{name}"},
            )
        identity = (name, extras, url)
        for other_identity, other_marker, other_specifier in combined_groups:
            if other_identity != identity or _markers_proven_disjoint(marker, other_marker):
                continue
            overlap = SpecifierSet(",".join(filter(None, (str(combined), str(other_specifier)))))
            if _specifier_has_obvious_conflict(overlap):
                raise EnvironmentRequirementError(
                    "conflicting package requirement specifiers under overlapping markers",
                    context={"path": f"requirements.{name}"},
                )
        combined_groups.append((identity, marker, combined))
        specifier = str(combined) or None
        text = name + ("[" + ",".join(extras) + "]" if extras else "")
        if url:
            text += f" @ {url}"
        if specifier:
            text += specifier
        if marker:
            text += f"; {marker}"
        merged.append(normalize_requirement_string(text))
    return tuple(sorted(set(merged), key=requirement_sort_key))


def _markers_proven_disjoint(left: str | None, right: str | None) -> bool:
    """Return whether supported conjunctive marker constraints cannot overlap."""

    if left is None or right is None:
        return False
    left_atoms = _conjunctive_marker_atoms(Requirement(f"x; {left}").marker)
    right_atoms = _conjunctive_marker_atoms(Requirement(f"x; {right}").marker)
    if left_atoms is None or right_atoms is None:
        return False
    return _marker_atoms_have_conflict(left_atoms + right_atoms)


def _conjunctive_marker_atoms(marker: Any) -> list[tuple[Any, Any, Any]] | None:
    """Return marker atoms when the expression contains conjunctions only."""

    atoms: list[tuple[Any, Any, Any]] = []

    def visit(value: Any) -> bool:
        if isinstance(value, tuple) and len(value) == 3:
            atoms.append(value)
            return True
        if not isinstance(value, list):
            return value == "and"
        for item in value:
            if item == "or" or not visit(item):
                return False
        return True

    return atoms if visit(getattr(marker, "_markers", ())) else None


def _marker_atoms_have_conflict(atoms: list[tuple[Any, Any, Any]]) -> bool:
    """Detect contradictions in the marker atom forms DRYML can prove."""

    version_variables = {"implementation_version", "python_full_version", "python_version"}
    version_operators = {"<", "<=", ">", ">=", "==", "!=", "~=", "==="}
    reverse_operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!=", "===": "==="}
    version_specifiers: dict[str, list[str]] = {}
    string_equalities: dict[str, set[str]] = {}
    string_exclusions: dict[str, set[str]] = {}
    for left, operator, right in atoms:
        if left.__class__.__name__ == "Variable" and right.__class__.__name__ == "Value":
            variable, value, op = str(left.value), str(right.value), str(operator)
        elif right.__class__.__name__ == "Variable" and left.__class__.__name__ == "Value":
            variable, value = str(right.value), str(left.value)
            op = reverse_operator.get(str(operator), "")
        else:
            continue
        if variable in version_variables and op in version_operators:
            version_specifiers.setdefault(variable, []).append(f"{op}{value}")
        elif op == "==":
            string_equalities.setdefault(variable, set()).add(value)
        elif op == "!=":
            string_exclusions.setdefault(variable, set()).add(value)

    for values in version_specifiers.values():
        try:
            combined = SpecifierSet(",".join(values))
        except InvalidSpecifier:
            continue
        if _specifier_has_obvious_conflict(combined):
            return True
    for variable, values in string_equalities.items():
        if len(values) > 1 or values & string_exclusions.get(variable, set()):
            return True
    return False


def _merge_schema_versions(left: Mapping[str, str], right: Mapping[str, str]) -> dict[str, str]:
    result = dict(left)
    for name, specifier in right.items():
        result[name] = _intersect_specifiers(result.get(name), specifier, path=f"schema_versions.{name}") or ""
    return result


def _specifier_has_obvious_conflict(specifier: SpecifierSet) -> bool:
    arbitrary_exact_versions = {item.version for item in specifier if item.operator == "==="}
    if len(arbitrary_exact_versions) > 1:
        return True
    # SpecifierSet preserves PEP 440 syntax but does not expose whether its
    # intersection is empty. Test a bounded set of interval/wildcard boundary
    # witnesses before falling back to the simple range checks below.
    candidates = _specifier_witnesses(specifier)
    if candidates and not any(candidate in specifier for candidate in candidates):
        return True
    exact_versions: set[Version] = set()
    lower: tuple[Version, bool] | None = None
    upper: tuple[Version, bool] | None = None
    for item in specifier:
        try:
            version = Version(item.version)
        except InvalidVersion:
            return False
        if item.operator == "==":
            exact_versions.add(version)
        elif item.operator in {">", ">="}:
            inclusive = item.operator == ">="
            if lower is None or version > lower[0] or (version == lower[0] and not inclusive and lower[1]):
                lower = (version, inclusive)
        elif item.operator in {"<", "<="}:
            inclusive = item.operator == "<="
            if upper is None or version < upper[0] or (version == upper[0] and not inclusive and upper[1]):
                upper = (version, inclusive)
    if len(exact_versions) > 1:
        return True
    if exact_versions:
        return next(iter(exact_versions)) not in specifier
    return lower is not None and upper is not None and (lower[0] > upper[0] or (lower[0] == upper[0] and (not lower[1] or not upper[1])))


def _specifier_witnesses(specifier: SpecifierSet) -> tuple[Version, ...]:
    """Return bounded PEP 440 candidates around every declared boundary."""

    candidates: set[Version] = set()
    for item in specifier:
        raw = item.version.rstrip(".*")
        try:
            version = Version(raw)
        except InvalidVersion:
            return ()
        release = version.release or (0,)
        candidates.update((version, Version(".".join(str(part) for part in (*release, 0))), Version(".".join(str(part) for part in (*release, 1)))))
        # Strict adjacent bounds can admit a post release between two public
        # releases, so do not mistake that valid PEP 440 interval for empty.
        try:
            candidates.add(Version(f"{version}.post1"))
        except InvalidVersion:
            pass
        if release[-1] > 0:
            before = (*release[:-1], release[-1] - 1, 999999)
            candidates.add(Version(".".join(str(part) for part in before)))
        if item.operator == "~=":
            prefix = release[:-1] if len(release) > 1 else ()
            upper = (prefix[-1] + 1,) if len(prefix) == 1 else (*prefix[:-1], prefix[-1] + 1) if prefix else (release[0] + 1,)
            candidates.add(Version(".".join(str(part) for part in (*upper, 0))))
        if item.operator in {"==", "!="} and item.version.endswith(".*"):
            prefix = release
            candidates.add(Version(".".join(str(part) for part in (*prefix, 1))))
            upper = (*prefix[:-1], prefix[-1] + 1, 0)
            candidates.add(Version(".".join(str(part) for part in upper)))
    return tuple(candidates)


__all__ = ["EnvironmentRequirement", "marker_environment_from_record"]
