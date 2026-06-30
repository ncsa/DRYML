"""Standalone software-environment metadata for DRYML.

The :mod:`dryml.environments` package describes, inspects, probes, compares,
serializes, and interns Python/software environment metadata. It is deliberately
separate from DRYML object identity, Repo/Store persistence, runtime resource
allocation, dispatch, providers, and heavyweight ML framework imports.
"""

from .compatibility import CompatibilityIssue, CompatibilityReport, coerce_policy
from .errors import (
    DrymlEnvironmentError,
    EnvironmentCompatibilityError,
    EnvironmentFeatureUnavailable,
    EnvironmentProbeError,
    EnvironmentRegistryError,
    EnvironmentRequirementError,
    EnvironmentSerializationError,
    EnvironmentSpecError,
)
from .fragments import (
    RequirementFragment,
    add_req,
    compose_fragments,
    fragments_for_class,
    override_req,
    req,
    requirements_for_class,
)
from .introspection import inspect_current
from .probe import EnvironmentProbeResult, probe, probe_conda, probe_current, probe_python
from .records import (
    DrymlRuntimeRecord,
    EnvironmentInternTable,
    EnvironmentRecord,
    PackageRecord,
    PlatformRecord,
    PythonRecord,
)
from .registry import EnvironmentRegistry, EnvironmentRegistryEntry
from .requirements import EnvironmentRequirement, marker_environment_from_record
from .schema import (
    COMPATIBILITY_REPORT_SCHEMA_VERSION,
    ENVIRONMENT_FRAGMENT_SCHEMA_VERSION,
    ENVIRONMENT_LOCK_REF_SCHEMA_VERSION,
    ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION,
    ENVIRONMENT_RECORD_SCHEMA_VERSION,
    ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION,
    ENVIRONMENT_SPEC_SCHEMA_VERSION,
)
from .specs import (
    CondaEnvironmentSpec,
    ContainerEnvironmentSpec,
    CurrentEnvironmentSpec,
    EnvironmentLockRef,
    PythonExecutableSpec,
    spec_from_data,
)
from .utils import build_probe_env, normalize_distribution_name, normalize_requirement_string

__all__ = [
    "COMPATIBILITY_REPORT_SCHEMA_VERSION",
    "ENVIRONMENT_FRAGMENT_SCHEMA_VERSION",
    "ENVIRONMENT_LOCK_REF_SCHEMA_VERSION",
    "ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION",
    "ENVIRONMENT_RECORD_SCHEMA_VERSION",
    "ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION",
    "ENVIRONMENT_SPEC_SCHEMA_VERSION",
    "CompatibilityIssue",
    "CompatibilityReport",
    "CondaEnvironmentSpec",
    "ContainerEnvironmentSpec",
    "CurrentEnvironmentSpec",
    "DrymlEnvironmentError",
    "DrymlRuntimeRecord",
    "EnvironmentCompatibilityError",
    "EnvironmentFeatureUnavailable",
    "EnvironmentInternTable",
    "EnvironmentLockRef",
    "EnvironmentProbeError",
    "EnvironmentProbeResult",
    "EnvironmentRecord",
    "EnvironmentRegistry",
    "EnvironmentRegistryEntry",
    "EnvironmentRegistryError",
    "EnvironmentRequirement",
    "EnvironmentRequirementError",
    "EnvironmentSerializationError",
    "EnvironmentSpecError",
    "PackageRecord",
    "PlatformRecord",
    "PythonExecutableSpec",
    "PythonRecord",
    "RequirementFragment",
    "add_req",
    "build_probe_env",
    "coerce_policy",
    "compose_fragments",
    "fragments_for_class",
    "inspect_current",
    "normalize_distribution_name",
    "normalize_requirement_string",
    "marker_environment_from_record",
    "override_req",
    "probe",
    "probe_conda",
    "probe_current",
    "probe_python",
    "req",
    "requirements_for_class",
    "spec_from_data",
]
