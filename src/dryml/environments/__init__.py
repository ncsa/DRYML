"""Import-light closed v1.1 environment declarations and compatibility APIs.

Probing and introspection remain explicit opt-in submodules; importing this
package neither probes the host nor imports probe workers, records sidecars,
runtime, worlds, or optional frameworks.
"""

from .compatibility import CompatibilityIssue, CompatibilityReport, coerce_policy, malformed_report, unavailable_report
from .current import current, reset_current, set_current, use
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
from .combination import requirements_for, requirements_for_method
from .declarations import ENVIRONMENT_REQUIREMENT_KEY, req
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
from .schema import COMPATIBILITY_REPORT_SCHEMA_VERSION, ENVIRONMENT_LOCK_REF_SCHEMA_VERSION, ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION, ENVIRONMENT_RECORD_SCHEMA_VERSION, ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION, ENVIRONMENT_SPEC_SCHEMA_VERSION
from .specs import (
    CondaEnvironmentSpec,
    ContainerEnvironmentSpec,
    CurrentEnvironmentSpec,
    EnvironmentLockRef,
    PythonExecutableSpec,
    spec_from_data,
)
from .utils import build_probe_env, normalize_distribution_name, normalize_requirement_string

_EXPLICIT_EXPORTS = {
    "EnvironmentRequirementsKernel":
    (".kernel", "EnvironmentRequirementsKernel"),
    "inspect_current": (".introspection", "inspect_current"),
    "EnvironmentProbeResult": (".probe", "EnvironmentProbeResult"),
    "probe": (".probe", "probe"),
    "probe_conda": (".probe", "probe_conda"),
    "probe_current": (".probe", "probe_current"),
    "probe_python": (".probe", "probe_python"),
    "ResolvedEnvironmentSelection":
    (".selection", "ResolvedEnvironmentSelection"),
    "compare_selection": (".selection", "compare_selection"),
    "resolve_environment_spec": (".selection", "resolve_environment_spec"),
    "software_digest": (".selection", "software_digest"),
}


def _load_explicit_export(name: str):
    """Load and cache one explicit lazy export.

    This also restores a callable package export when a sibling imports its
    identically named implementation submodule.

    Args:
        name: Exact supported lazy export name.

    Returns:
        The cached public API value from its explicit lightweight submodule.

    Raises:
        AttributeError: If ``name`` is not a supported lazy export.
    """

    try:
        module_name, attribute = _EXPLICIT_EXPORTS[name]
    except KeyError as exc:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from exc
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __getattr__(name: str):
    """
    Lazily expose explicit probe, introspection, selection, and kernel APIs.

        Args:
            name: Exact exported API name.

        Returns:
            The cached public API value from its explicit lightweight
            submodule.

        Raises:
            AttributeError: If ``name`` is not a supported lazy export.

        Side Effects:
            Imports and caches only the selected optional submodule. Selection
            access
            can inspect an explicitly requested environment but never imports
            Execute;
            kernel access imports generic code analysis but never probes a host
            or
            imports worlds.
    """

    return _load_explicit_export(name)

__all__ = [
    "COMPATIBILITY_REPORT_SCHEMA_VERSION",
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
    "EnvironmentRecord",
    "EnvironmentRegistry",
    "EnvironmentRegistryEntry",
    "EnvironmentRegistryError",
    "EnvironmentRequirement",
    "EnvironmentRequirementsKernel",
    "EnvironmentRequirementError",
    "EnvironmentSerializationError",
    "EnvironmentSpecError",
    "PackageRecord",
    "PlatformRecord",
    "PythonExecutableSpec",
    "PythonRecord",
    "coerce_policy",
    "malformed_report",
    "normalize_distribution_name",
    "normalize_requirement_string",
    "marker_environment_from_record",
    "req",
    "requirements_for",
    "requirements_for_method",
    "spec_from_data",
    "current",
    "reset_current",
    "set_current",
    "use",
    "unavailable_report",
    "EnvironmentProbeResult",
    "inspect_current",
    "probe",
    "probe_conda",
    "probe_current",
    "probe_python",
    "ResolvedEnvironmentSelection",
    "compare_selection",
    "resolve_environment_spec",
    "software_digest",
]
