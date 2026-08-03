"""Process-local runtime mode, allocation, visibility, and bootstrap.

``dryml.runtime`` is the runtime half of the environment/world/runtime split.
Its ordinary Python baseline is ``none`` with ``NoAllocation`` and enforcement
off; importing it does not alter inherited device visibility.
"""

from dryml.runtime.allocation import NoAllocation, RuntimeAllocationView
from dryml.runtime.bootstrap import FrameworkBootstrapPolicy, RuntimeBootstrapPlan, activate, activate_runtime_bootstrap, apply_runtime_bootstrap_plan, build_runtime_bootstrap_plan
from dryml.runtime.context import RuntimeBootstrapState, RuntimeState, active_runtime, active_runtime_bootstrap, active_runtime_mode, disable, disabled, enable, enforcement, enter_runtime, plain, reset_runtime, set_enforcement
from dryml.runtime.compatibility import RuntimeCompatibilityIssue, RuntimeCompatibilityReport, check_runtime_spec_satisfies_requirement
from dryml.runtime.devices import DeviceVisibilityPlan, DeviceVisibilityPolicy, apply_device_visibility_plan, build_device_visibility_plan
from dryml.runtime.decorators import default
from dryml.runtime.enforcement import REQUIREMENT_AXIS_NAMES, RequirementAxes, RuntimeEnforcement, normalize_enforcement, normalize_requirement_axes, startup_enforcement_from_env
from dryml.runtime.guards import BOOTSTRAP_MARKER_ENV, assert_framework_import_configured, assert_framework_import_safe, assert_no_workload_allocation, import_configured_framework, require_allocation, require_worker_allocation, require_workload_allocation
from dryml.runtime.frameworks import FrameworkCapabilities, FrameworkImportPlan, FrameworkPostResult, FrameworkRegistration, FrameworkRegistry, framework_registry
from dryml.runtime.modes import RuntimeMode
from dryml.runtime.publication import EffectPlan, EffectRecord, PublicationCandidate, PublicationService, SessionGeneration, publication
from dryml.runtime.specs import RuntimeContextSpec, attach_runtime_id, compute_runtime_id, make_runtime_spec, validate_runtime_spec

__all__ = [
    "DeviceVisibilityPlan",
    "DeviceVisibilityPolicy",
    "EffectPlan",
    "EffectRecord",
    "FrameworkBootstrapPolicy",
    "FrameworkCapabilities",
    "FrameworkImportPlan",
    "FrameworkPostResult",
    "FrameworkRegistration",
    "FrameworkRegistry",
    "BOOTSTRAP_MARKER_ENV",
    "NoAllocation",
    "PublicationCandidate",
    "PublicationService",
    "RuntimeBootstrapState",
    "RuntimeAllocationView",
    "RuntimeBootstrapPlan",
    "RuntimeContextSpec",
    "RuntimeCompatibilityIssue",
    "RuntimeCompatibilityReport",
    "RuntimeEnforcement",
    "RuntimeMode",
    "RuntimeState",
    "RequirementAxes",
    "REQUIREMENT_AXIS_NAMES",
    "SessionGeneration",
    "activate",
    "activate_runtime_bootstrap",
    "active_runtime",
    "active_runtime_bootstrap",
    "active_runtime_mode",
    "apply_device_visibility_plan",
    "apply_runtime_bootstrap_plan",
    "assert_framework_import_configured",
    "assert_framework_import_safe",
    "assert_no_workload_allocation",
    "attach_runtime_id",
    "build_device_visibility_plan",
    "build_runtime_bootstrap_plan",
    "compute_runtime_id",
    "check_runtime_spec_satisfies_requirement",
    "default",
    "disable",
    "disabled",
    "enable",
    "enforcement",
    "enter_runtime",
    "import_configured_framework",
    "make_runtime_spec",
    "normalize_enforcement",
    "normalize_requirement_axes",
    "plain",
    "publication",
    "framework_registry",
    "require_allocation",
    "require_worker_allocation",
    "require_workload_allocation",
    "reset_runtime",
    "set_enforcement",
    "startup_enforcement_from_env",
    "validate_runtime_spec",
]
