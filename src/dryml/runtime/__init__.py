"""Process-local runtime mode, allocation, visibility, and bootstrap.

``dryml.runtime`` is the runtime half of the environment/world/runtime split.
It defaults to ``orchestrator`` mode with ``NoAllocation`` so imports and
control-plane code do not accidentally capture workload accelerators.
"""

from dryml.runtime.allocation import NoAllocation, RuntimeAllocationView
from dryml.runtime.bootstrap import FrameworkBootstrapPolicy, RuntimeBootstrapPlan, activate, activate_runtime_bootstrap, apply_runtime_bootstrap_plan, build_runtime_bootstrap_plan
from dryml.runtime.context import RuntimeBootstrapState, RuntimeState, active_runtime, active_runtime_bootstrap, active_runtime_mode, enter_runtime, reset_runtime
from dryml.runtime.devices import DeviceVisibilityPlan, DeviceVisibilityPolicy, apply_device_visibility_plan, build_device_visibility_plan
from dryml.runtime.guards import BOOTSTRAP_MARKER_ENV, assert_framework_import_configured, assert_framework_import_safe, assert_no_workload_allocation, import_configured_framework, require_allocation, require_allocation_for_legacy_compute_reqs, require_worker_allocation, require_workload_allocation
from dryml.runtime.modes import RuntimeMode
from dryml.runtime.specs import RuntimeContextSpec, attach_runtime_id, compute_runtime_id, make_runtime_spec, validate_runtime_spec

__all__ = [
    "DeviceVisibilityPlan",
    "DeviceVisibilityPolicy",
    "FrameworkBootstrapPolicy",
    "BOOTSTRAP_MARKER_ENV",
    "NoAllocation",
    "RuntimeBootstrapState",
    "RuntimeAllocationView",
    "RuntimeBootstrapPlan",
    "RuntimeContextSpec",
    "RuntimeMode",
    "RuntimeState",
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
    "enter_runtime",
    "import_configured_framework",
    "make_runtime_spec",
    "require_allocation",
    "require_allocation_for_legacy_compute_reqs",
    "require_worker_allocation",
    "require_workload_allocation",
    "reset_runtime",
    "validate_runtime_spec",
]
