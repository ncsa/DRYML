"""PID-safe runtime declarations, publication, and materialization action scopes.

The package exposes only passive ``NONE``, strict definition-only
``ORCHESTRATOR``, and exact-allocation ``INLINE`` roles. Framework import
interception, public session configuration, decorators, workers, and probes
are intentionally outside this implementation unit.
"""

from .allocation import NoAllocation, RuntimeAllocationView, is_no_allocation
from .compatibility import RuntimeCompatibilityIssue, RuntimeCompatibilityReport, check_runtime_spec_satisfies_requirement
from .context import RuntimeState, active_runtime, active_runtime_mode, publication
from .devices import DeviceVisibilityPlan, DeviceVisibilityPolicy, build_device_visibility_plan
from .enforcement import CONTROL_CATEGORIES, ControlPlan, ControlStatus, RuntimeEnforcement, build_control_plan
from .errors import DeviceVisibilityError, ForkSafetyError, PublicationBusyError, PublicationError, PublicationFailedError, PublicationReentryError, RuntimeErrorBase, RuntimeSpecError, RuntimeTransitionError
from .guards import MaterializationAction, materialization_action, materialization_admission, materialization_scope
from .modes import RuntimeMode
from .publication import EffectPlan, EffectRecord, FrameworkAdmission, PublicationCandidate, PublicationService, SessionGeneration
from .specs import RuntimeContextSpec
from dryml.worlds import LocalResourceInventory

__all__ = ["CONTROL_CATEGORIES", "ControlPlan", "ControlStatus", "DeviceVisibilityError", "DeviceVisibilityPlan", "DeviceVisibilityPolicy", "EffectPlan", "EffectRecord", "ForkSafetyError", "FrameworkAdmission", "LocalResourceInventory", "MaterializationAction", "NoAllocation", "PublicationBusyError", "PublicationCandidate", "PublicationError", "PublicationFailedError", "PublicationReentryError", "PublicationService", "RuntimeAllocationView", "RuntimeCompatibilityIssue", "RuntimeCompatibilityReport", "RuntimeContextSpec", "RuntimeEnforcement", "RuntimeErrorBase", "RuntimeMode", "RuntimeSpecError", "RuntimeState", "RuntimeTransitionError", "SessionGeneration", "active_runtime", "active_runtime_mode", "build_control_plan", "build_device_visibility_plan", "check_runtime_spec_satisfies_requirement", "is_no_allocation", "materialization_action", "materialization_admission", "materialization_scope", "publication"]
