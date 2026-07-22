"""Dispatch request, planning, and local subprocess execution APIs."""

from .errors import DispatchCancelled, DispatchLaunchError, DispatchPlanningError, DispatchSpecError, DispatchTimeout, WorkerHandshakeError, WorkerProtocolError
from .fake import fake_execution_record
from .links import EXECUTION_KINDS, EXECUTION_STATUSES, normalize_backend_identity, normalize_execution_kind, normalize_execution_status
from .recipes import (
    EXECUTION_RECIPE_KIND,
    EXECUTION_RECIPE_SCHEMA,
    EXECUTION_RECIPE_SCHEMA_VERSION,
    EXECUTION_RECIPE_SPEC_FAMILY,
    attach_recipe_id,
    compute_recipe_id,
    make_execution_recipe,
    recipe_payload_for_id,
    validate_execution_recipe,
)
from .specs import (
    DISPATCH_KIND,
    DISPATCH_SCHEMA,
    DISPATCH_SCHEMA_VERSION,
    DISPATCH_SPEC_FAMILY,
    attach_dispatch_id,
    compute_dispatch_id,
    dispatch_payload_for_id,
    make_dispatch_spec,
    validate_dispatch_spec,
)


_LAZY_EXPORTS = {
    "DispatchPlan": (".planner", "DispatchPlan"),
    "DispatchExplanation": (".requirements", "DispatchExplanation"),
    "DispatchPlanningResolution": (".requirements", "DispatchPlanningResolution"),
    "Dispatcher": (".planner", "Dispatcher"),
    "NormalizedDispatchTarget": (".normalize", "NormalizedDispatchTarget"),
    "import_path_for_callable": (".normalize", "import_path_for_callable"),
    "normalize_user_operation": (".normalize", "normalize_user_operation"),
    "plan": (".planner", "plan"),
    "plan_world": (".planner", "plan_world"),
    "explain": (".planner", "explain"),
    "run": (".planner", "run"),
    "run_world": (".planner", "run_world"),
    "submit": (".planner", "submit"),
    "CandidateCheckReport": (".requirements", "CandidateCheckReport"),
    "CandidateConsideration": (".requirements", "CandidateConsideration"),
    "CandidateSelection": (".requirements", "CandidateSelection"),
    "RequirementPolicy": (".requirements", "RequirementPolicy"),
    "effective_requirement_policy": (".requirements", "effective_requirement_policy"),
    "resolve_dispatch_plan": (".requirements", "resolve_dispatch_plan"),
    "LocalSubprocessBackend": (".backends", "LocalSubprocessBackend"),
    "LocalSubprocessFuture": (".backends", "LocalSubprocessFuture"),
    "build_worker_command": (".backends", "build_worker_command"),
    "LocalResourceInventory": (".local_world", "LocalResourceInventory"),
    "LocalWorldBackend": (".local_world", "LocalWorldBackend"),
    "LocalWorldFuture": (".local_world", "LocalWorldFuture"),
    "LocalWorldPlan": (".local_world", "LocalWorldPlan"),
    "WorkerLaunchPlan": (".local_world", "WorkerLaunchPlan"),
    "WorldDispatchResult": (".local_world", "WorldDispatchResult"),
    "WorldWorkerKey": (".local_world", "WorldWorkerKey"),
    "allocate_local_world": (".local_world", "allocate_local_world"),
    "PickledCallable": (".operations", "PickledCallable"),
    "DispatchResult": (".protocol", "DispatchResult"),
    "ExecutionEnvelope": (".protocol", "ExecutionEnvelope"),
    "WorkerHandshakeRequest": (".protocol", "WorkerHandshakeRequest"),
    "WorkerHandshakeResponse": (".protocol", "WorkerHandshakeResponse"),
    "WorkerResponse": (".protocol", "WorkerResponse"),
    "WorkerStoreRef": (".protocol", "WorkerStoreRef"),
}


def __getattr__(name):
    """Lazily import execution-heavy dispatch APIs on first access."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "DISPATCH_KIND",
    "DISPATCH_SCHEMA",
    "DISPATCH_SCHEMA_VERSION",
    "DISPATCH_SPEC_FAMILY",
    "EXECUTION_KINDS",
    "EXECUTION_RECIPE_KIND",
    "EXECUTION_RECIPE_SCHEMA",
    "EXECUTION_RECIPE_SCHEMA_VERSION",
    "EXECUTION_RECIPE_SPEC_FAMILY",
    "EXECUTION_STATUSES",
    "DispatchCancelled",
    "DispatchExplanation",
    "DispatchPlanningResolution",
    "DispatchLaunchError",
    "DispatchPlan",
    "DispatchPlanningError",
    "DispatchResult",
    "DispatchSpecError",
    "Dispatcher",
    "DispatchTimeout",
    "CandidateCheckReport",
    "CandidateConsideration",
    "CandidateSelection",
    "ExecutionEnvelope",
    "LocalSubprocessBackend",
    "LocalSubprocessFuture",
    "LocalResourceInventory",
    "LocalWorldBackend",
    "LocalWorldFuture",
    "LocalWorldPlan",
    "NormalizedDispatchTarget",
    "PickledCallable",
    "RequirementPolicy",
    "WorkerLaunchPlan",
    "WorkerHandshakeError",
    "WorkerHandshakeRequest",
    "WorkerHandshakeResponse",
    "WorkerProtocolError",
    "WorkerResponse",
    "WorkerStoreRef",
    "WorldDispatchResult",
    "WorldWorkerKey",
    "allocate_local_world",
    "attach_dispatch_id",
    "attach_recipe_id",
    "build_worker_command",
    "compute_dispatch_id",
    "compute_recipe_id",
    "dispatch_payload_for_id",
    "effective_requirement_policy",
    "explain",
    "fake_execution_record",
    "make_dispatch_spec",
    "make_execution_recipe",
    "normalize_backend_identity",
    "normalize_execution_kind",
    "normalize_execution_status",
    "normalize_user_operation",
    "import_path_for_callable",
    "plan",
    "plan_world",
    "recipe_payload_for_id",
    "run",
    "run_world",
    "resolve_dispatch_plan",
    "submit",
    "validate_dispatch_spec",
    "validate_execution_recipe",
]
