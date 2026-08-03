"""Runtime guard APIs for allocation and import safety."""

from __future__ import annotations

import importlib
import sys
import threading
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .allocation import NoAllocation, RuntimeAllocationView, is_no_allocation
from .context import active_runtime, active_runtime_bootstrap
from .enforcement import RuntimeEnforcement
from .errors import (
    FrameworkImportSafetyError,
    NoAllocationError,
    PublicationFailedError,
    RuntimeTransitionError,
)
from .modes import RuntimeMode
from .publication import publication


BOOTSTRAP_MARKER_ENV = "DRYML_RUNTIME_BOOTSTRAPPED"


@dataclass(slots=True)
class _ControlPlaneGuardScope:
    """One task-bound leased admission for a guarded control-plane operation."""

    control_epoch: int
    owner: tuple[int, int | None]
    kind: str
    active: bool = True


_CONTROL_PLANE_GUARD: ContextVar[_ControlPlaneGuardScope | None] = ContextVar(
    "dryml_control_plane_guard", default=None
)


def _owner_identity() -> tuple[int, int | None]:
    """Return the current thread and asyncio-task identity without creating a task."""

    try:
        asyncio = sys.modules.get("asyncio")
        task = None if asyncio is None else asyncio.current_task()
    except (AttributeError, RuntimeError):
        task = None
    return threading.get_ident(), None if task is None else id(task)


def _scope_is_current(scope: _ControlPlaneGuardScope, *, kind: str) -> bool:
    """Check context ownership and control-epoch lifetime for a nested scope."""

    if not scope.active or scope.kind != kind or scope.owner != _owner_identity():
        return False
    return int(publication.current().metadata.get("control_epoch", 0)) == scope.control_epoch


def internal_construction_admitted() -> bool:
    """Report whether a live materialization scope admits private fresh construction.

    This private bridge is intentionally not a public object-mode override. A
    copied context, sibling task, foreign thread, stale control epoch, or scope
    that has exited cannot retain the admission.
    """

    scope = _CONTROL_PLANE_GUARD.get()
    return scope is not None and _scope_is_current(scope, kind="materialization")


@contextmanager
def _assert_control_plane_allowed(*, operation: str, kind: str):
    """Lease one top-level operation and apply orchestration lifecycle policy."""

    existing = _CONTROL_PLANE_GUARD.get()
    if existing is not None and _scope_is_current(existing, kind=kind):
        _enforce_control_plane_policy(operation=operation, kind=kind, warn=False)
        yield existing
        return

    with publication.lease() as generation:
        scope = _ControlPlaneGuardScope(
            control_epoch=int(generation.metadata.get("control_epoch", 0)),
            owner=_owner_identity(),
            kind=kind,
        )
        token = _CONTROL_PLANE_GUARD.set(scope)
        try:
            _enforce_control_plane_policy(operation=operation, kind=kind, warn=True)
            yield scope
            current = publication.current()
            if current.health == "failed":
                raise PublicationFailedError(
                    "runtime publication failed during guarded work; restart the process",
                    context={
                        "operation": operation,
                        "restart_guidance": current.restart_guidance,
                    },
                )
        finally:
            scope.active = False
            _CONTROL_PLANE_GUARD.reset(token)


def _enforce_control_plane_policy(*, operation: str, kind: str, warn: bool) -> None:
    """Apply the active orchestrator policy, optionally owning its warning."""

    runtime = active_runtime()
    if runtime.mode is not RuntimeMode.ORCHESTRATOR:
        return
    if kind == "materialization":
        message = "Orchestration mode prohibits Object materialization"
        fix = "use Definition/CDef APIs for metadata, or execute in a managed inline session or dispatched worker"
    else:
        message = "Orchestration mode prohibits local workload execution"
        fix = "dispatch the workload to a worker, or execute in a managed inline session"
    context = {
        "mode": runtime.mode.value,
        "enforcement": runtime.enforcement.value,
        "operation": operation,
        "fix": fix,
    }
    if runtime.enforcement is RuntimeEnforcement.STRICT:
        _handle_enforcement_violation(message, error_type=RuntimeTransitionError, context=context)
    elif runtime.enforcement is RuntimeEnforcement.WARN and warn:
        _handle_enforcement_violation(message, error_type=RuntimeTransitionError, context=context)


def assert_object_materialization_allowed(*, operation: str):
    """Return a leased guard for a DRYML-owned live Object operation.

    Args:
        operation: Stable identifier for the attempted materialization action.

    Returns:
        A context manager which pins the runtime publication until completion.

    Raises:
        RuntimeTransitionError: Under strict orchestrator enforcement before
            construction or restoration begins.

    Side Effects:
        Warns once for the top-level operation under WARN and grants private
        fresh-construction admission only while the returned scope is active.
    """

    return _assert_control_plane_allowed(operation=operation, kind="materialization")


def assert_control_plane_target_execution_allowed(*, operation: str):
    """Return a leased guard for direct local workload execution.

    Args:
        operation: Stable identifier for the attempted local workload action.

    Returns:
        A context manager which pins the runtime publication until completion.

    Raises:
        RuntimeTransitionError: Under strict orchestrator enforcement before
            the target begins.
        PublicationFailedError: If runtime publication becomes terminal before
            otherwise successful guarded work returns.

    Side Effects:
        Warns once for the top-level operation under WARN. Unlike object
        materialization, strict rejection identifies local workload execution
        so callers do not receive a misleading construction diagnostic.
    """

    return _assert_control_plane_allowed(operation=operation, kind="target_execution")


def require_allocation(reason: str | None = None) -> RuntimeAllocationView:
    """Return active allocation or raise if no workload allocation is active."""

    runtime = active_runtime()
    if is_no_allocation(runtime.allocation):
        if not _handle_enforcement_violation(
            "workload allocation is required but active runtime has NoAllocation",
            error_type=NoAllocationError,
            context={"mode": runtime.mode.value, "reason": reason, "allocation": repr(runtime.allocation), "fix": "enter worker/inline runtime with allocation"},
        ):
            return runtime.allocation
    return runtime.allocation


def require_worker_allocation(reason: str | None = None) -> RuntimeAllocationView:
    """Return active allocation when the current mode is ``worker``."""

    runtime = active_runtime()
    if runtime.mode is not RuntimeMode.WORKER:
        if not _handle_enforcement_violation(
            "worker allocation requires worker runtime mode",
            error_type=RuntimeTransitionError,
            context={"mode": runtime.mode.value, "reason": reason, "allocation": repr(runtime.allocation), "fix": "enter RuntimeMode.WORKER with allocation"},
        ):
            return runtime.allocation
    return require_allocation(reason)


def assert_no_workload_allocation() -> None:
    """Raise if the current runtime has a workload allocation."""

    runtime = active_runtime()
    if not is_no_allocation(runtime.allocation):
        _handle_enforcement_violation(
            "active runtime holds workload allocation",
            error_type=RuntimeTransitionError,
            context={"mode": runtime.mode.value, "allocation": repr(runtime.allocation), "fix": "use orchestrator/probe mode without allocation"},
        )


def assert_framework_import_configured(framework_name: str, desired_visibility: Any = None) -> None:
    """Raise unless current context has configured framework bootstrap."""

    bootstrap = active_runtime_bootstrap()
    if bootstrap is None:
        _handle_enforcement_violation(
            "framework import requires active runtime bootstrap",
            error_type=FrameworkImportSafetyError,
            context={"framework": framework_name, "desired_visibility": desired_visibility, "fix": "build/apply runtime visibility before importing frameworks"},
        )
        return
    if framework_name not in bootstrap.frameworks:
        _handle_enforcement_violation(
            "framework import was not configured by active runtime bootstrap",
            error_type=FrameworkImportSafetyError,
            context={"framework": framework_name, "configured_frameworks": sorted(bootstrap.frameworks), "fix": "include framework in RuntimeContextSpec.frameworks before bootstrap"},
        )


def assert_framework_import_safe(framework_name: str, desired_visibility: Any = None) -> None:
    """Compatibility alias for configured framework import checks."""

    assert_framework_import_configured(framework_name, desired_visibility=desired_visibility)


def import_configured_framework(framework_name: str, module_name: str | None = None):
    """Import a framework module, guarding imports that would first load it.

    If user/test code already imported the framework, this helper reuses that
    module. Runtime bootstrap is the barrier before DRYML itself newly imports a
    heavy framework; it is not a retroactive blocker after the framework is
    already loaded in the process.
    """

    target = module_name or framework_name
    if target in sys.modules or framework_name in sys.modules:
        module = importlib.import_module(target)
        _apply_framework_post_import(framework_name, target)
        return module
    assert_framework_import_configured(framework_name)
    module = importlib.import_module(target)
    _apply_framework_post_import(framework_name, target)
    return module


def _apply_framework_post_import(framework_name: str, module_name: str | None = None) -> None:
    bootstrap = active_runtime_bootstrap()
    if bootstrap is None or framework_name not in bootstrap.frameworks:
        return
    if framework_name not in sys.modules:
        # Helper packages such as tensorflow_datasets may be configured with a
        # framework plan without importing the framework root itself.
        return
    from .imports import finalize_helper

    # Raw imports are finalized by the wrapped loader.  This compatibility path
    # remains for an already-loaded module or a legacy custom adapter.
    if finalize_helper(framework_name, module_name):
        return
    result = bootstrap.framework_results.get(framework_name)
    if result is None:
        return

    from .frameworks import default_adapters

    adapter = default_adapters().get(framework_name)
    if adapter is not None:
        adapter.apply_post_import(result)


def require_workload_allocation(reason: str | None = None) -> RuntimeAllocationView:
    """Require worker/inline runtime mode with an active workload allocation."""

    runtime = active_runtime()
    if runtime.mode not in {RuntimeMode.WORKER, RuntimeMode.INLINE} or is_no_allocation(runtime.allocation):
        if not _handle_enforcement_violation(
            "workload resources require worker/inline runtime with allocation",
            error_type=FrameworkImportSafetyError,
            context={"mode": runtime.mode.value, "reason": reason, "allocation": repr(runtime.allocation), "fix": "enter worker/inline runtime before materializing workload objects"},
        ):
            return runtime.allocation
    return runtime.allocation


def _handle_enforcement_violation(message: str, *, error_type=RuntimeError, context: dict[str, Any] | None = None) -> bool:
    policy = active_runtime().enforcement
    if policy is RuntimeEnforcement.STRICT:
        raise error_type(message, context=context)
    if policy is RuntimeEnforcement.WARN:
        details = f" {context!r}" if context else ""
        warnings.warn(f"{message}{details}", RuntimeWarning, stacklevel=3)
        return False
    if policy is RuntimeEnforcement.OFF:
        return False
    raise AssertionError(f"unknown runtime enforcement policy: {policy!r}")


__all__ = [
    "BOOTSTRAP_MARKER_ENV",
    "assert_framework_import_configured",
    "assert_framework_import_safe",
    "assert_object_materialization_allowed",
    "assert_control_plane_target_execution_allowed",
    "assert_no_workload_allocation",
    "import_configured_framework",
    "internal_construction_admitted",
    "require_allocation",
    "require_worker_allocation",
    "require_workload_allocation",
]
