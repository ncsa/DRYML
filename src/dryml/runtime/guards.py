"""Runtime guard APIs for allocation and import safety."""

from __future__ import annotations

import importlib
import sys
from typing import Any

from .allocation import NoAllocation, RuntimeAllocationView, is_no_allocation
from .context import active_runtime, active_runtime_bootstrap
from .errors import FrameworkImportSafetyError, NoAllocationError, RuntimeTransitionError
from .modes import RuntimeMode


BOOTSTRAP_MARKER_ENV = "DRYML_RUNTIME_BOOTSTRAPPED"


def require_allocation(reason: str | None = None) -> RuntimeAllocationView:
    """Return active allocation or raise if no workload allocation is active."""

    runtime = active_runtime()
    if is_no_allocation(runtime.allocation):
        raise NoAllocationError(
            "workload allocation is required but active runtime has NoAllocation",
            context={"mode": runtime.mode.value, "reason": reason, "allocation": repr(runtime.allocation), "fix": "enter worker/inline runtime with allocation"},
        )
    return runtime.allocation


def require_worker_allocation(reason: str | None = None) -> RuntimeAllocationView:
    """Return active allocation when the current mode is ``worker``."""

    runtime = active_runtime()
    if runtime.mode is not RuntimeMode.WORKER:
        raise RuntimeTransitionError(
            "worker allocation requires worker runtime mode",
            context={"mode": runtime.mode.value, "reason": reason, "allocation": repr(runtime.allocation), "fix": "enter RuntimeMode.WORKER with allocation"},
        )
    return require_allocation(reason)


def assert_no_workload_allocation() -> None:
    """Raise if the current runtime has a workload allocation."""

    runtime = active_runtime()
    if not is_no_allocation(runtime.allocation):
        raise RuntimeTransitionError(
            "active runtime holds workload allocation",
            context={"mode": runtime.mode.value, "allocation": repr(runtime.allocation), "fix": "use orchestrator/probe mode without allocation"},
        )


def assert_framework_import_configured(framework_name: str, desired_visibility: Any = None) -> None:
    """Raise unless current context has configured framework bootstrap."""

    bootstrap = active_runtime_bootstrap()
    if bootstrap is None:
        raise FrameworkImportSafetyError(
            "framework import requires active runtime bootstrap",
            context={"framework": framework_name, "desired_visibility": desired_visibility, "fix": "build/apply runtime visibility before importing frameworks"},
        )
    if framework_name not in bootstrap.frameworks:
        raise FrameworkImportSafetyError(
            "framework import was not configured by active runtime bootstrap",
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
        _apply_framework_post_import(framework_name)
        return module
    assert_framework_import_configured(framework_name)
    module = importlib.import_module(target)
    _apply_framework_post_import(framework_name)
    return module


def _apply_framework_post_import(framework_name: str) -> None:
    bootstrap = active_runtime_bootstrap()
    if bootstrap is None or framework_name not in bootstrap.frameworks:
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
        raise FrameworkImportSafetyError(
            "workload resources require worker/inline runtime with allocation",
            context={"mode": runtime.mode.value, "reason": reason, "allocation": repr(runtime.allocation), "fix": "enter worker/inline runtime before materializing workload objects"},
        )
    return runtime.allocation


__all__ = ["BOOTSTRAP_MARKER_ENV", "assert_framework_import_configured", "assert_framework_import_safe", "assert_no_workload_allocation", "import_configured_framework", "require_allocation", "require_worker_allocation", "require_workload_allocation"]
