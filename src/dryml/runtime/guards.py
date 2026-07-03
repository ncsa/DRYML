"""Runtime guard APIs for allocation and import safety."""

from __future__ import annotations

import sys
from typing import Any

from .allocation import NoAllocation, RuntimeAllocationView, is_no_allocation
from .context import active_runtime
from .errors import FrameworkImportSafetyError, NoAllocationError, RuntimeTransitionError
from .modes import RuntimeMode


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


def assert_framework_import_safe(framework_name: str, desired_visibility: Any = None) -> None:
    """Raise if a framework was already imported before visibility setup."""

    if framework_name in sys.modules:
        raise FrameworkImportSafetyError(
            "framework was imported before runtime visibility setup",
            context={"framework": framework_name, "desired_visibility": desired_visibility, "fix": "build/apply runtime visibility before importing frameworks"},
        )


__all__ = ["assert_framework_import_safe", "assert_no_workload_allocation", "require_allocation", "require_worker_allocation"]
