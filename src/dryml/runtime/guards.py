"""Runtime guard APIs for allocation and import safety."""

from __future__ import annotations

from collections.abc import Mapping
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


def require_allocation_for_legacy_compute_reqs(reqs: Any, reason: str | None = None) -> RuntimeAllocationView:
    """Return active allocation if it satisfies legacy ``__compute_reqs__``.

    This is a compatibility bridge for old context-style requirements. New
    code should express hard resource constraints with ``WorldRequirement`` and
    validate them before entering runtime.
    """

    allocation = require_allocation(reason)
    resources = _combine_legacy_compute_reqs(reqs)
    cpus = tuple(getattr(allocation, "cpus", ()))
    accelerators = getattr(allocation, "accelerators", {})
    gpus = tuple(accelerators.get("gpu", ()))
    memory = getattr(allocation, "memory", None)

    failures: dict[str, Any] = {}
    if len(cpus) < resources["num_cpus"]:
        failures["num_cpus"] = {"required": resources["num_cpus"], "actual": len(cpus)}
    if len(gpus) < resources["num_gpus"]:
        failures["num_gpus"] = {"required": resources["num_gpus"], "actual": len(gpus)}
    if resources["memory_bytes"] is not None and (memory is None or memory < resources["memory_bytes"]):
        failures["memory_bytes"] = {"required": resources["memory_bytes"], "actual": memory}
    for key, need in resources["specific"].items():
        kind, _, index = key.partition("/")
        assigned = cpus if kind == "cpu" else gpus if kind == "gpu" else ()
        if str(index) not in {str(item) for item in assigned}:
            failures[key] = {"required_fraction": need, "actual": "missing"}
    if failures:
        raise NoAllocationError(
            "active runtime allocation does not satisfy legacy compute requirements",
            context={
                "mode": active_runtime().mode.value,
                "reason": reason,
                "requirements": resources,
                "failures": failures,
                "allocation": repr(allocation),
                "fix": "enter worker/inline runtime with an allocation that satisfies the compute requirements",
            },
        )
    return allocation


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


def require_workload_allocation(reason: str | None = None) -> RuntimeAllocationView:
    """Require worker/inline runtime mode with an active workload allocation."""

    runtime = active_runtime()
    if runtime.mode not in {RuntimeMode.WORKER, RuntimeMode.INLINE} or is_no_allocation(runtime.allocation):
        raise FrameworkImportSafetyError(
            "workload resources require worker/inline runtime with allocation",
            context={"mode": runtime.mode.value, "reason": reason, "allocation": repr(runtime.allocation), "fix": "enter worker/inline runtime before materializing workload objects"},
        )
    return runtime.allocation


def _combine_legacy_compute_reqs(reqs: Any) -> dict[str, Any]:
    specs = _iter_legacy_specs(reqs)
    specific_keys = set()
    for spec in specs:
        specific_keys.update(spec["specific"])
    return {
        "num_cpus": max((spec["num_cpus"] for spec in specs), default=0),
        "num_gpus": max((spec["num_gpus"] for spec in specs), default=0),
        "memory_bytes": max((spec["memory_bytes"] for spec in specs if spec["memory_bytes"] is not None), default=None),
        "specific": {key: max(spec["specific"].get(key, 0.0) for spec in specs) for key in specific_keys},
    }


def _iter_legacy_specs(reqs: Any) -> list[dict[str, Any]]:
    if reqs is None:
        return []
    if isinstance(reqs, str):
        return [_legacy_spec_from_user(None)]
    if isinstance(reqs, Mapping):
        return [_legacy_spec_from_user(value) for value in reqs.values()]
    try:
        iterator = iter(reqs)
    except TypeError as exc:
        raise TypeError("legacy compute requirements must be None, string, iterable, or mapping") from exc
    specs = []
    for item in iterator:
        if not isinstance(item, str):
            raise TypeError("legacy iterable compute requirements must contain context names")
        specs.append(_legacy_spec_from_user(None))
    return specs


def _legacy_spec_from_user(value: Any) -> dict[str, Any]:
    if value is None:
        return {"num_cpus": 0, "num_gpus": 0, "memory_bytes": None, "specific": {}}
    if hasattr(value, "num_cpus") and hasattr(value, "num_gpus"):
        return {
            "num_cpus": _as_nonneg_int("num_cpus", value.num_cpus),
            "num_gpus": _as_nonneg_int("num_gpus", value.num_gpus),
            "memory_bytes": _memory_bytes(getattr(value, "memory_bytes", None)),
            "specific": {str(key): float(req) for key, req in dict(getattr(value, "specific", {})).items()},
        }
    if not isinstance(value, Mapping):
        raise TypeError("legacy per-context compute requirement must be a mapping or ResourceSpec-like object")
    raw = dict(value)
    num_cpus = _as_nonneg_int("num_cpus", raw.pop("num_cpus", 0))
    num_gpus = _as_nonneg_int("num_gpus", raw.pop("num_gpus", 0))
    memory_bytes = _memory_bytes(raw.pop("memory_bytes", None))
    specific = {str(key): float(req) for key, req in raw.items()}
    return {"num_cpus": num_cpus, "num_gpus": num_gpus, "memory_bytes": memory_bytes, "specific": specific}


def _as_nonneg_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be an integer >= 0")
    return value


def _memory_bytes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("memory_bytes must be an integer >= 0")
    value = int(value)
    if value < 0:
        raise TypeError("memory_bytes must be >= 0")
    return value


__all__ = ["BOOTSTRAP_MARKER_ENV", "assert_framework_import_configured", "assert_framework_import_safe", "assert_no_workload_allocation", "require_allocation", "require_allocation_for_legacy_compute_reqs", "require_worker_allocation", "require_workload_allocation"]
