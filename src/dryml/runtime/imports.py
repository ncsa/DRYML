"""Runtime lifecycle half of the passive watched-framework finder.

The loader wrapper keeps its original ``ModuleSpec`` and delegated loader.
This module adds only admission, pre-import visibility, status finalization,
and failure handling around those PEP 451 callbacks.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from dryml._framework_imports import coordinator

from .bootstrap import build_framework_bootstrap_plan
from .context import publication
from .errors import FrameworkImportSafetyError
from .frameworks import FrameworkPostResult, FrameworkRegistration, framework_registry
from .modes import RuntimeMode
from .publication import FrameworkAdmission


@dataclass(slots=True)
class _Lifecycle:
    """Retain one root loader's reader, generation lease, and immutable plan."""

    fullname: str
    registration: FrameworkRegistration
    admission: FrameworkAdmission | None
    adapter: Any | None
    plan: Any | None
    reader: AbstractContextManager[Any]
    lease: AbstractContextManager[Any]
    controlled: bool
    executed: bool = False


_lifecycles: dict[int, _Lifecycle] = {}
_lock = threading.Lock()
_group_plans: dict[tuple[int, str], Any] = {}


def register_framework(registration: FrameworkRegistration) -> None:
    """Register an advanced adapter group before it is observed or loaded."""
    framework_registry.register(registration)


def create_module(fullname: str, wrapped_spec: Any, loader: Any, spec: Any) -> Any:
    """Validate pre-import controls and delegate original module creation.

    A controlled root retains its import reader and U5 generation lease through
    the matching ``exec_module`` call, preventing incompatible publication in
    Python's ordinary creation/execution gap.
    """
    registration = framework_registry.resolve(fullname)
    if registration is None or fullname not in registration.roots:
        return _delegate_create(loader, spec)
    key = id(wrapped_spec)
    reader = coordinator.reader()
    lease = publication.lease()
    reader_entered = lease_entered = False
    try:
        reader.__enter__()
        reader_entered = True
        generation = lease.__enter__()
        lease_entered = True
        controlled = getattr(generation.runtime, "mode", RuntimeMode.NONE) is not RuntimeMode.NONE
        adapter = framework_registry.adapter_for(registration) if controlled else None
        admission = publication.admit_status_finalization() if controlled else None
        plan = build_framework_bootstrap_plan(registration, generation.runtime, adapter) if controlled else None
        if controlled:
            plan = _group_plan(registration, admission, plan)
        lifecycle = _Lifecycle(fullname, registration, admission, adapter, plan, reader, lease, controlled)
        if controlled:
            _apply_pre_import(lifecycle)
        module = _delegate_create(loader, spec)
        with _lock:
            if key in _lifecycles:
                raise FrameworkImportSafetyError("repeated framework module creation is unsupported")
            _lifecycles[key] = lifecycle
        return module
    except BaseException:
        _close(reader, lease, reader_entered=reader_entered, lease_entered=lease_entered)
        raise


def exec_module(fullname: str, wrapped_spec: Any, loader: Any, module: Any) -> None:
    """Delegate module execution and finalize root statuses before returning."""
    registration = framework_registry.resolve(fullname)
    if registration is None or fullname not in registration.roots:
        return _delegate_exec(loader, module)
    key = id(wrapped_spec)
    with _lock:
        lifecycle = _lifecycles.pop(key, None)
    if lifecycle is None:
        raise FrameworkImportSafetyError("watched framework reload is unsupported; restart the process")
    try:
        # Even an exception from module code can follow irreversible native
        # initialization, so it is terminal once delegated execution begins.
        lifecycle.executed = True
        _delegate_exec(loader, module)
        if lifecycle.controlled and fullname in lifecycle.registration.roots:
            _finalize(lifecycle)
    except BaseException as exc:
        if lifecycle.controlled and lifecycle.executed:
            publication.fail_status_finalization(lifecycle.admission, exc)
        raise
    finally:
        _close(lifecycle.reader, lifecycle.lease)


def _apply_pre_import(lifecycle: _Lifecycle) -> None:
    """Validate and apply mandatory staged environment before module code."""
    method = getattr(lifecycle.adapter, "validate_before_import", None)
    if method is not None:
        method(lifecycle.plan)
    method = getattr(lifecycle.adapter, "apply_pre_import", None)
    if method is not None:
        method(lifecycle.plan.adapter_plan)
    updates = dict(lifecycle.plan.visibility.env_updates)
    adapter_updates = getattr(lifecycle.plan.adapter_plan, "env_updates", {})
    updates.update(adapter_updates)
    publication.apply_framework_preimport(lifecycle.admission, updates)


def _group_plan(registration: FrameworkRegistration, admission: FrameworkAdmission, plan: Any) -> Any:
    """Retain one equivalent adapter plan for each group control epoch.

    Args:
        registration: Adapter group owning the importing root.
        admission: Same-epoch finalization admission held by that root.
        plan: Newly planned immutable bootstrap controls.

    Returns:
        The group's original equivalent plan.

    Raises:
        FrameworkImportSafetyError: If another root in the group planned
            incompatible controls for the same immutable control epoch.
    """
    key = (admission.control_epoch, registration.name)
    with _lock:
        existing = _group_plans.get(key)
        if existing is None:
            _group_plans[key] = plan
            return plan
        if existing != plan:
            raise FrameworkImportSafetyError("framework group planned incompatible controls in one control epoch")
        return existing


def _finalize(lifecycle: _Lifecycle) -> None:
    """Run post-import controls and publish same-epoch immutable statuses."""
    method = getattr(lifecycle.adapter, "post_import", None)
    outcome = method(lifecycle.plan.adapter_plan, lifecycle.fullname) if method is not None else FrameworkPostResult({"visibility": "visibility-enforced"})
    if not isinstance(outcome, FrameworkPostResult):
        raise FrameworkImportSafetyError("framework post-import hook must return FrameworkPostResult")
    statuses = {f"{lifecycle.registration.name}:{name}": status for name, status in outcome.statuses.items()}
    publication.finalize_statuses(lifecycle.admission, statuses)


def _delegate_create(loader: Any, spec: Any) -> Any:
    """Call an original optional ``create_module`` implementation unchanged."""
    method = getattr(loader, "create_module", None)
    return method(spec) if method is not None else None


def _delegate_exec(loader: Any, module: Any) -> None:
    """Call an original required ``exec_module`` implementation unchanged."""
    method = getattr(loader, "exec_module", None)
    if method is None:
        raise FrameworkImportSafetyError("watched framework loader lacks PEP-451 exec_module")
    method(module)


def _close(reader: AbstractContextManager[Any], lease: AbstractContextManager[Any], *, reader_entered: bool = True, lease_entered: bool = True) -> None:
    """Release admissions in reverse order after all loader outcomes."""
    try:
        if lease_entered:
            lease.__exit__(None, None, None)
    finally:
        if reader_entered:
            reader.__exit__(None, None, None)


__all__ = ["FrameworkImportSafetyError", "create_module", "exec_module", "register_framework"]
