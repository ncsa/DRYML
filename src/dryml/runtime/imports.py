"""Runtime half of the passive framework import interceptor.

This module owns no current session state.  It is entered only by the
standard-library bootstrap loader after a watched import has already selected
its original ``ModuleSpec`` and loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

from .context import active_runtime_bootstrap
from .errors import FrameworkImportSafetyError, PublicationError
from .frameworks import FrameworkPostResult, FrameworkRegistration, framework_registry
from .publication import FrameworkAdmission, MaterializationFence, publication


@dataclass(frozen=True, slots=True)
class _Lifecycle:
    registration: FrameworkRegistration
    admission: FrameworkAdmission | None
    controlled: bool
    result: Any
    adapter: Any | None


def register_framework(registration: FrameworkRegistration) -> None:
    """Register an advanced framework group before it is observed or loaded."""

    framework_registry.register(registration)


def create_module(fullname: str, wrapped_spec: Any, loader: Any, spec: Any) -> Any:
    """Run bounded pure validation and publish a creation fence before return."""

    registration, _ = framework_registry.resolve(fullname)
    if registration is None:
        return _delegate_create(loader, spec)
    key = (fullname, id(wrapped_spec))
    with publication.reader():
        lifecycle = _lifecycle(fullname, registration)
        try:
            _validate_before_creation(lifecycle)
            module = _delegate_create(loader, spec)
            admission = lifecycle.admission
            if admission is not None:
                publication.store_materialization(key, MaterializationFence(admission, id(wrapped_spec), id(module) if module is not None else None))
            return module
        except BaseException as exc:
            _poison(lifecycle, exc)
            raise


def exec_module(fullname: str, wrapped_spec: Any, loader: Any, module: Any) -> None:
    """Require the immutable creation fence, delegate, then finalize once."""

    registration, _ = framework_registry.resolve(fullname)
    if registration is None:
        return _delegate_exec(loader, module)
    key = (fullname, id(wrapped_spec))
    with publication.reader():
        lifecycle = _lifecycle(fullname, registration)
        try:
            fence = publication.materialization(key)
            if fence.module_id is not None and fence.module_id != id(module):
                raise FrameworkImportSafetyError("framework execution received a different module than creation", context={"module": fullname})
            if lifecycle.admission is not None:
                publication.validate_materialization(fence, lifecycle.admission)
            _delegate_exec(loader, module)
            if lifecycle.controlled:
                _finalize(lifecycle, fullname, id(wrapped_spec))
        except BaseException as exc:
            _poison(lifecycle, exc)
            raise


def finalize_helper(framework_name: str, module_name: str | None = None) -> bool:
    """Route helper/bootstrap compatibility calls through the raw finalizer."""

    target = module_name or framework_name
    registration, _ = framework_registry.resolve(target)
    if registration is None:
        return False
    module = sys.modules.get(target)
    spec = getattr(module, "__spec__", None)
    if spec is None:
        return False
    if publication.framework_finalizer_seen(registration.name, target, id(spec)):
        return True
    bootstrap = active_runtime_bootstrap()
    if bootstrap is None or registration.name not in bootstrap.frameworks:
        return False
    with publication.reader():
        lifecycle = _lifecycle(target, registration)
        if not lifecycle.controlled:
            return False
        _finalize(lifecycle, target, id(spec))
    return True


def _lifecycle(fullname: str, registration: FrameworkRegistration) -> _Lifecycle:
    bootstrap = active_runtime_bootstrap()
    if bootstrap is not None:
        controlled = registration.name in bootstrap.frameworks
        result = bootstrap.framework_results.get(registration.name) if controlled else None
    else:
        # A persistent facade session publishes immutable adapter plans on the
        # sole runtime generation.  Scoped bootstrap state remains an advanced
        # override, not a second session authority.
        generation = publication.current()
        results = generation.metadata.get("framework_results", {})
        controlled = bool(generation.metadata.get("session_active")) and registration.name in results
        result = results.get(registration.name) if controlled else None
    adapter = framework_registry.adapter_for(registration) if controlled else None
    resolved, revision = framework_registry.resolve(fullname)
    if resolved is not registration:
        raise FrameworkImportSafetyError("framework registration changed during loader lifecycle", context={"module": fullname})
    fingerprint = repr((registration.name, registration.roots, result, registration.capabilities))
    admission = publication.admit_framework(registration.name, fullname, fingerprint, revision)
    return _Lifecycle(registration, admission, controlled, result, adapter)


def _validate_before_creation(lifecycle: _Lifecycle) -> None:
    if not lifecycle.controlled:
        return
    non_idempotent = bool(getattr(lifecycle.adapter, "pre_import_non_idempotent", False))
    if non_idempotent:
        publication.claim_framework_pre_stage(lifecycle.admission)
    validator = getattr(lifecycle.adapter, "validate_before_import", None)
    if validator is not None:
        validator(lifecycle.result)
    publication.publish_framework_pre_stage(lifecycle.admission)
    if non_idempotent:
        publication.complete_framework_pre_stage(lifecycle.admission)


def _finalize(lifecycle: _Lifecycle, fullname: str, spec_id: int) -> None:
    if lifecycle.admission is None:
        return
    current_registration, current_revision = framework_registry.resolve(fullname)
    if current_registration is not lifecycle.registration or current_revision != lifecycle.admission.registry_revision:
        raise FrameworkImportSafetyError("framework registration changed before post-import finalization", context={"module": fullname})
    if not publication.claim_framework_finalizer(lifecycle.admission, spec_id):
        return
    post = _post(lifecycle.adapter, lifecycle.result, fullname)
    statuses = {f"{lifecycle.registration.name}:{fullname}:{control}": state for control, state in post.statuses.items()}
    try:
        publication.finalize_framework(lifecycle.admission, statuses)
    except BaseException as exc:
        # A post-status write is after framework execution. It is no safer to
        # continue than any other uncertain framework outcome.
        publication.fail_framework(lifecycle.admission, exc)
        raise
    publication.complete_framework_finalizer(lifecycle.admission, spec_id)


def _post(adapter: Any, result: Any, fullname: str) -> FrameworkPostResult:
    method = getattr(adapter, "post_import", None)
    if method is not None:
        return method(result, fullname)
    # Existing advanced bootstrap adapters retain their old hook contract.
    adapter.apply_post_import(result)
    return FrameworkPostResult(fullname, {"visibility": "visibility-enforced"})


def _poison(lifecycle: _Lifecycle, exc: BaseException) -> None:
    # Protocol/fence rejection occurs before user or framework code and does not
    # leave native state to recover.  A controlled delegated/hook failure does.
    if lifecycle.controlled and not isinstance(exc, PublicationError):
        publication.fail_framework(lifecycle.admission, exc)


def _delegate_create(loader: Any, spec: Any) -> Any:
    create = getattr(loader, "create_module", None)
    return create(spec) if create is not None else None


def _delegate_exec(loader: Any, module: Any) -> None:
    execute = getattr(loader, "exec_module", None)
    if execute is None:
        raise FrameworkImportSafetyError("framework loader does not support PEP-451 execution")
    execute(module)


__all__ = ["create_module", "exec_module", "finalize_helper", "register_framework"]
