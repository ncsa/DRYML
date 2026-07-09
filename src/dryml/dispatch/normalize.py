"""Normalize Python-shaped dispatch inputs into OperationSpec payloads.

This module is an internal dispatch/developer API. It classifies user-facing
targets, builds the existing operation spec IR, and records metadata for later
requirement analysis. It deliberately does not resolve requirements, choose
candidate environments/worlds, or run probes.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from dryml.code import CodeTargetSpec, target_from_callable, target_from_definition_method
from dryml.core2.definition import ConcreteDefinition, Definition
from dryml.core2.object import Object
from dryml.core2.repo import Repo
from dryml.formats.refs import format_cdef_id
from dryml.operations import (
    OPERATION_KINDS,
    OPERATION_SCHEMA,
    OperationSpecError,
    attach_operation_id,
    make_function_call_spec,
    make_method_call_spec,
    validate_operation_spec,
)

from .errors import DispatchPlanningError
from .operations import PickledCallable, write_pickled_callable


NORMALIZATION_METADATA_VERSION = 1
_RESERVED_NORMALIZATION_KEYS = frozenset(
    {
        "dryml.dispatch.normalized",
        "dryml.dispatch.normalization_version",
        "dryml.dispatch.user_target_kind",
        "dryml.dispatch.transport",
        "dryml.code_target",
    }
)


@dataclass(frozen=True, slots=True)
class NormalizedDispatchTarget:
    """Normalized dispatch target consumed by the current planner.

    Args:
        operation_spec: Canonical operation spec mapping to feed into the
            existing dispatch pipeline.
        launch: Launch-only transport data for the backend. Live paths and
            pickle files belong here, not in operation metadata.
        code_target: JSON-serializable target metadata for later requirement
            analysis.
        live_annotation_targets: In-memory Python objects that later planning
            stages may inspect before serialization. These are never written to
            operation metadata.
        subject_class: Resolved method-dispatch subject class when available.
        method_name: User-requested method name for method dispatch.
        transport: Normalized transport label.
        diagnostics: Non-blocking normalization diagnostics.
    """

    operation_spec: Mapping[str, Any]
    launch: Mapping[str, Any] = field(default_factory=dict)
    code_target: CodeTargetSpec | None = None
    live_annotation_targets: tuple[Any, ...] = ()
    subject_class: type | None = None
    method_name: str | None = None
    transport: str = "operation_spec"
    diagnostics: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_spec", dict(self.operation_spec))
        object.__setattr__(self, "launch", dict(self.launch))
        object.__setattr__(self, "live_annotation_targets", tuple(self.live_annotation_targets))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True, slots=True)
class _Importability:
    path: str | None
    reason: str | None
    module: str | None = None
    qualname: str | None = None


def normalize_user_operation(
    operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable | Any,
    method_name: str | None = None,
    *,
    store: Any | None = None,
    args: tuple[Any, ...] | list[Any] | None = (),
    kwargs: Mapping[str, Any] | None = None,
    allow_pickle: bool = False,
) -> NormalizedDispatchTarget:
    """Normalize a user dispatch target into the existing OperationSpec IR.

    Args:
        operation: Explicit OperationSpec, callable, PickledCallable, CDef /
            Definition, or DRYML Object instance.
        method_name: Optional method name for DRYML Definition/CDef/Object
            method dispatch.
        store: Store used to confirm or create method-call subject persistence.
        args: Explicit user arguments for the target call.
        kwargs: Explicit user keyword arguments for the target call.
        allow_pickle: Whether non-importable callables may use the existing
            ``pickle_small`` same-environment transport.

    Returns:
        A normalized target with operation, launch, and code-target metadata.

    Raises:
        DispatchPlanningError: If the target cannot be normalized safely.
    """

    norm_args = _normalize_args(args)
    norm_kwargs = _normalize_kwargs(kwargs)
    norm_method = _validate_optional_method_name(method_name)

    if is_definition_or_cdef(operation):
        return normalize_definition_method_operation(operation, norm_method, store=store, args=norm_args, kwargs=norm_kwargs)
    if is_dryml_object_instance(operation):
        return normalize_object_method_operation(operation, norm_method, store=store, args=norm_args, kwargs=norm_kwargs)
    if looks_like_operation_spec(operation):
        if norm_method is not None:
            raise DispatchPlanningError("method_name cannot be supplied with an explicit OperationSpec")
        return normalize_existing_operation_spec(operation, args=norm_args, kwargs=norm_kwargs)
    if isinstance(operation, Mapping):
        _raise_invalid_operation_mapping(operation)
    if callable(operation) or isinstance(operation, PickledCallable):
        if norm_method is not None:
            raise DispatchPlanningError("method_name is only valid for DRYML Definition/CDef/Object targets")
        return normalize_callable_operation(operation, args=norm_args, kwargs=norm_kwargs, allow_pickle=allow_pickle)
    raise DispatchPlanningError(
        "unsupported dispatch target; expected callable, OperationSpec, or DRYML Definition/CDef/Object plus method name",
        context={"type": type(operation).__name__},
    )


def is_definition_or_cdef(value: Any) -> bool:
    """Return whether *value* is a DRYML Definition or ConcreteDefinition."""

    return isinstance(value, (Definition, ConcreteDefinition))


def is_dryml_object_instance(value: Any) -> bool:
    """Return whether *value* is a live DRYML Object instance."""

    return isinstance(value, Object)


def looks_like_operation_spec(value: Any) -> bool:
    """Return whether *value* validates as an OperationSpec."""

    if not isinstance(value, Mapping):
        return False
    try:
        validate_operation_spec(value)
    except OperationSpecError:
        return False
    return True


def import_path_for_callable(func: Callable[..., Any]) -> str | None:
    """Return ``module:qualname`` when *func* resolves by import identity."""

    return _callable_importability(func).path


def normalize_existing_operation_spec(
    operation: Mapping[str, Any], *, args: tuple[Any, ...] = (), kwargs: Mapping[str, Any] | None = None
) -> NormalizedDispatchTarget:
    """Preserve an explicit OperationSpec while adding safe metadata."""

    if args or kwargs:
        raise DispatchPlanningError("explicit OperationSpec already contains arguments; do not also pass args/kwargs")
    try:
        op = attach_operation_id(validate_operation_spec(operation))
    except OperationSpecError as exc:
        raise DispatchPlanningError(str(exc), context=exc.context) from exc
    code_target = _infer_code_target(op)
    op = _attach_normalization_metadata(op, user_target_kind="operation_spec", transport="operation_spec", code_target=code_target, preserve_existing=True)
    return NormalizedDispatchTarget(op, {}, code_target, transport="operation_spec")


def normalize_callable_operation(
    operation: Callable[..., Any] | PickledCallable,
    *,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    allow_pickle: bool = False,
) -> NormalizedDispatchTarget:
    """Normalize a Python callable through import-path or pickle transport."""

    explicit_pickle = isinstance(operation, PickledCallable)
    func = operation.callable if explicit_pickle else operation
    if _is_bound_instance_method(func) and not explicit_pickle:
        raise DispatchPlanningError(
            "bound instance method dispatch is not supported for this target; "
            "use dispatch.submit(cdef, \"method\", ...), or PickledCallable(...) "
            "for explicit same-environment pickle transport"
        )

    importability = _callable_importability(func)
    if importability.path is not None and not explicit_pickle:
        code_target = target_from_callable(func).spec
        metadata_target = _code_target_with_metadata(
            code_target,
            {"dispatch_target": "callable", "importability": "importable", "transport": "import_path"},
            import_path=importability.path,
        )
        try:
            op = attach_operation_id(make_function_call_spec(importability.path, args=args, kwargs=kwargs, metadata=_normalization_metadata("callable", "import_path", metadata_target)))
        except OperationSpecError as exc:
            raise DispatchPlanningError(str(exc), context=exc.context) from exc
        return NormalizedDispatchTarget(
            op,
            {"call_transport": "import_ref", "portable": True},
            metadata_target,
            live_annotation_targets=(func,),
            transport="import_path",
        )

    if not allow_pickle and not explicit_pickle:
        reason = importability.reason or "not_importable"
        raise DispatchPlanningError(
            "callable is not importable; pass allow_pickle=True or define it at module scope",
            context={"reason": reason, "module": importability.module, "qualname": importability.qualname},
        )
    return _normalize_pickled_callable(func, args=args, kwargs=kwargs or {}, importability=importability)


def normalize_definition_method_operation(
    subject: Definition | ConcreteDefinition,
    method_name: str | None,
    *,
    store: Any | None = None,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> NormalizedDispatchTarget:
    """Normalize Definition/CDef plus method name into a method_call spec."""

    method = _require_method_name(method_name)
    if store is None:
        raise DispatchPlanningError("store is required to dispatch this DRYML method target")
    cdef = subject.concretize(repo=Repo(stores=[store])) if isinstance(subject, Definition) else subject
    if not isinstance(cdef, ConcreteDefinition):
        raise DispatchPlanningError("could not persist/reference dispatch subject for method call")
    if not _store_has_cdef(store, cdef):
        raise DispatchPlanningError("store is required to dispatch this DRYML method target; subject CDef is not present in the store")
    return _method_target_from_cdef(cdef, method, store=store, args=args, kwargs=kwargs or {}, user_target_kind="definition_method")


def normalize_object_method_operation(
    subject: Object,
    method_name: str | None,
    *,
    store: Any | None = None,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> NormalizedDispatchTarget:
    """Persist a live DRYML object and normalize it into a method_call spec."""

    method = _require_method_name(method_name)
    if store is None:
        raise DispatchPlanningError("store is required to dispatch this DRYML method target")
    repo = Repo(stores=[store])
    repo.save(subject, store=store, record_policy="none")
    return _method_target_from_cdef(subject.definition, method, store=store, args=args, kwargs=kwargs or {}, user_target_kind="object_method", subject_class=type(subject))


def _method_target_from_cdef(
    cdef: ConcreteDefinition,
    method_name: str,
    *,
    store: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    user_target_kind: str,
    subject_class: type | None = None,
) -> NormalizedDispatchTarget:
    cdef_id = format_cdef_id(cdef.stable_hash())
    cls = subject_class if subject_class is not None else cdef.cls if isinstance(cdef.cls, type) else None
    code_target = target_from_definition_method(cdef_id, cls, method_name).spec
    code_target = _code_target_with_metadata(code_target, {"dispatch_target": "definition_method", "transport": "method_call"}, subject_ref=cdef_id, method_name=method_name)
    try:
        op = attach_operation_id(make_method_call_spec(cdef_id, method_name, args=args, kwargs=kwargs, metadata=_normalization_metadata(user_target_kind, "method_call", code_target)))
    except OperationSpecError as exc:
        raise DispatchPlanningError(str(exc), context=exc.context) from exc
    live_target = getattr(cls, method_name, None) if cls is not None and hasattr(cls, method_name) else None
    return NormalizedDispatchTarget(
        op,
        {"call_transport": "method_call", "portable": True},
        code_target,
        live_annotation_targets=() if live_target is None else (live_target,),
        subject_class=cls,
        method_name=method_name,
        transport="method_call",
    )


def _normalize_pickled_callable(
    func: Callable[..., Any], *, args: tuple[Any, ...], kwargs: Mapping[str, Any], importability: _Importability
) -> NormalizedDispatchTarget:
    work_dir = tempfile.mkdtemp(prefix="dryml-dispatch-pickle-")
    pickle_path = os.path.join(work_dir, "callable.pkl")
    write_pickled_callable(func, pickle_path)
    with open(pickle_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    identity_marker = {"$literal": f"dryml.pickled_callable.sha256:{digest}"}
    base_target = target_from_callable(func).spec
    code_target = _code_target_with_metadata(
        base_target,
        {
            "dispatch_target": "callable",
            "pickle_transport": "pickle_small",
            "transport_restrictions": ["same_environment_only"],
            "importability": "not_importable" if importability.path is None else "explicit_pickle",
            "importability_reason": importability.reason,
        },
    )
    callable_metadata = {
        "module": getattr(func, "__module__", None),
        "qualname": getattr(func, "__qualname__", None),
        "importability_reason": importability.reason,
    }
    try:
        op = attach_operation_id(
            make_function_call_spec(
                "dryml.dispatch.operations:import_function",
                args=[*args, identity_marker],
                kwargs=kwargs,
                metadata=_normalization_metadata("callable", "pickle_small", code_target),
            )
        )
    except OperationSpecError as exc:
        raise DispatchPlanningError(str(exc), context=exc.context) from exc
    launch = {
        "call_transport": "pickle_small",
        "pickle_path": pickle_path,
        "identity_arg_count": len(args),
        "pickle_sha256": digest,
        "portable": False,
        "same_environment_only": True,
        "cleanup_paths": [work_dir],
        "callable_metadata": callable_metadata,
        "transport_restrictions": ["same_environment_only"],
    }
    return NormalizedDispatchTarget(op, launch, code_target, live_annotation_targets=(func,), transport="pickle_small")


def _normalize_args(args: tuple[Any, ...] | list[Any] | None) -> tuple[Any, ...]:
    if args is None:
        return ()
    if not isinstance(args, (tuple, list)):
        raise DispatchPlanningError("dispatch args must be a tuple or list", context={"type": type(args).__name__})
    return tuple(args)


def _normalize_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    if kwargs is None:
        return {}
    if not isinstance(kwargs, Mapping):
        raise DispatchPlanningError("dispatch kwargs must be a mapping", context={"type": type(kwargs).__name__})
    return dict(kwargs)


def _validate_optional_method_name(method_name: str | None) -> str | None:
    if method_name is None:
        return None
    if not isinstance(method_name, str):
        raise DispatchPlanningError("method_name must be a string", context={"type": type(method_name).__name__})
    if not method_name:
        raise DispatchPlanningError("method_name must not be empty")
    parts = method_name.split(".")
    if any(not part.isidentifier() for part in parts):
        raise DispatchPlanningError("method_name must be a dotted Python attribute path", context={"method_name": method_name})
    return method_name


def _require_method_name(method_name: str | None) -> str:
    if method_name is None:
        raise DispatchPlanningError("method_name is required for DRYML Definition/CDef/Object dispatch; use dispatch.submit(subject, \"method\", ...)")
    return method_name


def _callable_importability(func: Callable[..., Any]) -> _Importability:
    if getattr(func, "__name__", None) == "<lambda>":
        return _Importability(None, "lambda", getattr(func, "__module__", None), getattr(func, "__qualname__", None))
    module_name = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if not module_name or not qualname:
        return _Importability(None, "missing_module_or_qualname", module_name, qualname)
    if module_name == "__main__":
        return _Importability(None, "__main__", module_name, qualname)
    if "<locals>" in qualname:
        return _Importability(None, "local_function", module_name, qualname)
    try:
        module = importlib.import_module(module_name)
        resolved = module
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except Exception:
        return _Importability(None, "import_mismatch", module_name, qualname)
    if resolved is not func:
        return _Importability(None, "import_mismatch", module_name, qualname)
    return _Importability(f"{module_name}:{qualname}", None, module_name, qualname)


def _is_bound_instance_method(func: Any) -> bool:
    return inspect.ismethod(func) and getattr(func, "__self__", None) is not None and not isinstance(getattr(func, "__self__"), type)


def _store_has_cdef(store: Any, cdef: ConcreteDefinition) -> bool:
    try:
        return bool(store.has(cdef))
    except Exception:
        return False


def _infer_code_target(op: Mapping[str, Any]) -> CodeTargetSpec | None:
    payload = op.get("payload") if isinstance(op.get("payload"), Mapping) else {}
    if op.get("kind") == "function_call" and isinstance(payload.get("function"), str):
        return CodeTargetSpec("import_path", import_path=payload["function"], metadata={"dispatch_target": "function"})
    if op.get("kind") == "method_call" and isinstance(payload.get("subject"), str) and isinstance(payload.get("method"), str):
        return CodeTargetSpec("definition_method", subject_ref=payload["subject"], method_name=payload["method"], metadata={"dispatch_target": "definition_method"})
    return None


def _code_target_with_metadata(
    target: CodeTargetSpec,
    metadata: Mapping[str, Any],
    *,
    import_path: str | None = None,
    subject_ref: str | None = None,
    method_name: str | None = None,
) -> CodeTargetSpec:
    merged = dict(target.metadata)
    merged.update(metadata)
    return CodeTargetSpec(
        target.kind,
        import_path=import_path if import_path is not None else target.import_path,
        source_spec=target.source_spec,
        method_name=method_name if method_name is not None else target.method_name,
        subject_ref=subject_ref if subject_ref is not None else target.subject_ref,
        metadata=merged,
    )


def _normalization_metadata(user_target_kind: str, transport: str, code_target: CodeTargetSpec | None) -> dict[str, Any]:
    return {
        "dryml.dispatch.normalized": True,
        "dryml.dispatch.normalization_version": NORMALIZATION_METADATA_VERSION,
        "dryml.dispatch.user_target_kind": user_target_kind,
        "dryml.dispatch.transport": transport,
        "dryml.code_target": code_target.to_data() if code_target is not None else None,
    }


def _attach_normalization_metadata(
    op: Mapping[str, Any], *, user_target_kind: str, transport: str, code_target: CodeTargetSpec | None, preserve_existing: bool
) -> dict[str, Any]:
    result = dict(op)
    metadata = dict(result.get("metadata") or {})
    update = _normalization_metadata(user_target_kind, transport, code_target)
    for key in _RESERVED_NORMALIZATION_KEYS:
        metadata.pop(key, None)
    metadata.update(update)
    result["metadata"] = metadata
    try:
        return attach_operation_id(result)
    except OperationSpecError as exc:
        raise DispatchPlanningError(str(exc), context=exc.context) from exc


def _raise_invalid_operation_mapping(operation: Mapping[str, Any]) -> None:
    if operation.get("schema") == OPERATION_SCHEMA or operation.get("kind") in OPERATION_KINDS or "payload" in operation:
        try:
            validate_operation_spec(operation)
        except OperationSpecError as exc:
            raise DispatchPlanningError(str(exc), context=exc.context) from exc
    raise DispatchPlanningError(
        "unsupported dispatch target; expected callable, OperationSpec, or DRYML Definition/CDef/Object plus method name",
        context={"type": type(operation).__name__},
    )


__all__ = [
    "NORMALIZATION_METADATA_VERSION",
    "NormalizedDispatchTarget",
    "import_path_for_callable",
    "is_definition_or_cdef",
    "is_dryml_object_instance",
    "looks_like_operation_spec",
    "normalize_callable_operation",
    "normalize_definition_method_operation",
    "normalize_existing_operation_spec",
    "normalize_object_method_operation",
    "normalize_user_operation",
]
