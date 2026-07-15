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
import shutil
import tempfile
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from dryml.code import CodeTargetSpec, target_from_callable, target_from_definition_method
from dryml.core2.definition import ConcreteDefinition, Definition
from dryml.core2.object import Object
from dryml.core2.repo import Repo
from dryml.core2.symbol import resolve_symbol
from dryml.core2.utils.general import pickle_load
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
# Planning evidence is request/run-specific.  Operation sidecars are immutable
# for their whole canonical bytes even though their IDs exclude metadata, so an
# explicit OperationSpec must never carry caller-provided planning evidence.
_RESERVED_PLANNING_KEYS = frozenset(
    {
        "dryml.dispatch.planning_version", "dryml.code_analysis", "dryml.code_probe",
        "dryml.requirements", "dryml.requirement_sources", "dryml.environment_selection",
        "dryml.environment_probe", "dryml.environment_check", "dryml.environment_resolution",
        "dryml.world_selection", "dryml.world_check", "dryml.world_synthesis",
        "dryml.local_inventory", "dryml.world_allocation", "dryml.runtime_selection",
        "dryml.runtime_check", "dryml.requirement_policy", "dryml.runtime_enforcement",
        "dryml.dispatch.launchable", "dryml.dispatch.diagnostics", "dryml.dispatch.dynamic_trace",
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
    definition_target: Definition | ConcreteDefinition | None = None
    # Trace-only state is deliberately private to one planning request.  It is
    # never copied into the operation spec, launch data, or persisted metadata.
    trace_live_target: Callable[..., Any] | None = None
    trace_store: Any | None = None
    trace_cdef_side_table: Mapping[str, tuple[ConcreteDefinition, ...]] = field(default_factory=dict)
    trace_cdef_positions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_spec", dict(self.operation_spec))
        object.__setattr__(self, "launch", dict(self.launch))
        object.__setattr__(self, "live_annotation_targets", tuple(self.live_annotation_targets))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "trace_cdef_side_table", {key: tuple(value) for key, value in self.trace_cdef_side_table.items()})
        object.__setattr__(self, "trace_cdef_positions", tuple(self.trace_cdef_positions))


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
    persist_object: bool = True,
    trace_enabled: bool = False,
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

    if trace_enabled:
        norm_args, norm_kwargs, trace_cdefs, trace_positions = _trace_normalize_invocation(args, kwargs, store=store)
    else:
        norm_args = _normalize_args(args)
        norm_kwargs = _normalize_kwargs(kwargs)
        trace_cdefs = {}
        trace_positions = ()
    norm_method = _validate_optional_method_name(method_name)

    if is_definition_or_cdef(operation):
        return normalize_definition_method_operation(operation, norm_method, store=store, args=norm_args, kwargs=norm_kwargs)
    if is_dryml_object_instance(operation):
        return normalize_object_method_operation(operation, norm_method, store=store, args=norm_args, kwargs=norm_kwargs, persist=persist_object)
    if looks_like_operation_spec(operation):
        if norm_method is not None:
            raise DispatchPlanningError("method_name cannot be supplied with an explicit OperationSpec")
        return normalize_existing_operation_spec(operation, store=store, args=norm_args, kwargs=norm_kwargs)
    if isinstance(operation, Mapping):
        _raise_invalid_operation_mapping(operation)
    # Requested tracing has a narrower target contract than ordinary dispatch.
    # Check it before generic callable normalization reads callable attributes,
    # resolves importability, or creates a pickle transport.  In particular,
    # callable instances can expose arbitrary attribute hooks even though they
    # are not traceable 9C targets.
    if trace_enabled:
        trace_candidate = operation.callable if isinstance(operation, PickledCallable) else operation
        if not _is_trace_eligible_function(trace_candidate):
            raise DispatchPlanningError(
                "dynamic tracing requires a live exact synchronous Python function",
                context={"dynamic_trace": "unsupported_target"},
            )
    if callable(operation) or isinstance(operation, PickledCallable):
        if norm_method is not None:
            raise DispatchPlanningError("method_name is only valid for DRYML Definition/CDef/Object targets")
        return normalize_callable_operation(
            operation,
            args=norm_args,
            kwargs=norm_kwargs,
            allow_pickle=allow_pickle,
            trace_cdef_side_table=trace_cdefs,
            trace_cdef_positions=trace_positions,
            trace_store=store,
        )
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
    operation: Mapping[str, Any], *, store: Any | None = None, args: tuple[Any, ...] = (), kwargs: Mapping[str, Any] | None = None
) -> NormalizedDispatchTarget:
    """Preserve an explicit OperationSpec while adding safe metadata."""

    if args or kwargs:
        raise DispatchPlanningError("explicit OperationSpec already contains arguments; do not also pass args/kwargs")
    try:
        op = attach_operation_id(validate_operation_spec(operation))
    except OperationSpecError as exc:
        raise DispatchPlanningError(str(exc), context=exc.context) from exc
    definition_target = _definition_target_for_operation(op, store)
    code_target = _infer_code_target(op, subject_class=_definition_class(definition_target))
    op = _attach_normalization_metadata(op, user_target_kind="operation_spec", transport="operation_spec", code_target=code_target, preserve_existing=True)
    payload = op.get("payload") if isinstance(op.get("payload"), Mapping) else {}
    method_name = payload.get("method") if op.get("kind") == "method_call" and isinstance(payload.get("method"), str) else None
    return NormalizedDispatchTarget(
        op,
        {},
        code_target,
        subject_class=_definition_class(definition_target),
        method_name=method_name,
        transport="operation_spec",
        definition_target=definition_target,
        trace_store=store,
    )


def normalize_callable_operation(
    operation: Callable[..., Any] | PickledCallable,
    *,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    allow_pickle: bool = False,
    trace_cdef_side_table: Mapping[str, tuple[ConcreteDefinition, ...]] | None = None,
    trace_cdef_positions: tuple[tuple[str, str], ...] = (),
    trace_store: Any | None = None,
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
            trace_live_target=func,
            trace_store=trace_store,
            trace_cdef_side_table={} if trace_cdef_side_table is None else trace_cdef_side_table,
            trace_cdef_positions=trace_cdef_positions,
        )

    if not allow_pickle and not explicit_pickle:
        reason = importability.reason or "not_importable"
        raise DispatchPlanningError(
            "callable is not importable; pass allow_pickle=True or define it at module scope",
            context={"reason": reason, "module": importability.module, "qualname": importability.qualname},
        )
    normalized = _normalize_pickled_callable(func, args=args, kwargs=kwargs or {}, importability=importability)
    return NormalizedDispatchTarget(
        normalized.operation_spec,
        normalized.launch,
        normalized.code_target,
        normalized.live_annotation_targets,
        normalized.subject_class,
        normalized.method_name,
        normalized.transport,
        normalized.diagnostics,
        normalized.definition_target,
        func,
        trace_store,
        {} if trace_cdef_side_table is None else trace_cdef_side_table,
        trace_cdef_positions,
    )


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
    persist: bool = True,
) -> NormalizedDispatchTarget:
    """Normalize a live DRYML object into a method_call spec.

    ``persist=False`` is used by non-mutating explanation requests and requires
    the object's existing definition to already be available in the store.
    """

    method = _require_method_name(method_name)
    if store is None:
        raise DispatchPlanningError("store is required to dispatch this DRYML method target")
    if persist:
        Repo(stores=[store]).save(subject, store=store, record_policy="none")
    elif not _store_has_cdef(store, subject.definition):
        raise DispatchPlanningError("explain requires an already-stored DRYML object; use its CDef or save it before explaining")
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
    cls = subject_class if subject_class is not None else _definition_class(cdef)
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
        definition_target=cdef,
    )


def _normalize_pickled_callable(
    func: Callable[..., Any], *, args: tuple[Any, ...], kwargs: Mapping[str, Any], importability: _Importability
) -> NormalizedDispatchTarget:
    work_dir = tempfile.mkdtemp(prefix="dryml-dispatch-pickle-")
    pickle_path = os.path.join(work_dir, "callable.pkl")
    try:
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
        op = attach_operation_id(
            make_function_call_spec(
                "dryml.dispatch.operations:import_function",
                args=[*args, identity_marker],
                kwargs=kwargs,
                metadata=_normalization_metadata("callable", "pickle_small", code_target),
            )
        )
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
    except OperationSpecError as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise DispatchPlanningError(str(exc), context=exc.context) from exc
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def _trace_normalize_invocation(
    args: Any,
    kwargs: Any,
    *,
    store: Any | None,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, tuple[ConcreteDefinition, ...]], tuple[tuple[str, str], ...]]:
    """Return the canonical trace grammar and private live-CDef side table.

    The normal operation grammar intentionally accepts Python-shaped lists and
    mappings.  Requested tracing is narrower: it needs an exact, reproducible
    worker call before trusted code can run.  Concrete definitions become raw
    worker CDef references while the corresponding live values remain only in
    this request-local table.
    """

    if type(args) is not tuple:
        raise DispatchPlanningError("dynamic tracing requires args to be an exact tuple")
    if kwargs is None:
        kwargs = {}
    if type(kwargs) is not dict:
        raise DispatchPlanningError("dynamic tracing requires kwargs to be an exact dict")
    if any(type(key) is not str for key in kwargs):
        raise DispatchPlanningError("dynamic tracing requires exact string kwargs keys")

    side_table: dict[str, list[ConcreteDefinition]] = {}
    positions: list[tuple[str, str]] = []
    seen_containers: set[int] = set()

    def convert(value: Any, path: str) -> Any:
        if isinstance(value, Definition) and not isinstance(value, ConcreteDefinition):
            raise DispatchPlanningError("dynamic tracing does not support plain Definition arguments")
        if isinstance(value, ConcreteDefinition):
            if store is None or not _store_has_cdef(store, value):
                raise DispatchPlanningError("dynamic tracing requires every ConcreteDefinition argument to be present in the store")
            cdef_id = format_cdef_id(value.stable_hash())
            side_table.setdefault(cdef_id, []).append(value)
            positions.append((path, cdef_id))
            return cdef_id
        if type(value) in {list, tuple, dict}:
            identity = id(value)
            if identity in seen_containers:
                raise DispatchPlanningError("dynamic tracing does not support aliased or cyclic invocation containers")
            seen_containers.add(identity)
            if type(value) is dict:
                if any(type(key) is not str for key in value):
                    raise DispatchPlanningError("dynamic tracing requires exact string mapping keys")
                return {key: convert(item, f"{path}/{key}") for key, item in value.items()}
            return [convert(item, f"{path}/{index}") for index, item in enumerate(value)]
        return value

    converted_args = tuple(convert(value, f"args/{index}") for index, value in enumerate(args))
    converted_kwargs = {key: convert(value, f"kwargs/{key}") for key, value in kwargs.items()}
    return converted_args, converted_kwargs, {key: tuple(values) for key, values in side_table.items()}, tuple(positions)


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


def _is_trace_eligible_function(value: Any) -> bool:
    """Return whether *value* is an exact synchronous Python trace target.

    This deliberately reads function implementation state only after an exact
    type check.  It must not inspect user-defined callable, descriptor, or
    wrapper attributes while deciding whether an opt-in trace is supported.
    """

    if type(value) is not types.FunctionType:
        return False
    return not (
        inspect.iscoroutinefunction(value)
        or inspect.isasyncgenfunction(value)
        or inspect.isgeneratorfunction(value)
    )


def _is_bound_instance_method(func: Any) -> bool:
    return inspect.ismethod(func) and getattr(func, "__self__", None) is not None and not isinstance(getattr(func, "__self__"), type)


def _store_has_cdef(store: Any, cdef: ConcreteDefinition) -> bool:
    try:
        return bool(store.has(cdef))
    except Exception:
        return False


def _infer_code_target(op: Mapping[str, Any], *, subject_class: type | None = None) -> CodeTargetSpec | None:
    payload = op.get("payload") if isinstance(op.get("payload"), Mapping) else {}
    if op.get("kind") == "function_call" and isinstance(payload.get("function"), str):
        return CodeTargetSpec("import_path", import_path=payload["function"], metadata={"dispatch_target": "function"})
    if op.get("kind") == "method_call" and isinstance(payload.get("subject"), str) and isinstance(payload.get("method"), str):
        return target_from_definition_method(payload["subject"], subject_class, payload["method"]).spec
    return None


def _definition_target_for_operation(operation: Mapping[str, Any], store: Any | None) -> ConcreteDefinition | None:
    """Load a stored method subject definition without materializing its object."""

    if store is None or operation.get("kind") != "method_call":
        return None
    payload = operation.get("payload") if isinstance(operation.get("payload"), Mapping) else {}
    subject = payload.get("subject")
    if not isinstance(subject, str):
        return None
    try:
        path = os.path.join(store.object_dir_for_cdef_id(subject), "def.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        definition = pickle_load(path)
    except Exception:
        definition = None
    if isinstance(definition, ConcreteDefinition):
        return definition
    try:
        for candidate in store.hydrate_index():
            if isinstance(candidate, ConcreteDefinition) and format_cdef_id(candidate.stable_hash()) == subject:
                return candidate
    except Exception:
        pass
    return None


def _definition_class(definition: ConcreteDefinition | Definition | None) -> type | None:
    cls = getattr(definition, "cls", None)
    if isinstance(cls, type):
        return cls
    try:
        resolved = resolve_symbol(cls)
    except Exception:
        return None
    return resolved if isinstance(resolved, type) else None


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
    for key in _RESERVED_NORMALIZATION_KEYS | _RESERVED_PLANNING_KEYS:
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
