from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from .facts import DiagnosticFact, json_compatible


@dataclass(frozen=True, slots=True)
class CodeTargetSpec:
    """Serializable description of a Python or DRYML code target.

    Args:
        kind: Descriptive target kind such as ``"function"`` or ``"class"``.
        import_path: Optional ``module:qualname`` reference.
        source_spec: Optional serialized source-backed reference.
        method_name: Optional method name for method-like targets.
        subject_ref: Optional serializable subject reference for definition methods.
        metadata: JSON-compatible auxiliary metadata.
    """

    kind: str
    import_path: str | None = None
    source_spec: Mapping[str, Any] | None = None
    method_name: str | None = None
    subject_ref: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_spec", json_compatible(self.source_spec) if self.source_spec is not None else None)
        object.__setattr__(self, "metadata", json_compatible(self.metadata))

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this target spec."""

        return {
            "kind": self.kind,
            "import_path": self.import_path,
            "source_spec": json_compatible(self.source_spec) if self.source_spec is not None else None,
            "method_name": self.method_name,
            "subject_ref": self.subject_ref,
            "metadata": json_compatible(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CodeTargetSpec":
        """Build a target spec from JSON-compatible data."""

        return cls(
            kind=data.get("kind", "unknown"),
            import_path=data.get("import_path"),
            source_spec=data.get("source_spec"),
            method_name=data.get("method_name"),
            subject_ref=data.get("subject_ref"),
            metadata=data.get("metadata") or {},
        )

    @classmethod
    def from_import_path(cls, path: str) -> "CodeTargetSpec":
        """Build an import-path target spec without importing the target."""

        return cls(kind="import_path", import_path=path)


@dataclass(frozen=True, slots=True)
class CodeTarget:
    """Local analysis wrapper that may include live Python objects.

    Args:
        spec: Serializable target description.
        obj: Optional live object used by local analyzers.
        owner: Optional owner class for method-like targets.
        attribute_name: Optional attribute name on the owner.
        raw_descriptor: Optional raw descriptor from a class dictionary.
        unwrapped: Optional unwrapped callable object.
        metadata: Local metadata for analyzers.
        diagnostics: Diagnostics produced while normalizing the target.
    """

    spec: CodeTargetSpec
    obj: Any | None = None
    owner: type | None = None
    attribute_name: str | None = None
    raw_descriptor: Any | None = None
    unwrapped: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[DiagnosticFact, ...] = ()


def normalize_target(
    target: Any,
    *,
    method_name: str | None = None,
    subject_ref: str | None = None,
    allow_import: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> CodeTarget:
    """Normalize an arbitrary target into a :class:`CodeTarget`.

    Args:
        target: Live object, import path string, :class:`CodeTarget`, or spec.
        method_name: Optional method name for representational method targets.
        subject_ref: Optional serializable subject reference.
        allow_import: Whether import-path strings may be resolved.
        metadata: Optional serializable metadata to add to the target spec.

    Returns:
        A local target wrapper. Import failures are captured as diagnostics.
    """

    if isinstance(target, CodeTarget):
        return target
    if isinstance(target, CodeTargetSpec):
        if target.import_path:
            return target_from_import_path(target.import_path, allow_import=allow_import, spec=target)
        return CodeTarget(spec=target, metadata=metadata or {})
    if isinstance(target, str):
        return target_from_import_path(target, allow_import=allow_import)
    if method_name is not None:
        cls = target if inspect.isclass(target) else type(target) if target is not None else None
        return target_from_definition_method(subject_ref, cls, method_name)
    if inspect.ismethod(target):
        return target_from_method(target, metadata=metadata)
    if inspect.isclass(target):
        return _target_from_object(target, "class", metadata=metadata)
    if callable(target):
        return target_from_callable(target, metadata=metadata)
    return CodeTarget(
        spec=CodeTargetSpec("unknown", metadata=metadata or {"type": type(target).__name__}),
        obj=target,
    )


def target_from_import_path(
    path: str,
    *,
    allow_import: bool = True,
    spec: CodeTargetSpec | None = None,
) -> CodeTarget:
    """Create a target from an import path, optionally resolving it.

    Args:
        path: Import path of the form ``module:qualname``.
        allow_import: Whether to import and resolve the object.
        spec: Optional existing spec to preserve metadata.

    Returns:
        A target with diagnostics for malformed or unresolved paths.
    """

    diagnostics: list[DiagnosticFact] = []
    target_spec = spec or CodeTargetSpec("import_path", import_path=path)
    parsed = _parse_import_path(path)
    if parsed is None:
        diagnostics.append(DiagnosticFact(
            severity="error",
            code="dryml.code.import_path_invalid",
            message="Import path must be of the form 'module:qualname'.",
            data={"import_path": path},
        ))
        return CodeTarget(spec=target_spec, diagnostics=tuple(diagnostics))
    if not allow_import:
        return CodeTarget(spec=target_spec)

    module_name, qualname = parsed
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        diagnostics.append(DiagnosticFact(
            severity="error",
            code="dryml.code.import_failed",
            message=f"Could not import module {module_name!r}.",
            data={"import_path": path, "error": repr(exc)},
        ))
        return CodeTarget(spec=target_spec, diagnostics=tuple(diagnostics))

    try:
        obj = _resolve_qualname(module, qualname)
    except Exception as exc:
        diagnostics.append(DiagnosticFact(
            severity="error",
            code="dryml.code.qualname_resolution_failed",
            message=f"Could not resolve qualname {qualname!r} in module {module_name!r}.",
            data={"import_path": path, "error": repr(exc)},
        ))
        return CodeTarget(spec=target_spec, diagnostics=tuple(diagnostics))

    live = normalize_target(obj, allow_import=False)
    return CodeTarget(
        spec=target_spec,
        obj=live.obj,
        owner=live.owner,
        attribute_name=live.attribute_name,
        raw_descriptor=live.raw_descriptor,
        unwrapped=live.unwrapped,
        metadata=live.metadata,
        diagnostics=tuple(diagnostics),
    )


def target_from_callable(func: Callable[..., Any], *, metadata: Mapping[str, Any] | None = None) -> CodeTarget:
    """Create a target from a live callable without invoking it."""

    if inspect.ismethod(func):
        return target_from_method(func, metadata=metadata)
    if inspect.isclass(func):
        return _target_from_object(func, "class", metadata=metadata)
    if inspect.isfunction(func):
        qualname = getattr(func, "__qualname__", "") or ""
        if getattr(func, "__name__", None) == "<lambda>":
            kind = "lambda"
        elif "<locals>" in qualname:
            kind = "local_function"
        elif "." in qualname:
            kind = "unbound_method"
        else:
            kind = "function"
        return _target_from_object(func, kind, metadata=metadata)
    return _target_from_object(func, "callable_instance", unwrapped=getattr(type(func), "__call__", None), metadata=metadata)


def target_from_method(method: Callable[..., Any], *, metadata: Mapping[str, Any] | None = None) -> CodeTarget:
    """Create a target from a live bound method."""

    func = getattr(method, "__func__", method)
    owner = method.__self__ if isinstance(getattr(method, "__self__", None), type) else type(getattr(method, "__self__", None))
    kind = "class_method" if isinstance(getattr(method, "__self__", None), type) else "bound_method"
    spec = _spec_for_object(func, kind, method_name=getattr(func, "__name__", None), metadata=metadata)
    return CodeTarget(
        spec=spec,
        obj=method,
        owner=owner,
        attribute_name=getattr(func, "__name__", None),
        raw_descriptor=getattr(owner, "__dict__", {}).get(getattr(func, "__name__", "")) if owner else None,
        unwrapped=func,
        metadata=metadata or {},
    )


def target_from_class_attribute(
    cls: type,
    name: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CodeTarget:
    """Create a target from a raw class attribute without descriptor binding.

    Args:
        cls: Class owning the attribute.
        name: Attribute name to inspect with :func:`inspect.getattr_static`.
        metadata: Optional serializable metadata to add to the target spec.

    Returns:
        A target preserving the raw descriptor for analyzers that need to inspect
        ``staticmethod`` or ``classmethod`` metadata attached to the descriptor.
    """

    diagnostics: list[DiagnosticFact] = []
    try:
        raw_descriptor = inspect.getattr_static(cls, name)
    except AttributeError as exc:
        diagnostics.append(DiagnosticFact(
            severity="error",
            code="dryml.code.class_attribute_missing",
            message=f"Class attribute {name!r} is not present on {cls.__qualname__!r}.",
            source={"target_kind": "class_attribute", "attribute_name": name},
            data={"error": repr(exc)},
        ))
        raw_descriptor = None

    obj = getattr(cls, name, None)
    if isinstance(raw_descriptor, classmethod):
        kind = "class_method"
    elif isinstance(raw_descriptor, staticmethod):
        kind = "static_method"
    elif inspect.isfunction(_unwrap_descriptor(raw_descriptor)):
        kind = "unbound_method"
    else:
        kind = "unknown"

    local_metadata = {
        "module": getattr(_unwrap_descriptor(raw_descriptor), "__module__", getattr(cls, "__module__", None)),
        "qualname": getattr(_unwrap_descriptor(raw_descriptor), "__qualname__", None),
        "owner_module": getattr(cls, "__module__", None),
        "owner_qualname": getattr(cls, "__qualname__", None),
        **dict(metadata or {}),
    }
    spec = CodeTargetSpec(
        kind,
        import_path=_object_import_path(raw_descriptor),
        method_name=name,
        metadata=local_metadata,
    )
    return CodeTarget(
        spec=spec,
        obj=obj,
        owner=cls,
        attribute_name=name,
        raw_descriptor=raw_descriptor,
        unwrapped=_unwrap_descriptor(raw_descriptor),
        metadata=metadata or {},
        diagnostics=tuple(diagnostics),
    )


def target_from_definition_method(
    subject_ref: str | None,
    cls: type | None,
    method_name: str,
) -> CodeTarget:
    """Create a representational target for a method on a definition/class."""

    raw_descriptor = getattr(cls, "__dict__", {}).get(method_name) if cls is not None else None
    spec = CodeTargetSpec(
        "definition_method",
        import_path=_object_import_path(raw_descriptor) if raw_descriptor is not None else None,
        method_name=method_name,
        subject_ref=subject_ref,
        metadata={"owner": _object_import_path(cls) if cls is not None else None},
    )
    return CodeTarget(
        spec=spec,
        obj=getattr(cls, method_name, None) if cls is not None else None,
        owner=cls,
        attribute_name=method_name,
        raw_descriptor=raw_descriptor,
        unwrapped=_unwrap_descriptor(raw_descriptor),
    )


def _target_from_object(
    obj: Any,
    kind: str,
    *,
    unwrapped: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CodeTarget:
    return CodeTarget(
        spec=_spec_for_object(obj, kind, metadata=metadata),
        obj=obj,
        unwrapped=unwrapped or _unwrap_descriptor(obj),
        metadata=metadata or {},
    )


def _spec_for_object(
    obj: Any,
    kind: str,
    *,
    method_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CodeTargetSpec:
    return CodeTargetSpec(
        kind=kind,
        import_path=_object_import_path(obj),
        method_name=method_name,
        metadata={
            "module": getattr(obj, "__module__", None),
            "qualname": getattr(obj, "__qualname__", None),
            **dict(metadata or {}),
        },
    )


def _parse_import_path(path: str) -> tuple[str, str] | None:
    if not isinstance(path, str) or ":" not in path:
        return None
    module_name, qualname = path.split(":", 1)
    module_name = module_name.strip()
    qualname = qualname.strip()
    if not module_name or not qualname:
        return None
    return module_name, qualname


def _resolve_qualname(root: Any, qualname: str) -> Any:
    obj = root
    for part in qualname.split("."):
        if part == "<locals>":
            raise ValueError("Cannot resolve local qualname components.")
        obj = getattr(obj, part)
    return obj


def _object_import_path(obj: Any) -> str | None:
    target = _unwrap_descriptor(obj)
    module_name = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not module_name or not qualname or module_name == "__main__" or "<locals>" in qualname:
        return None
    try:
        module = importlib.import_module(module_name)
        resolved = _resolve_qualname(module, qualname)
    except Exception:
        return None
    if resolved is target:
        return f"{module_name}:{qualname}"
    return None


def _unwrap_descriptor(obj: Any) -> Any:
    if isinstance(obj, (staticmethod, classmethod)):
        return obj.__func__
    return getattr(obj, "__func__", obj)


__all__ = [
    "CodeTarget",
    "CodeTargetSpec",
    "normalize_target",
    "target_from_callable",
    "target_from_class_attribute",
    "target_from_definition_method",
    "target_from_import_path",
    "target_from_method",
]
