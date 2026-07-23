from __future__ import annotations

import importlib
import inspect
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from dryml.core.methods import bound_method_parts, descriptor_function

from .facts import DiagnosticFact, json_compatible


@dataclass(frozen=True, slots=True)
class CodeTargetSpec:
    """Serializable description of a Python or DRYML code target.

    Args:
        kind: Descriptive target kind such as ``"function"`` or ``"class"``.
        import_path: Optional ``module:qualname`` reference.
        source_spec: Optional serialized source-backed reference retained for
            diagnostics and provenance. It does not reconstruct a live target
            for subprocess analysis.
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

    # Exact-type checks avoid invoking a hostile target's ``__class__`` hook
    # before it can be classified as an unsupported callable instance.
    if type(target) is CodeTarget:
        return target
    if type(target) is CodeTargetSpec:
        if target.import_path:
            return target_from_import_path(target.import_path, allow_import=allow_import, spec=target)
        return CodeTarget(spec=target, metadata=metadata or {})
    if type(target) is str:
        return target_from_import_path(target, allow_import=allow_import, metadata=metadata)
    if method_name is not None:
        cls = target if _is_class(target) else type(target) if target is not None else None
        return target_from_definition_method(subject_ref, cls, method_name)
    if bound_method_parts(target) is not None:
        return target_from_method(target, metadata=metadata)
    unwrapped_descriptor = descriptor_function(target)
    if unwrapped_descriptor is not target and type(unwrapped_descriptor) is types.FunctionType:
        normalized = target_from_callable(unwrapped_descriptor, metadata=metadata)
        return CodeTarget(
            spec=normalized.spec,
            obj=unwrapped_descriptor,
            owner=normalized.owner,
            attribute_name=getattr(unwrapped_descriptor, "__name__", None),
            raw_descriptor=target,
            unwrapped=unwrapped_descriptor,
            metadata=normalized.metadata,
            diagnostics=normalized.diagnostics,
        )
    if _is_class(target):
        if type(target) is not type:
            return CodeTarget(
                spec=CodeTargetSpec("class", metadata={"type": object.__getattribute__(target, "__name__"), **dict(metadata or {})}),
                obj=target,
                metadata=metadata or {},
            )
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
    metadata: Mapping[str, Any] | None = None,
) -> CodeTarget:
    """Create a target from an import path, optionally resolving it.

    Args:
        path: Import path of the form ``module:qualname``.
        allow_import: Whether to import and resolve the object.
        spec: Optional existing spec to preserve metadata.
        metadata: Optional metadata to add when creating a new import-path spec.

    Returns:
        A target with diagnostics for malformed or unresolved paths.
    """

    diagnostics: list[DiagnosticFact] = []
    target_spec = spec or CodeTargetSpec("import_path", import_path=path, metadata=metadata or {})
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
        obj, owner, attribute_name = _resolve_qualname(module, qualname)
    except Exception as exc:
        diagnostics.append(DiagnosticFact(
            severity="error",
            code="dryml.code.qualname_resolution_failed",
            message=f"Could not resolve qualname {qualname!r} in module {module_name!r}.",
            data={"import_path": path, "error": repr(exc)},
        ))
        return CodeTarget(spec=target_spec, diagnostics=tuple(diagnostics))

    live = _target_from_static_resolution(obj, owner, attribute_name)
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

    if bound_method_parts(func) is not None:
        return target_from_method(func, metadata=metadata)
    if _is_class(func):
        if type(func) is not type:
            return CodeTarget(
                spec=CodeTargetSpec("class", metadata={"type": object.__getattribute__(func, "__name__"), **dict(metadata or {})}),
                obj=func,
                metadata=metadata or {},
            )
        return _target_from_object(func, "class", metadata=metadata)
    if type(func) is types.FunctionType:
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
    # Do not inspect callable-instance or metaclass attributes. Those lookups can
    # execute user-defined hooks before an analyzer has a chance to reject them.
    return CodeTarget(
        spec=CodeTargetSpec(
            "callable_instance",
            metadata={"type": object.__getattribute__(type(func), "__name__"), **dict(metadata or {})},
        ),
        obj=func,
        metadata=metadata or {},
    )


def target_from_method(method: Callable[..., Any], *, metadata: Mapping[str, Any] | None = None) -> CodeTarget:
    """Create a target from a live bound method."""

    parts = bound_method_parts(method)
    if parts is None:
        raise TypeError("target_from_method requires a bound method-like target.")
    receiver, func = parts
    receiver_is_class = _is_class(receiver)
    owner = receiver if receiver_is_class else type(receiver)
    method_name = object.__getattribute__(func, "__name__")
    kind = "class_method" if receiver_is_class else "bound_method"
    spec = _spec_for_object(func, kind, method_name=method_name, metadata=metadata)
    try:
        raw_descriptor = _class_static_attribute(owner, method_name)
    except AttributeError:
        raw_descriptor = None
    return CodeTarget(
        spec=spec,
        obj=method,
        owner=owner,
        attribute_name=method_name,
        raw_descriptor=raw_descriptor,
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
        raw_descriptor = _class_static_attribute(cls, name)
    except AttributeError as exc:
        diagnostics.append(DiagnosticFact(
            severity="error",
            code="dryml.code.class_attribute_missing",
            message=f"Class attribute {name!r} is not present on {object.__getattribute__(cls, '__qualname__')!r}.",
            source={"target_kind": "class_attribute", "attribute_name": name},
            data={"error": repr(exc)},
        ))
        raw_descriptor = None

    unwrapped = _unwrap_descriptor(raw_descriptor)
    if type(raw_descriptor) is classmethod:
        kind = "class_method"
    elif type(raw_descriptor) is staticmethod:
        kind = "static_method"
    elif type(unwrapped) is types.FunctionType:
        kind = "unbound_method"
    else:
        kind = "unknown"

    local_metadata = {
        "module": _object_module(unwrapped) or _object_module(cls),
        "qualname": _object_qualname(unwrapped),
        "owner_module": _object_module(cls),
        "owner_qualname": _object_qualname(cls),
        **dict(metadata or {}),
    }
    spec = CodeTargetSpec(
        kind,
        import_path=_object_import_path(unwrapped),
        method_name=name,
        metadata=local_metadata,
    )
    return CodeTarget(
        spec=spec,
        obj=unwrapped,
        owner=cls,
        attribute_name=name,
        raw_descriptor=raw_descriptor,
        unwrapped=unwrapped,
        metadata=metadata or {},
        diagnostics=tuple(diagnostics),
    )


def target_from_definition_method(
    subject_ref: str | None,
    cls: type | None,
    method_name: str,
) -> CodeTarget:
    """Create a representational target for a method on a definition/class."""

    try:
        raw_descriptor = _class_static_attribute(cls, method_name) if cls is not None else None
    except AttributeError:
        raw_descriptor = None
    unwrapped = _unwrap_descriptor(raw_descriptor)
    spec = CodeTargetSpec(
        "definition_method",
        import_path=_object_import_path(unwrapped) if raw_descriptor is not None else None,
        method_name=method_name,
        subject_ref=subject_ref,
        metadata={"owner": _object_import_path(cls) if cls is not None else None},
    )
    return CodeTarget(
        spec=spec,
        obj=unwrapped,
        owner=cls,
        attribute_name=method_name,
        raw_descriptor=raw_descriptor,
        unwrapped=unwrapped,
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
        unwrapped=unwrapped if unwrapped is not None else _unwrap_descriptor(obj),
        metadata=metadata or {},
    )


def _is_class(value: Any) -> bool:
    """Return whether *value* is a class without reading target attributes."""

    return issubclass(type(value), type)


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
            "module": _object_module(obj),
            "qualname": _object_qualname(obj),
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


def _resolve_qualname(root: Any, qualname: str) -> tuple[Any, type | None, str | None]:
    """Resolve an import qualname without invoking descriptors or hooks."""

    obj = root
    owner: type | None = None
    attribute_name: str | None = None
    for part in qualname.split("."):
        if part == "<locals>":
            raise ValueError("Cannot resolve local qualname components.")
        owner = obj if _is_class(obj) else None
        obj = _class_static_attribute(obj, part) if owner is not None else inspect.getattr_static(obj, part)
        attribute_name = part
    return obj, owner, attribute_name


def _target_from_static_resolution(
    obj: Any,
    owner: type | None,
    attribute_name: str | None,
) -> CodeTarget:
    """Normalize a statically resolved import target without descriptor binding."""

    if owner is not None and attribute_name is not None:
        target = target_from_class_attribute(owner, attribute_name)
        return target
    unwrapped = _unwrap_descriptor(obj)
    if type(unwrapped) in {types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType, type}:
        return normalize_target(unwrapped, allow_import=False)
    # A raw non-function descriptor has no safe live representation for analyzers.
    return CodeTarget(spec=CodeTargetSpec("unknown"))


def _object_import_path(obj: Any) -> str | None:
    target = _unwrap_descriptor(obj)
    if type(target) not in {types.FunctionType, type}:
        return None
    module_name = _object_module(target)
    qualname = _object_qualname(target)
    if not module_name or not qualname or module_name == "__main__" or "<locals>" in qualname:
        return None
    try:
        module = importlib.import_module(module_name)
        resolved = _resolve_static_qualname(module, qualname)
    except Exception:
        return None
    if resolved is target:
        return f"{module_name}:{qualname}"
    return None


def _unwrap_descriptor(obj: Any) -> Any:
    return descriptor_function(obj)


def _object_module(obj: Any) -> str | None:
    """Return safe module metadata for a plain function or ordinary class."""

    if type(obj) not in {types.FunctionType, type}:
        return None
    value = object.__getattribute__(obj, "__module__")
    return value if isinstance(value, str) else None


def _object_qualname(obj: Any) -> str | None:
    """Return safe qualname metadata for a plain function or ordinary class."""

    if type(obj) not in {types.FunctionType, type}:
        return None
    value = object.__getattribute__(obj, "__qualname__")
    return value if isinstance(value, str) else None


def _resolve_static_qualname(root: Any, qualname: str) -> Any:
    """Resolve a qualname without binding descriptors or dynamic attributes."""

    obj = root
    for part in qualname.split("."):
        if part == "<locals>":
            raise ValueError("Cannot resolve local qualname components.")
        obj = _class_static_attribute(obj, part) if _is_class(obj) else inspect.getattr_static(obj, part)
    return _unwrap_descriptor(obj)


def _class_static_attribute(cls: type, name: str) -> Any:
    """Read a class dictionary through ``type`` without metaclass dispatch."""

    for base in type.__getattribute__(cls, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    raise AttributeError(name)


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
