"""Closed static normalization for generic code-analysis targets."""

from __future__ import annotations

import ast
import importlib
import os
import types
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeAlias

from .callable_info import _function_from_descriptor, _raw_call_descriptor, analyze_callable
from .errors import InvalidTargetError, SourceUnavailableError
from .source import SourceInfo


TargetKind: TypeAlias = Literal["function", "bound_method", "callable_instance", "descriptor", "class", "import", "source"]
DescriptorKind: TypeAlias = Literal["function", "staticmethod", "classmethod"]


def _metadata(func: types.FunctionType) -> tuple[str | None, str | None, str | None]:
    """Return safe raw function name, module, and qualified name."""

    name = func.__name__ if type(func.__name__) is str else None
    module = func.__module__ if type(func.__module__) is str else None
    qualname = func.__qualname__ if type(func.__qualname__) is str else None
    return name, module, qualname


def _class_metadata(cls: type) -> tuple[str | None, str | None, str | None]:
    """Read class metadata while bypassing custom metaclass attribute hooks."""

    name = type.__getattribute__(cls, "__name__")
    module = type.__getattribute__(cls, "__module__")
    qualname = type.__getattribute__(cls, "__qualname__")
    return (
        name if type(name) is str else None,
        module if type(module) is str else None,
        qualname if type(qualname) is str else None,
    )


def _is_class(value: object) -> bool:
    """Return whether a value is a class without reading its ``__class__`` hook."""

    return issubclass(type(value), type)


def _safe_filename(module: str | None, filename: str | None) -> str | None:
    """Convert raw source provenance to a logical module name or basename."""

    if module:
        return module
    if type(filename) is str and filename:
        return os.path.basename(filename.replace("\\", "/")) or None
    return None


def _function_source(func: types.FunctionType) -> SourceInfo | None:
    """Retrieve file-backed source lazily without dynamic source-loader hooks."""

    from .source import get_source_info

    return get_source_info(func)


def _require_source(source: SourceInfo | None) -> SourceInfo:
    """Reject a live target without an admitted ordinary source file."""

    if source is None:
        raise InvalidTargetError("target source is unavailable")
    return source


def _descriptor(owner: type, name: str) -> tuple[object, type] | None:
    """Find an owner-MRO descriptor without binding or invoking it."""

    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name], base
    return None


def _descriptor_kind(descriptor: object) -> DescriptorKind | None:
    """Classify an admitted raw descriptor without calling descriptor hooks."""

    if type(descriptor) is types.FunctionType:
        return "function"
    if type(descriptor) is staticmethod and type(descriptor.__func__) is types.FunctionType:
        return "staticmethod"
    if type(descriptor) is classmethod and type(descriptor.__func__) is types.FunctionType:
        return "classmethod"
    return None


@dataclass(frozen=True, slots=True)
class SourceTarget:
    """Static source target that is parsed but never compiled or executed.

    Args:
        source: Python source containing exactly one supported top-level subject.
        name: Optional selected function or class name.
        filename: Optional request-local source filename.
        start_line: Optional one-based source origin line.

    Raises:
        ValueError: If carrier field types are invalid.

    Side Effects:
        None. Validation does not parse or execute the source.
    """

    source: str
    name: str | None = None
    filename: str | None = None
    start_line: int | None = None

    def __post_init__(self) -> None:
        """Validate static carrier types before later source parsing."""

        if type(self.source) is not str or (self.name is not None and type(self.name) is not str):
            raise ValueError("source target is invalid")
        if self.filename is not None and type(self.filename) is not str:
            raise ValueError("source target filename is invalid")
        if self.start_line is not None and (type(self.start_line) is not int or self.start_line < 1):
            raise ValueError("source target start line is invalid")


@dataclass(frozen=True, slots=True)
class ImportTarget:
    """Explicit import target using the ``module[:qualname]`` grammar.

    Args:
        path: Module path alone or module path followed by a static qualname.

    Raises:
        ValueError: If ``path`` is not an exact built-in string.

    Side Effects:
        None. Import occurs only during normalization.
    """

    path: str

    def __post_init__(self) -> None:
        """Validate only the carrier type; grammar validation is typed analysis."""

        if type(self.path) is not str:
            raise ValueError("import target path is invalid")


@dataclass(frozen=True, slots=True)
class DescriptorTarget:
    """Raw owner/name reference resolved through class dictionaries and MRO.

    Args:
        owner: Class declaring or inheriting the target descriptor.
        name: Descriptor name to resolve without binding.

    Raises:
        ValueError: If the descriptor carrier fields are invalid.

    Side Effects:
        None. Descriptor binding and dynamic class lookup are never performed.
    """

    owner: type
    name: str

    def __post_init__(self) -> None:
        """Validate descriptor carrier types without inspecting owner members."""

        if not _is_class(self.owner) or type(self.name) is not str or not self.name:
            raise ValueError("descriptor target is invalid")


@dataclass(frozen=True, slots=True)
class TargetInfo:
    """Immutable metadata-only provenance for one normalized code target.

    Args:
        kind: Closed normalized target kind.
        name: Optional callable, class, descriptor, or selected-source name.
        module: Optional target module name.
        qualname: Optional target qualified name.
        owner_module: Optional descriptor or receiver owner module name.
        owner_qualname: Optional descriptor or receiver owner qualified name.
        descriptor_kind: Optional admitted raw descriptor category.
        filename: Sanitized logical module name or source basename.
        start_line: Optional one-based source origin line.
        import_path: Optional caller-specified explicit import path.

    Raises:
        ValueError: If framework-owned metadata is invalid or contains an
            absolute filesystem path.

    Side Effects:
        None. It retains no live target handle.
    """

    kind: TargetKind
    name: str | None
    module: str | None
    qualname: str | None
    owner_module: str | None
    owner_qualname: str | None
    descriptor_kind: DescriptorKind | None
    filename: str | None
    start_line: int | None
    import_path: str | None

    def __post_init__(self) -> None:
        """Validate immutable metadata and sanitize any source path."""

        if self.kind not in ("function", "bound_method", "callable_instance", "descriptor", "class", "import", "source"):
            raise ValueError("target kind is invalid")
        for value in (self.name, self.module, self.qualname, self.owner_module, self.owner_qualname, self.filename, self.import_path):
            if value is not None and type(value) is not str:
                raise ValueError("target metadata is invalid")
        if self.descriptor_kind not in (None, "function", "staticmethod", "classmethod"):
            raise ValueError("descriptor kind is invalid")
        if self.start_line is not None and (type(self.start_line) is not int or self.start_line < 1):
            raise ValueError("target source line is invalid")
        object.__setattr__(self, "filename", _safe_filename(self.module, self.filename))


@dataclass(frozen=True, slots=True)
class CodeTarget:
    """Request-local normalized target plus immutable metadata provenance.

    Args:
        info: Metadata-only normalized target provenance.
        original: Optional original live target handle.
        callable: Optional admitted raw callable function.
        owner: Optional receiver or descriptor owner class.
        descriptor: Optional raw unbound descriptor.
        source: Optional request-local static source.
        import_path: Optional explicit import path used for admission.

    Raises:
        ValueError: If framework-owned fields are not exact supported carriers.

    Side Effects:
        None. Live fields are request-local and must not be copied into graphs or
        returned framework provenance.
    """

    info: TargetInfo
    original: object | None
    callable: Callable[..., Any] | None
    owner: type | None
    descriptor: object | None
    source: SourceInfo | None
    import_path: str | None

    def __post_init__(self) -> None:
        """Validate framework-owned container fields without inspecting handles."""

        if type(self.info) is not TargetInfo or (self.owner is not None and not _is_class(self.owner)):
            raise ValueError("code target is invalid")
        if self.source is not None and type(self.source) is not SourceInfo:
            raise ValueError("code target source is invalid")
        if self.import_path is not None and type(self.import_path) is not str:
            raise ValueError("code target import path is invalid")


CodeTargetInput: TypeAlias = CodeTarget | SourceTarget | ImportTarget | DescriptorTarget | Callable[..., Any] | type


def _normal_function(func: types.FunctionType) -> CodeTarget:
    """Normalize a direct Python function after safe callable admission."""

    info = analyze_callable(func)
    name, module, qualname = _metadata(func)
    source = _require_source(_function_source(func))
    return CodeTarget(
        TargetInfo("function", name, module, qualname, None, None, None, _safe_filename(module, source.filename if source else None), source.start_line if source else None, None),
        func,
        info.func,
        None,
        None,
        source,
        None,
    )


def _bound_method(method: types.MethodType) -> CodeTarget:
    """Normalize a bound Python method without inspecting its receiver state."""

    info = analyze_callable(method)
    name, module, qualname = _metadata(info.func)
    owner = type(info.bound_self)
    _, owner_module, owner_qualname = _class_metadata(owner)
    source = _require_source(_function_source(info.func))
    return CodeTarget(
        TargetInfo("bound_method", name, module, qualname, owner_module, owner_qualname, None, _safe_filename(module, source.filename if source else None), source.start_line if source else None, None),
        method,
        info.func,
        owner,
        None,
        source,
        None,
    )


def _callable_instance(instance: object) -> CodeTarget:
    """Normalize a supported instance through its raw class ``__call__`` only."""

    info = analyze_callable(instance)  # type: ignore[arg-type]
    owner = type(instance)
    _, owner_module, owner_qualname = _class_metadata(owner)
    name, module, qualname = _metadata(info.func)  # type: ignore[arg-type]
    source = _require_source(_function_source(info.func))  # type: ignore[arg-type]
    return CodeTarget(
        TargetInfo("callable_instance", name, module, qualname, owner_module, owner_qualname, None, _safe_filename(module, source.filename if source else None), source.start_line if source else None, None),
        instance,
        info.func,
        owner,
        _raw_call_descriptor(owner),
        source,
        None,
    )


def _class_target(cls: type) -> CodeTarget:
    """Normalize a class solely as a static source subject."""

    from .source import get_source_info

    name, module, qualname = _class_metadata(cls)
    source = _require_source(get_source_info(cls))
    return CodeTarget(
        TargetInfo("class", name, module, qualname, None, None, None, _safe_filename(module, source.filename if source else None), source.start_line if source else None, None),
        cls,
        None,
        None,
        None,
        source,
        None,
    )


def _descriptor_target(target: DescriptorTarget) -> CodeTarget:
    """Normalize an admitted raw descriptor using static MRO lookup."""

    found = _descriptor(target.owner, target.name)
    if found is None:
        raise InvalidTargetError("descriptor target is unavailable")
    descriptor, declaring_owner = found
    kind = _descriptor_kind(descriptor)
    func = _function_from_descriptor(descriptor)
    if kind is None or func is None:
        raise InvalidTargetError("unsupported descriptor target")
    analyze_callable(func)
    _, module, qualname = _metadata(func)
    _, owner_module, owner_qualname = _class_metadata(declaring_owner)
    source = _require_source(_function_source(func))
    return CodeTarget(
        TargetInfo("descriptor", target.name, module, qualname, owner_module, owner_qualname, kind, _safe_filename(module, source.filename if source else None), source.start_line if source else None, None),
        None,
        func,
        target.owner,
        descriptor,
        source,
        None,
    )


def _source_target(target: SourceTarget) -> CodeTarget:
    """Parse and admit exactly one unambiguous static source subject."""

    from .ast_tools import parse_source

    try:
        tree = parse_source(target.source)
    except SourceUnavailableError:
        raise SourceUnavailableError("source is invalid", code="source.invalid") from None
    candidates: list[tuple[str | None, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            candidates.append((node.name, node))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Lambda):
            candidates.append((None, node.value))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Lambda):
            candidates.append((node.targets[0].id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.value, ast.Lambda):
            candidates.append((node.target.id, node.value))
    if target.name is not None:
        candidates = [candidate for candidate in candidates if candidate[0] == target.name]
    if len(candidates) != 1 or len(tree.body) != 1:
        raise InvalidTargetError("source target is ambiguous")
    name, _ = candidates[0]
    source = SourceInfo(target.source, target.filename, target.start_line)
    return CodeTarget(
        TargetInfo("source", name, None, None, None, None, None, _safe_filename(None, target.filename), target.start_line, None),
        None,
        None,
        None,
        None,
        source,
        None,
    )


def _parse_import_path(path: str) -> tuple[str, tuple[str, ...]]:
    """Validate and split the exact explicit import grammar."""

    if path.count(":") > 1:
        raise InvalidTargetError("invalid import target")
    module, separator, qualname = path.partition(":")
    if not module or any(not segment.isidentifier() for segment in module.split(".")):
        raise InvalidTargetError("invalid import target")
    if separator:
        if not qualname or any(not segment or segment == "<locals>" or not segment.isidentifier() for segment in qualname.split(".")):
            raise InvalidTargetError("invalid import target")
    return module, tuple(qualname.split(".")) if separator else ()


def _static_member(value: object, name: str) -> object | None:
    """Traverse one imported qualified-name segment without dynamic lookup."""

    if isinstance(value, types.ModuleType):
        namespace = types.ModuleType.__getattribute__(value, "__dict__")
        return namespace.get(name)
    if _is_class(value):
        found = _descriptor(value, name)
        return found[0] if found is not None else None
    return None


def _import_target(target: ImportTarget) -> CodeTarget:
    """Import only the requested module then traverse its qualname statically."""

    module_name, segments = _parse_import_path(target.path)
    try:
        resolved: object = importlib.import_module(module_name)
    except Exception:
        raise InvalidTargetError("import target could not be resolved", code="target.import_failed") from None
    parent: object | None = None
    final_segment: str | None = None
    for segment in segments:
        parent = resolved
        final_segment = segment
        resolved = _static_member(resolved, segment)
        if resolved is None:
            raise InvalidTargetError("import target could not be resolved")
    if not segments:
        return CodeTarget(TargetInfo("import", None, module_name, None, None, None, None, module_name, None, target.path), resolved, None, None, None, None, target.path)
    if _is_class(parent) and final_segment is not None:
        found = _descriptor(parent, final_segment)
        if found is not None and _descriptor_kind(found[0]) is not None:
            normalized = _descriptor_target(DescriptorTarget(parent, final_segment))
        else:
            normalized = normalize_target(resolved)
    else:
        normalized = normalize_target(resolved)  # Static traversal has already selected the raw member.
    return CodeTarget(
        TargetInfo("import", normalized.info.name, normalized.info.module, normalized.info.qualname, normalized.info.owner_module, normalized.info.owner_qualname, normalized.info.descriptor_kind, normalized.info.filename, normalized.info.start_line, target.path),
        normalized.original,
        normalized.callable,
        normalized.owner,
        normalized.descriptor,
        normalized.source,
        target.path,
    )


def normalize_target(target: CodeTargetInput) -> CodeTarget:
    """Normalize one supported static code target through a closed whitelist.

    Args:
        target: Existing normalized target, explicit source/import/descriptor
            wrapper, Python function, bound Python method, class, or admitted
            callable instance.

    Returns:
        Request-local target handles paired with immutable metadata-only
        provenance.

    Raises:
        InvalidTargetError: If a target form is unsupported, dynamic, malformed,
            or cannot be resolved through the explicit import boundary.
        SourceUnavailableError: If explicit source is malformed.

    Side Effects:
        Explicit import targets may execute the selected module's top-level code.
        All other forms avoid target execution, class construction, descriptor
        binding, dynamic lookup, and custom reflection hooks.
    """

    if type(target) is CodeTarget:
        return target
    if type(target) is SourceTarget:
        return _source_target(target)
    if type(target) is ImportTarget:
        return _import_target(target)
    if type(target) is DescriptorTarget:
        return _descriptor_target(target)
    if type(target) is types.FunctionType:
        return _normal_function(target)
    if type(target) is types.MethodType and type(target.__func__) is types.FunctionType:
        return _bound_method(target)
    if _is_class(target):
        return _class_target(target)
    return _callable_instance(target)


__all__ = [
    "CodeTarget",
    "CodeTargetInput",
    "DescriptorKind",
    "DescriptorTarget",
    "ImportTarget",
    "SourceTarget",
    "TargetInfo",
    "TargetKind",
    "normalize_target",
]
