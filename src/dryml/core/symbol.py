from __future__ import annotations

import __main__
import ast
import importlib
import inspect
import json
import textwrap
from dataclasses import dataclass
from types import FrameType, FunctionType
from typing import Any, Literal


SourceSpecKind = Literal["function", "class"]


def _normalize_source(source: str) -> str:
    source = textwrap.dedent(source).strip()
    if not source:
        raise ValueError("Source cannot be empty.")
    return source


def _resolve_qualname(root, qualname: str):
    obj = root
    for part in qualname.split("."):
        if part == "<locals>":
            raise ValueError(f"Cannot resolve qualname {qualname!r}; it contains <locals>.")
        obj = getattr(obj, part)
    return obj


@dataclass(frozen=True, slots=True)
class ImportRef:
    """Canonical reference to one importable Python module or symbol.

    Args:
        module: Non-empty importable module name.
        qualname: Optional qualified symbol path within ``module``.

    Raises:
        ValueError: If either path component is invalid or an object has no
            stable import path.

    Side Effects:
        :meth:`resolve` imports ``module`` and traverses its qualified path.
    """

    module: str
    qualname: str | None = None

    def __post_init__(self):
        if not self.module or not isinstance(self.module, str):
            raise ValueError("ImportRef requires a non-empty module string.")
        if self.qualname is not None and (not self.qualname or not isinstance(self.qualname, str)):
            raise ValueError("ImportRef qualname must be a non-empty string or None.")

    @classmethod
    def from_import_path(cls, path: str) -> "ImportRef":
        """Build an import reference from ``module`` or ``module:qualname``.

        Args:
            path: Canonical import path string.

        Returns:
            The corresponding immutable import reference.

        Raises:
            ValueError: If the path is empty or malformed.

        Side Effects:
            None.
        """

        path = path.strip()
        if not path:
            raise ValueError("Import path cannot be empty.")
        if ":" not in path:
            return cls(module=path, qualname=None)
        module, qualname = path.split(":", 1)
        module, qualname = module.strip(), qualname.strip()
        if not module or not qualname:
            raise ValueError("Import path must be 'module' or 'module:qualname'.")
        return cls(module=module, qualname=qualname)

    @classmethod
    def from_object(cls, obj) -> "ImportRef":
        """Project an object onto its stable import path.

        Args:
            obj: Module or symbol to validate through its defining module.

        Returns:
            An import reference resolving to ``obj`` by object identity.

        Raises:
            ValueError: If no stable import path resolves back to ``obj``.

        Side Effects:
            May import the object's declared module for validation.
        """

        ref = _object_import_ref(obj)
        if ref is None:
            raise ValueError(f"Object of type {type(obj).__name__} does not have a stable import path.")
        return ref

    def import_path(self) -> str:
        """Return this reference's canonical import-path string.

        Returns:
            ``module`` or ``module:qualname``.

        Side Effects:
            None.
        """

        return self.module if self.qualname is None else f"{self.module}:{self.qualname}"

    def resolve(self):
        """Import and return the referenced module or symbol.

        Returns:
            The imported module or qualified symbol.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the qualified path cannot be resolved.

        Side Effects:
            Imports the referenced module.
        """

        module = importlib.import_module(self.module)
        return module if self.qualname is None else _resolve_qualname(module, self.qualname)

    def __repr__(self) -> str:
        if self.qualname is None:
            return f"ImportRef(module={self.module!r})"
        return f"ImportRef(module={self.module!r}, qualname={self.qualname!r})"

    def __stable_leaf_bytes__(self):
        return json.dumps({"kind": "import", "module": self.module, "qualname": self.qualname}, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _coerce_import_ref(obj) -> ImportRef:
    if isinstance(obj, ImportRef):
        return obj
    if isinstance(obj, str):
        return ImportRef.from_import_path(obj)
    return ImportRef.from_object(obj)


def _object_import_ref(obj) -> ImportRef | None:
    if inspect.ismodule(obj):
        name = getattr(obj, "__name__", None)
        return ImportRef(name) if name else None
    module_name = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not module_name or not qualname or module_name == "__main__" or "<locals>" in qualname:
        return None
    try:
        module = importlib.import_module(module_name)
        resolved = _resolve_qualname(module, qualname)
    except Exception:
        return None
    return ImportRef(module_name, qualname) if resolved is obj else None


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Serializable source-backed specification for a non-importable function or class.

    Args:
        kind: Whether ``source`` reconstructs a function or class.
        source: One normalized source expression or definition.
        name: Required namespace name for class and named-function definitions.
        imports: Mapping from external source names to stable import references.

    Raises:
        ValueError: If source, kind, class name, or captured dependencies are
            invalid or cannot be represented as stable imports.
        TypeError: If live construction receives an unsupported object type.

    Side Effects:
        :meth:`from_function` and :meth:`from_class` inspect source and may
        import dependency modules only to validate stable import references.
        :meth:`resolve` executes the normalized trusted source.
    """

    kind: SourceSpecKind
    source: str
    name: str | None = None
    imports: dict[str, ImportRef | str] | None = None

    def __post_init__(self):
        if self.kind not in ("function", "class"):
            raise ValueError(f"Invalid SourceSpec kind {self.kind!r}.")
        object.__setattr__(self, "source", _normalize_source(self.source))
        if self.kind == "class" and self.name is None:
            raise ValueError("Class SourceSpec requires a name.")
        imports = {name: _coerce_import_ref(ref) for name, ref in (self.imports or {}).items()}
        object.__setattr__(self, "imports", dict(sorted(imports.items())))

    @classmethod
    def from_function(cls, fn: FunctionType) -> "SourceSpec":
        """Capture one non-closure Python function as source and stable imports.

        Args:
            fn: Python function with retrievable file-backed source.

        Returns:
            A source specification that resolves equivalently in a clean namespace.

        Raises:
            TypeError: If ``fn`` is not a Python function.
            ValueError: If source is unavailable, a closure is required, or an
                external dependency cannot be projected to an import path.

        Side Effects:
            Inspects source and validates importable dependencies.
        """

        if not inspect.isfunction(fn):
            raise TypeError(f"Expected a Python function, got {type(fn).__name__}.")
        frame = inspect.currentframe()
        anchor = frame.f_back if frame is not None else None
        try:
            return _source_spec_from_function(fn, anchor)
        finally:
            del frame
            del anchor

    @classmethod
    def from_class(cls, obj: type) -> "SourceSpec":
        """Capture one non-importable Python class as source and stable imports.

        Args:
            obj: Python class with retrievable file-backed source.

        Returns:
            A source specification that resolves the class in a clean namespace.

        Raises:
            TypeError: If ``obj`` is not a class.
            ValueError: If source is unavailable or dependencies are not stable.

        Side Effects:
            Inspects source and validates importable dependencies.
        """

        if not inspect.isclass(obj):
            raise TypeError(f"Expected a Python class, got {type(obj).__name__}.")
        frame = inspect.currentframe()
        anchor = frame.f_back if frame is not None else None
        try:
            return _source_spec_from_class(obj, anchor)
        finally:
            del frame
            del anchor

    @classmethod
    def from_source(cls, source: str, *, kind: SourceSpecKind = "function", name: str | None = None, imports: dict[str, ImportRef | str] | None = None) -> "SourceSpec":
        """Construct a source specification from caller-supplied trusted source.

        Args:
            source: Function expression or definition, or class definition.
            kind: Expected reconstructed kind.
            name: Optional definition name; inferred for a sole class definition.
            imports: Stable imports required by the source.

        Returns:
            The normalized source specification.

        Raises:
            ValueError: If source or the selected kind/name is invalid.

        Side Effects:
            Parses source only when inferring a class name; it never executes it.
        """

        if kind == "class" and name is None:
            tree = ast.parse(_normalize_source(source))
            if len(tree.body) == 1 and isinstance(tree.body[0], ast.ClassDef):
                name = tree.body[0].name
        return cls(kind=kind, source=source, name=name, imports=imports)

    def resolve(self):
        """Resolve trusted source using captured import references.

        Returns:
            The reconstructed callable or class, or a matching live main symbol.

        Raises:
            ValueError: If named source does not define its requested name.
            TypeError: If reconstructed output does not match ``kind``.

        Side Effects:
            Imports captured references and evaluates or executes trusted source.
        """

        live = _matching_live_source(self)
        if live is not None:
            return live
        ns: dict[str, object] = {name: ref.resolve() for name, ref in (self.imports or {}).items()}
        if self.kind == "function" and self.name is None:
            obj = eval(self.source, ns, ns)
        else:
            exec(self.source, ns, ns)
            if self.name not in ns:
                raise ValueError(f"Resolved namespace does not contain symbol {self.name!r}.")
            obj = ns[self.name]
        if self.kind == "class" and not inspect.isclass(obj):
            raise TypeError("Resolved SourceSpec object is not a class.")
        if self.kind == "function" and not callable(obj):
            raise TypeError("Resolved SourceSpec object is not callable.")
        return obj

    def __repr__(self) -> str:
        return f"SourceSpec(kind={self.kind!r}, name={self.name!r}, source={self.source!r}, imports={self.imports!r})"

    def __stable_leaf_bytes__(self):
        payload = {"kind": self.kind, "source": self.source, "name": self.name, "imports": {name: ref.import_path() for name, ref in sorted((self.imports or {}).items())}}
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _extract_lambda_source(source: str) -> str:
    source = _normalize_source(source)
    tree = ast.parse(source)
    if len(tree.body) != 1:
        raise ValueError("Lambda source is ambiguous.")
    stmt = tree.body[0]
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Lambda):
        return ast.unparse(stmt.value)
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(stmt.value, ast.Lambda):
        return ast.unparse(stmt.value)
    raise ValueError("Could not extract a standalone lambda expression.")


def _extract_named_function_source(fn: FunctionType, source: str) -> tuple[str, str]:
    source = _normalize_source(source)
    tree = ast.parse(source)
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == fn.__name__:
            stmt.decorator_list = []
            return ast.unparse(stmt), stmt.name
    raise ValueError(f"Could not find function definition for {fn.__name__!r} in extracted source.")


def _extract_named_class_source(cls: type, source: str) -> tuple[str, str]:
    source = _normalize_source(source)
    tree = ast.parse(source)
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == cls.__name__:
            stmt.decorator_list = []
            return ast.unparse(stmt), stmt.name
    raise ValueError(f"Could not find class definition for {cls.__name__!r} in extracted source.")


def _source_spec_from_function(fn: FunctionType, caller_anchor: FrameType | None) -> SourceSpec:
    if fn.__code__.co_freevars:
        raise ValueError("Cannot create SourceSpec for a closure; environment capture is not implemented yet.")
    try:
        raw_source = inspect.getsource(fn)
    except (OSError, IOError) as error:
        raise ValueError("Could not retrieve function source.") from error
    if fn.__name__ == "<lambda>":
        source = _extract_lambda_source(raw_source)
        return SourceSpec("function", source, imports=_collect_source_imports(fn, source, caller_anchor))
    source, name = _extract_named_function_source(fn, raw_source)
    return SourceSpec("function", source, name, _collect_source_imports(fn, source, caller_anchor))


def _source_spec_from_class(cls: type, caller_anchor: FrameType | None) -> SourceSpec:
    try:
        raw_source = inspect.getsource(cls)
    except (OSError, IOError) as error:
        raise ValueError("Could not retrieve class source.") from error
    source, name = _extract_named_class_source(cls, raw_source)
    return SourceSpec("class", source, name, _collect_source_imports(cls, source, caller_anchor))


def _lookup_live_name_for_source(obj, name: str, caller_anchor: FrameType | None):
    """Resolve a lexical dependency under core's globals/module/caller policy."""

    globals_dict = getattr(obj, "__globals__", None)
    if globals_dict is not None and name in globals_dict:
        return globals_dict[name]
    module_name = getattr(obj, "__module__", None)
    if module_name:
        import sys

        module = sys.modules.get(module_name)
        if module is not None and name in vars(module):
            return vars(module)[name]
    frame = caller_anchor
    try:
        while frame is not None:
            if name in frame.f_locals:
                return frame.f_locals[name]
            frame = frame.f_back
    finally:
        del frame
    raise KeyError(name)


def _source_import_error(obj, missing: list[str] | None = None) -> ValueError:
    name = getattr(obj, "__name__", type(obj).__name__)
    message = f"Could not capture stable import paths for source-backed object {name!r}."
    if missing is not None:
        message += f" Missing/unimportable globals: {missing}"
    return ValueError(message)


def _collect_source_imports(obj, source: str, caller_anchor: FrameType | None) -> dict[str, ImportRef]:
    """Discover generic free names, then project them through core import policy."""

    candidate = _normalize_source(source)
    try:
        # This is the sole core-to-code boundary: discovery stays generic while
        # core retains caller lookup, importability validation, and identity.
        from dryml.code.algorithms.lexical_dependencies import (
            LexicalDependencies,
            LexicalDependency,
            collect_lexical_dependencies,
        )
        from dryml.code.targets import SourceTarget

        evidence = collect_lexical_dependencies(SourceTarget(candidate))
    except Exception:
        raise _source_import_error(obj) from None
    if type(evidence) is not LexicalDependencies or type(evidence.dependencies) is not tuple:
        raise _source_import_error(obj)
    names: list[str] = []
    for dependency in evidence.dependencies:
        if type(dependency) is not LexicalDependency or type(dependency.name) is not str or not dependency.name or dependency.name in names:
            raise _source_import_error(obj)
        names.append(dependency.name)
    imports: dict[str, ImportRef] = {}
    missing: list[str] = []
    for name in names:
        try:
            dep = _lookup_live_name_for_source(obj, name, caller_anchor)
        except KeyError:
            missing.append(name)
            continue
        ref = _object_import_ref(dep)
        if ref is None:
            missing.append(name)
        else:
            imports[name] = ref
    if missing:
        raise _source_import_error(obj, missing)
    return imports


def _current_main_ns() -> dict[str, object]:
    try:
        return vars(__main__)
    except Exception:
        return {}


def _matching_live_source(spec: SourceSpec):
    if spec.name is None:
        return None
    candidate = _current_main_ns().get(spec.name)
    if spec.kind == "function":
        if not inspect.isfunction(candidate):
            return None
        try:
            candidate_spec = SourceSpec.from_function(candidate)
        except Exception:
            return None
    elif spec.kind == "class":
        if not inspect.isclass(candidate):
            return None
        try:
            candidate_spec = SourceSpec.from_class(candidate)
        except Exception:
            return None
    else:
        return None
    if candidate_spec.kind == spec.kind and candidate_spec.name == spec.name and candidate_spec.source == spec.source and (candidate_spec.imports or {}) == (spec.imports or {}):
        return candidate
    return None


def symbol_ref(obj: ImportRef | SourceSpec | FunctionType | type | str, *, name: str | None = None) -> ImportRef | SourceSpec:
    """Return an import reference or source specification for a supported symbol.

    Args:
        obj: Existing reference, Python function, class, or trusted source string.
        name: Optional symbol name for trusted function source strings.

    Returns:
        An existing or validated import/source reference.

    Raises:
        TypeError: If ``obj`` has no supported symbol form.
        ValueError: If source-backed capture cannot safely project dependencies.

    Side Effects:
        Source-backed inputs inspect source and validate stable imports.
    """

    if isinstance(obj, (ImportRef, SourceSpec)):
        return obj
    frame = inspect.currentframe()
    caller_anchor = frame.f_back if frame is not None else None
    try:
        if inspect.isfunction(obj):
            ref = _object_import_ref(obj)
            return ref if ref is not None else _source_spec_from_function(obj, caller_anchor)
        if inspect.isclass(obj):
            ref = _object_import_ref(obj)
            return ref if ref is not None else _source_spec_from_class(obj, caller_anchor)
        if isinstance(obj, str):
            return SourceSpec.from_source(obj, kind="function", name=name)
    finally:
        del frame
        del caller_anchor
    raise TypeError(f"Cannot convert object of type {type(obj).__name__} to ImportRef or SourceSpec.")


def maybe_symbol_ref(obj: Any, *, functions: bool = True) -> ImportRef | SourceSpec | None:
    """Return a symbol reference when ``obj`` is safely representable.

    Args:
        obj: Candidate existing reference, class, or optionally function.
        functions: Whether functions are eligible for conversion.

    Returns:
        A reference or ``None`` when conversion is unavailable.

    Side Effects:
        Eligible source-backed candidates may inspect source and import modules.
    """

    if isinstance(obj, (ImportRef, SourceSpec)):
        return obj
    if inspect.isclass(obj) or (functions and inspect.isfunction(obj)):
        try:
            return symbol_ref(obj)
        except Exception:
            return None
    return None


def resolve_symbol(obj: Any) -> Any:
    """Resolve a symbol reference or return an ordinary value unchanged.

    Args:
        obj: Candidate :class:`ImportRef`, :class:`SourceSpec`, or live value.

    Returns:
        The resolved symbol or the original non-reference value.

    Side Effects:
        Resolving a reference imports modules and may execute trusted source.
    """

    return obj.resolve() if isinstance(obj, (ImportRef, SourceSpec)) else obj


def resolve_function(obj: ImportRef | SourceSpec | FunctionType):
    """Resolve a symbol reference or validate and return a live Python function.

    Args:
        obj: Import/source reference or Python function.

    Returns:
        The resolved or original function.

    Raises:
        TypeError: If ``obj`` is neither a reference nor a Python function.

    Side Effects:
        Resolving a reference imports modules and may execute trusted source.
    """

    if isinstance(obj, (ImportRef, SourceSpec)):
        return obj.resolve()
    if inspect.isfunction(obj):
        return obj
    raise TypeError(f"Cannot resolve object of type {type(obj).__name__} as a function.")


__all__ = ["ImportRef", "SourceSpec", "maybe_symbol_ref", "resolve_function", "resolve_symbol", "symbol_ref"]
