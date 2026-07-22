from __future__ import annotations

import builtins
import json
import __main__
from dataclasses import dataclass
import ast
import importlib
import inspect
import textwrap
from types import FunctionType
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
            raise ValueError(
                f"Cannot resolve qualname {qualname!r}; it contains <locals>."
            )
        obj = getattr(obj, part)
    return obj


@dataclass(frozen=True, slots=True)
class ImportRef:
    """Canonical reference to an importable Python symbol."""

    module: str
    qualname: str | None = None

    def __post_init__(self):
        if not self.module or not isinstance(self.module, str):
            raise ValueError("ImportRef requires a non-empty module string.")
        if self.qualname is not None and (not self.qualname or not isinstance(self.qualname, str)):
            raise ValueError("ImportRef qualname must be a non-empty string or None.")

    @classmethod
    def from_import_path(cls, path: str) -> "ImportRef":
        path = path.strip()
        if not path:
            raise ValueError("Import path cannot be empty.")
        if ":" not in path:
            return cls(module=path, qualname=None)
        module, qualname = path.split(":", 1)
        module = module.strip()
        qualname = qualname.strip()
        if not module or not qualname:
            raise ValueError("Import path must be 'module' or 'module:qualname'.")
        return cls(module=module, qualname=qualname)

    @classmethod
    def from_object(cls, obj) -> "ImportRef":
        ref = _object_import_ref(obj)
        if ref is None:
            raise ValueError(
                f"Object of type {type(obj).__name__} does not have a stable import path."
            )
        return ref

    def import_path(self) -> str:
        if self.qualname is None:
            return self.module
        return f"{self.module}:{self.qualname}"

    def resolve(self):
        module = importlib.import_module(self.module)
        if self.qualname is None:
            return module
        return _resolve_qualname(module, self.qualname)

    def __repr__(self) -> str:
        if self.qualname is None:
            return f"ImportRef(module={self.module!r})"
        return f"ImportRef(module={self.module!r}, qualname={self.qualname!r})"

    def __stable_leaf_bytes__(self):
        payload = {
            "kind": "import",
            "module": self.module,
            "qualname": self.qualname,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


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

    if not module_name or not qualname:
        return None

    if module_name == "__main__":
        return None

    if "<locals>" in qualname:
        return None

    try:
        module = importlib.import_module(module_name)
        resolved = _resolve_qualname(module, qualname)
    except Exception:
        return None

    if resolved is obj:
        return ImportRef(module_name, qualname)

    return None


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Source-backed specification for non-importable functions and classes."""

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
        imports = {
            name: _coerce_import_ref(ref)
            for name, ref in (self.imports or {}).items()
        }
        object.__setattr__(self, "imports", dict(sorted(imports.items())))

    @classmethod
    def from_function(cls, fn: FunctionType) -> "SourceSpec":
        if not inspect.isfunction(fn):
            raise TypeError(f"Expected a Python function, got {type(fn).__name__}.")
        return _source_spec_from_function(fn)

    @classmethod
    def from_class(cls, obj: type) -> "SourceSpec":
        if not inspect.isclass(obj):
            raise TypeError(f"Expected a Python class, got {type(obj).__name__}.")
        return _source_spec_from_class(obj)

    @classmethod
    def from_source(
        cls,
        source: str,
        *,
        kind: SourceSpecKind = "function",
        name: str | None = None,
        imports: dict[str, ImportRef | str] | None = None,
    ) -> "SourceSpec":
        if kind == "class" and name is None:
            source_tree = ast.parse(_normalize_source(source))
            if len(source_tree.body) == 1 and isinstance(source_tree.body[0], ast.ClassDef):
                name = source_tree.body[0].name
        return cls(kind=kind, source=source, name=name, imports=imports)

    def resolve(self):
        live = _matching_live_source(self)
        if live is not None:
            return live

        ns: dict[str, object] = {}
        for name, ref in (self.imports or {}).items():
            ns[name] = ref.resolve()

        if self.kind == "function" and self.name is None:
            obj = eval(self.source, ns, ns)
        else:
            exec(self.source, ns, ns)
            if self.name not in ns:
                raise ValueError(
                    f"Resolved namespace does not contain symbol {self.name!r}."
                )
            obj = ns[self.name]

        if self.kind == "class" and not inspect.isclass(obj):
            raise TypeError("Resolved SourceSpec object is not a class.")
        if self.kind == "function" and not callable(obj):
            raise TypeError("Resolved SourceSpec object is not callable.")
        return obj

    def __repr__(self) -> str:
        return (
            f"SourceSpec(kind={self.kind!r}, name={self.name!r}, "
            f"source={self.source!r}, imports={self.imports!r})"
        )

    def __stable_leaf_bytes__(self):
        payload = {
            "kind": self.kind,
            "source": self.source,
            "name": self.name,
            "imports": {
                name: ref.import_path()
                for name, ref in sorted((self.imports or {}).items())
            },
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _extract_lambda_source(source: str) -> str:
    """
    Best-effort extraction of a lambda expression from source returned by
    inspect.getsource(...).

    Supported common forms:
      lambda x: x + 1
      f = lambda x: x + 1
      obj = some_call(lambda x: x + 1)   # not supported, intentionally
    """
    source = _normalize_source(source)
    tree = ast.parse(source)

    if len(tree.body) != 1:
        raise ValueError("Lambda source is ambiguous.")

    stmt = tree.body[0]

    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Lambda):
        return ast.unparse(stmt.value)

    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Lambda):
        return ast.unparse(stmt.value)

    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Lambda):
        return ast.unparse(stmt.value)

    raise ValueError("Could not extract a standalone lambda expression.")


def _extract_named_function_source(fn: FunctionType, source: str) -> tuple[str, str]:
    """
    Extract a normalized `def ...` block for a named function.

    Decorators are stripped to avoid requiring decorator symbols at resolve time.
    """
    source = _normalize_source(source)
    tree = ast.parse(source)

    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == fn.__name__:
            stmt.decorator_list = []
            return ast.unparse(stmt), stmt.name

    raise ValueError(
        f"Could not find function definition for {fn.__name__!r} in extracted source."
    )


def _source_spec_from_function(fn: FunctionType) -> SourceSpec:
    """
    Build a source-backed SourceSpec from a live function.

    This is intentionally conservative:
    - closures are rejected for now because environment capture is disabled
    - globals are not captured; source-backed functions should be self-contained
    """
    if fn.__code__.co_freevars:
        raise ValueError(
            "Cannot create SourceSpec for a closure; "
            "environment capture is not implemented yet."
        )

    try:
        raw_source = inspect.getsource(fn)
    except (OSError, IOError) as e:
        raise ValueError("Could not retrieve function source.") from e

    if fn.__name__ == "<lambda>":
        lambda_source = _extract_lambda_source(raw_source)
        imports = _collect_source_imports(fn, lambda_source)
        return SourceSpec(
            kind="function",
            source=lambda_source,
            name=None,
            imports=imports,
        )

    source, name = _extract_named_function_source(fn, raw_source)
    imports = _collect_source_imports(fn, source)
    return SourceSpec(
        kind="function",
        source=source,
        name=name,
        imports=imports,
    )


def _extract_named_class_source(cls: type, source: str) -> tuple[str, str]:
    source = _normalize_source(source)
    tree = ast.parse(source)

    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == cls.__name__:
            stmt.decorator_list = []
            return ast.unparse(stmt), stmt.name

    raise ValueError(
        f"Could not find class definition for {cls.__name__!r} in extracted source."
    )


def _source_spec_from_class(cls: type) -> SourceSpec:
    try:
        raw_source = inspect.getsource(cls)
    except (OSError, IOError) as e:
        raise ValueError("Could not retrieve class source.") from e

    source, name = _extract_named_class_source(cls, raw_source)
    imports = _collect_source_imports(cls, source)
    return SourceSpec(
        kind="class",
        source=source,
        name=name,
        imports=imports,
    )


def _lookup_live_name_for_source(obj, name: str):
    """
    Best-effort resolution of a name referenced by function source.

    Order:
    1. function globals
    2. active caller-frame locals, nearest first

    This supports notebook/test-style nested definitions that rely on names
    imported in an enclosing local scope.
    """
    globals_dict = getattr(obj, "__globals__", None)
    if globals_dict is not None and name in globals_dict:
        return globals_dict[name]

    module_name = getattr(obj, "__module__", None)
    if module_name:
        import sys

        module = sys.modules.get(module_name)
        if module is not None and name in vars(module):
            return vars(module)[name]

    frame = inspect.currentframe()
    try:
        # Skip this helper frame and walk outward.
        frame = frame.f_back
        while frame is not None:
            if name in frame.f_locals:
                return frame.f_locals[name]
            frame = frame.f_back
    finally:
        # Avoid reference cycles.
        del frame

    raise KeyError(name)


@dataclass(frozen=True)
class _ScopeDeps:
    bound_here: set[str]
    needed_globals: set[str]


class _LexicalDependencyCollector:
    """
    Compute external global names required by a serialized scope tree.

    Key rule:
    - function header expressions (defaults, annotations, decorators, returns)
      are evaluated in the enclosing definition scope
    - function body expressions are evaluated in the function-local lexical scope
    """

    def __init__(self):
        self._builtins = set(dir(builtins))

    def collect(self, node: ast.AST) -> set[str]:
        deps = self._collect_scope(node, available_from_parents=set())
        return deps.needed_globals

    def _collect_scope(
        self,
        node: ast.AST,
        available_from_parents: set[str],
    ) -> _ScopeDeps:
        needed_globals: set[str] = set()

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # ----------------------------------------------------------
            # Phase 1: header/declaration-time expressions
            # These are evaluated in the enclosing scope, NOT in the
            # function-local scope.
            # ----------------------------------------------------------
            header_bound: set[str] = set()
            header_used: set[str] = set()

            def walk_header(n: ast.AST | None):
                if n is None:
                    return
                self._walk_node(
                    n,
                    bound_here=header_bound,
                    used_here=header_used,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )

            for dec in node.decorator_list:
                walk_header(dec)

            self._walk_function_header_annotations_and_defaults(
                node,
                walk_header,
            )

            unresolved_header = (
                header_used
                - header_bound
                - available_from_parents
                - self._builtins
            )
            needed_globals |= unresolved_header

            # ----------------------------------------------------------
            # Phase 2: body/local lexical scope
            # ----------------------------------------------------------
            body_bound: set[str] = {node.name}
            self._bind_arguments(node.args, body_bound)
            body_used: set[str] = set()

            for stmt in node.body:
                self._walk_node(
                    stmt,
                    bound_here=body_bound,
                    used_here=body_used,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents | body_bound,
                )

            unresolved_body = (
                body_used
                - body_bound
                - available_from_parents
                - self._builtins
            )
            needed_globals |= unresolved_body

            return _ScopeDeps(
                bound_here=body_bound,
                needed_globals=needed_globals,
            )

        elif isinstance(node, ast.Lambda):
            bound_here: set[str] = set()
            self._bind_arguments(node.args, bound_here)
            used_here: set[str] = set()

            self._walk_node(
                node.body,
                bound_here=bound_here,
                used_here=used_here,
                needed_globals=needed_globals,
                available_from_parents=available_from_parents | bound_here,
            )

            unresolved = (
                used_here
                - bound_here
                - available_from_parents
                - self._builtins
            )
            needed_globals |= unresolved

            return _ScopeDeps(
                bound_here=bound_here,
                needed_globals=needed_globals,
            )

        elif isinstance(node, ast.ClassDef):
            # Minimal class support:
            # bases/keywords/decorators are evaluated in enclosing scope
            header_bound: set[str] = set()
            header_used: set[str] = set()

            def walk_header(n: ast.AST | None):
                if n is None:
                    return
                self._walk_node(
                    n,
                    bound_here=header_bound,
                    used_here=header_used,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )

            for dec in node.decorator_list:
                walk_header(dec)
            for base in node.bases:
                walk_header(base)
            for kw in node.keywords:
                walk_header(kw.value)

            unresolved_header = (
                header_used
                - header_bound
                - available_from_parents
                - self._builtins
            )
            needed_globals |= unresolved_header

            body_bound: set[str] = set()
            body_used: set[str] = set()

            for stmt in node.body:
                self._walk_node(
                    stmt,
                    bound_here=body_bound,
                    used_here=body_used,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents | body_bound,
                )

            unresolved_body = (
                body_used
                - body_bound
                - available_from_parents
                - self._builtins
            )
            needed_globals |= unresolved_body

            return _ScopeDeps(
                bound_here={node.name},
                needed_globals=needed_globals,
            )

        else:
            raise TypeError(f"Unsupported scope root {type(node).__name__}")

    def _walk_function_header_annotations_and_defaults(self, node, walk_header):
        args = node.args

        for arg in args.posonlyargs:
            walk_header(arg.annotation)
        for arg in args.args:
            walk_header(arg.annotation)
        if args.vararg:
            walk_header(args.vararg.annotation)
        for arg in args.kwonlyargs:
            walk_header(arg.annotation)
        if args.kwarg:
            walk_header(args.kwarg.annotation)

        for default in args.defaults:
            walk_header(default)

        for default in args.kw_defaults:
            walk_header(default)

        walk_header(node.returns)

    def _walk_node(
        self,
        n: ast.AST | None,
        *,
        bound_here: set[str],
        used_here: set[str],
        needed_globals: set[str],
        available_from_parents: set[str],
    ):
        if n is None:
            return

        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound_here.add(n.name)

            if n.decorator_list:
                raise ValueError(
                    f"Decorated nested function {n.name!r} is not supported."
                )

            child = self._collect_scope(
                n,
                available_from_parents=available_from_parents | bound_here,
            )
            needed_globals |= child.needed_globals
            return

        if isinstance(n, ast.ClassDef):
            bound_here.add(n.name)

            if n.decorator_list:
                raise ValueError(
                    f"Decorated nested class {n.name!r} is not supported."
                )

            child = self._collect_scope(
                n,
                available_from_parents=available_from_parents | bound_here,
            )
            needed_globals |= child.needed_globals
            return

        if isinstance(n, ast.Lambda):
            child = self._collect_scope(
                n,
                available_from_parents=available_from_parents | bound_here,
            )
            needed_globals |= child.needed_globals
            return

        if isinstance(n, ast.Import):
            for alias in n.names:
                bound_here.add(alias.asname or alias.name.split(".", 1)[0])
            return

        if isinstance(n, ast.ImportFrom):
            for alias in n.names:
                bound_here.add(alias.asname or alias.name)
            return

        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Load):
                used_here.add(n.id)
            elif isinstance(n.ctx, (ast.Store, ast.Del)):
                bound_here.add(n.id)
            return

        if isinstance(n, ast.For):
            self._walk_node(
                n.iter,
                bound_here=bound_here,
                used_here=used_here,
                needed_globals=needed_globals,
                available_from_parents=available_from_parents,
            )
            self._bind_target(n.target, bound_here)
            for stmt in n.body:
                self._walk_node(
                    stmt,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
            for stmt in n.orelse:
                self._walk_node(
                    stmt,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
            return

        if isinstance(n, ast.AsyncFor):
            self._walk_node(
                n.iter,
                bound_here=bound_here,
                used_here=used_here,
                needed_globals=needed_globals,
                available_from_parents=available_from_parents,
            )
            self._bind_target(n.target, bound_here)
            for stmt in n.body:
                self._walk_node(
                    stmt,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
            for stmt in n.orelse:
                self._walk_node(
                    stmt,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
            return

        if isinstance(n, ast.With):
            for item in n.items:
                self._walk_node(
                    item.context_expr,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
                if item.optional_vars is not None:
                    self._bind_target(item.optional_vars, bound_here)
            for stmt in n.body:
                self._walk_node(
                    stmt,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
            return

        if isinstance(n, ast.AsyncWith):
            for item in n.items:
                self._walk_node(
                    item.context_expr,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
                if item.optional_vars is not None:
                    self._bind_target(item.optional_vars, bound_here)
            for stmt in n.body:
                self._walk_node(
                    stmt,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
            return

        if isinstance(n, ast.ExceptHandler):
            if n.type is not None:
                self._walk_node(
                    n.type,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
            if n.name:
                bound_here.add(n.name)
            for stmt in n.body:
                self._walk_node(
                    stmt,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
            return

        if isinstance(n, ast.NamedExpr):
            self._walk_node(
                n.value,
                bound_here=bound_here,
                used_here=used_here,
                needed_globals=needed_globals,
                available_from_parents=available_from_parents,
            )
            self._bind_target(n.target, bound_here)
            return

        if isinstance(n, ast.comprehension):
            self._walk_node(
                n.iter,
                bound_here=bound_here,
                used_here=used_here,
                needed_globals=needed_globals,
                available_from_parents=available_from_parents,
            )
            self._bind_target(n.target, bound_here)
            for if_ in n.ifs:
                self._walk_node(
                    if_,
                    bound_here=bound_here,
                    used_here=used_here,
                    needed_globals=needed_globals,
                    available_from_parents=available_from_parents,
                )
            return

        if isinstance(n, (ast.Global, ast.Nonlocal)):
            return

        for child in ast.iter_child_nodes(n):
            self._walk_node(
                child,
                bound_here=bound_here,
                used_here=used_here,
                needed_globals=needed_globals,
                available_from_parents=available_from_parents,
            )

    def _bind_arguments(self, args: ast.arguments, bound: set[str]):
        for arg in (
            list(args.posonlyargs)
            + list(args.args)
            + list(args.kwonlyargs)
        ):
            bound.add(arg.arg)

        if args.vararg:
            bound.add(args.vararg.arg)
        if args.kwarg:
            bound.add(args.kwarg.arg)

    def _bind_target(self, node: ast.AST, bound: set[str]):
        if isinstance(node, ast.Name):
            bound.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._bind_target(elt, bound)
        elif isinstance(node, ast.Starred):
            self._bind_target(node.value, bound)


def _collect_source_imports(
    obj,
    source: str,
) -> dict[str, ImportRef]:
    tree = ast.parse(source)

    if len(tree.body) != 1:
        raise ValueError("Expected a single function/class expression or definition.")

    root = tree.body[0]
    if isinstance(root, ast.Expr) and isinstance(root.value, ast.Lambda):
        scope_node = root.value
    elif isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        scope_node = root
    else:
        raise ValueError("Unsupported source form for SourceSpec.")

    collector = _LexicalDependencyCollector()
    free_names = collector.collect(scope_node)

    imports: dict[str, ImportRef] = {}
    missing: list[str] = []

    for name in sorted(free_names):
        try:
            dep = _lookup_live_name_for_source(obj, name)
        except KeyError:
            missing.append(name)
            continue

        ref = _object_import_ref(dep)
        if ref is None:
            missing.append(name)
            continue

        imports[name] = ref

    if missing:
        raise ValueError(
            "Could not capture stable import paths for source-backed object "
            f"{getattr(obj, '__name__', type(obj).__name__)!r}. Missing/unimportable globals: {missing}"
        )

    return imports


def _current_main_ns() -> dict[str, object]:
    try:
        return vars(__main__)
    except Exception:
        return {}


def _matching_live_source(spec: SourceSpec):
    if spec.name is None:
        return None

    ns = _current_main_ns()
    candidate = ns.get(spec.name)

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

    if (
        candidate_spec.kind == spec.kind
        and candidate_spec.name == spec.name
        and candidate_spec.source == spec.source
        and (candidate_spec.imports or {}) == (spec.imports or {})
    ):
        return candidate

    return None


def symbol_ref(
    obj: ImportRef | SourceSpec | FunctionType | type | str,
    *,
    name: str | None = None,
) -> ImportRef | SourceSpec:
    if isinstance(obj, (ImportRef, SourceSpec)):
        return obj

    if inspect.isfunction(obj):
        ref = _object_import_ref(obj)
        if ref is not None:
            return ref
        return SourceSpec.from_function(obj)

    if inspect.isclass(obj):
        ref = _object_import_ref(obj)
        if ref is not None:
            return ref
        return SourceSpec.from_class(obj)

    if isinstance(obj, str):
        return SourceSpec.from_source(obj, kind="function", name=name)

    raise TypeError(
        f"Cannot convert object of type {type(obj).__name__} to ImportRef or SourceSpec."
    )


def maybe_symbol_ref(obj: Any, *, functions: bool = True) -> ImportRef | SourceSpec | None:
    if isinstance(obj, (ImportRef, SourceSpec)):
        return obj
    if inspect.isclass(obj) or (functions and inspect.isfunction(obj)):
        try:
            return symbol_ref(obj)
        except Exception:
            return None
    return None


def resolve_symbol(obj: Any) -> Any:
    if isinstance(obj, ImportRef):
        return obj.resolve()

    if isinstance(obj, SourceSpec):
        return obj.resolve()

    return obj


def resolve_function(obj: ImportRef | SourceSpec | FunctionType):
    """
    Resolve either a symbol reference or a live function.
    """
    if isinstance(obj, (ImportRef, SourceSpec)):
        return obj.resolve()

    if inspect.isfunction(obj):
        return obj

    raise TypeError(
        f"Cannot resolve object of type {type(obj).__name__} as a function."
    )


__all__ = [
    "ImportRef",
    "SourceSpec",
    "maybe_symbol_ref",
    "resolve_function",
    "resolve_symbol",
    "symbol_ref",
]
