from __future__ import annotations

import builtins
import json
import types
import __main__
from dataclasses import dataclass
import ast
import importlib
import inspect
import textwrap
from types import FunctionType
from typing import Literal


FunctionSpecKind = Literal["import", "source"]


def _object_import_path(obj) -> str | None:
    """
    Return a stable import path for modules and importable objects.

    Forms:
      'numpy'
      'numpy:sin'
      'dryml.core.function_spec:FunctionSpec'
    """
    if inspect.ismodule(obj):
        name = getattr(obj, "__name__", None)
        return name if name else None

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
        return f"{module_name}:{qualname}"

    return None


def _resolve_import_path(path: str):
    if ":" not in path:
        return importlib.import_module(path)

    module_name, qualname = path.split(":", 1)
    module = importlib.import_module(module_name)
    return _resolve_qualname(module, qualname)


def _normalize_source(source: str) -> str:
    source = textwrap.dedent(source).strip()
    if not source:
        raise ValueError("Function source cannot be empty.")
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


def _has_stable_import_path(fn: FunctionType) -> bool:
    """
    Conservative check for whether this function can be reconstructed by
    importing module + qualname in another process.

    This intentionally rejects many cases:
    - __main__
    - notebook cell defs
    - nested defs (<locals>)
    """
    if not inspect.isfunction(fn):
        return False

    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)

    if not module_name or not qualname:
        return False

    if module_name == "__main__":
        return False

    if "<locals>" in qualname:
        return False

    try:
        module = importlib.import_module(module_name)
        resolved = _resolve_qualname(module, qualname)
    except Exception:
        return False

    return resolved is fn


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


def _source_spec_from_function(fn: FunctionType):
    """
    Build a source-backed FunctionSpec from a live function.

    This is intentionally conservative:
    - closures are rejected for now because environment capture is disabled
    - globals are not captured; source-backed functions should be self-contained
    """
    if fn.__code__.co_freevars:
        raise ValueError(
            "Cannot create source-backed FunctionSpec for a closure; "
            "environment capture is not implemented yet."
        )

    try:
        raw_source = inspect.getsource(fn)
    except (OSError, IOError) as e:
        raise ValueError("Could not retrieve function source.") from e

    if fn.__name__ == "<lambda>":
        lambda_source = _extract_lambda_source(raw_source)
        imports = _collect_source_imports(fn, lambda_source)
        return FunctionSpec(
            kind="source",
            source=lambda_source,
            name=None,
            imports=imports,
        )

    source, name = _extract_named_function_source(fn, raw_source)
    imports = _collect_source_imports(fn, source)
    return FunctionSpec(
        kind="source",
        source=source,
        name=name,
        imports=imports,
    )


@dataclass(frozen=True)
class _ScopeDeps:
    bound_here: set[str]
    needed_globals: set[str]


class _LexicalDependencyCollector:
    """
    Compute external global names required by a serialized scope tree.

    A nested scope may use names bound in an enclosing serialized scope without
    requiring those names to be captured externally.
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
        bound_here = self._initial_bound_names(node)
        used_here: set[str] = set()
        needed_globals: set[str] = set()

        def walk(n: ast.AST | None):
            nonlocal needed_globals
            if n is None:
                return

            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Nested function name is bound in the current scope.
                bound_here.add(n.name)

                # Optional: fail early on nested decorated defs.
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
                walk(n.iter)
                self._bind_target(n.target, bound_here)
                for stmt in n.body:
                    walk(stmt)
                for stmt in n.orelse:
                    walk(stmt)
                return

            if isinstance(n, ast.AsyncFor):
                walk(n.iter)
                self._bind_target(n.target, bound_here)
                for stmt in n.body:
                    walk(stmt)
                for stmt in n.orelse:
                    walk(stmt)
                return

            if isinstance(n, ast.With):
                for item in n.items:
                    walk(item.context_expr)
                    if item.optional_vars is not None:
                        self._bind_target(item.optional_vars, bound_here)
                for stmt in n.body:
                    walk(stmt)
                return

            if isinstance(n, ast.AsyncWith):
                for item in n.items:
                    walk(item.context_expr)
                    if item.optional_vars is not None:
                        self._bind_target(item.optional_vars, bound_here)
                for stmt in n.body:
                    walk(stmt)
                return

            if isinstance(n, ast.ExceptHandler):
                if n.type is not None:
                    walk(n.type)
                if n.name:
                    bound_here.add(n.name)
                for stmt in n.body:
                    walk(stmt)
                return

            if isinstance(n, ast.NamedExpr):
                walk(n.value)
                self._bind_target(n.target, bound_here)
                return

            if isinstance(n, ast.comprehension):
                walk(n.iter)
                self._bind_target(n.target, bound_here)
                for if_ in n.ifs:
                    walk(if_)
                return

            if isinstance(n, (ast.Global, ast.Nonlocal)):
                return

            for child in ast.iter_child_nodes(n):
                walk(child)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
        elif isinstance(node, ast.Lambda):
            body = [node.body]
        else:
            raise TypeError(f"Unsupported scope root {type(node).__name__}")

        for item in body:
            walk(item)

        unresolved_here = (
            used_here
            - bound_here
            - available_from_parents
            - self._builtins
        )

        needed_globals |= unresolved_here
        return _ScopeDeps(bound_here=bound_here, needed_globals=needed_globals)

    def _initial_bound_names(self, node: ast.AST) -> set[str]:
        bound: set[str] = set()

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            self._bind_arguments(node.args, bound)
        elif isinstance(node, ast.Lambda):
            self._bind_arguments(node.args, bound)
        elif isinstance(node, ast.ClassDef):
            pass
        else:
            raise TypeError(f"Unsupported scope root {type(node).__name__}")

        return bound

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
    fn: FunctionType,
    source: str,
) -> dict[str, str]:
    tree = ast.parse(source)

    if len(tree.body) != 1:
        raise ValueError("Expected a single function expression or definition.")

    root = tree.body[0]
    if isinstance(root, ast.Expr) and isinstance(root.value, ast.Lambda):
        scope_node = root.value
    elif isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
        scope_node = root
    else:
        raise ValueError("Unsupported source form for FunctionSpec.")

    collector = _LexicalDependencyCollector()
    free_names = collector.collect(scope_node)

    imports: dict[str, str] = {}
    missing: list[str] = []

    for name in sorted(free_names):
        if name not in fn.__globals__:
            missing.append(name)
            continue

        obj = fn.__globals__[name]
        path = _object_import_path(obj)
        if path is None:
            missing.append(name)
            continue

        imports[name] = path

    if missing:
        raise ValueError(
            "Could not capture stable import paths for source-backed function "
            f"{fn.__name__!r}. Missing/unimportable globals: {missing}"
        )

    return imports


def _current_main_ns() -> dict[str, object]:
    try:
        return vars(__main__)
    except Exception:
        return {}


def _matching_live_function(spec: "FunctionSpec"):
    if spec.kind != "source" or spec.name is None:
        return None

    ns = _current_main_ns()
    candidate = ns.get(spec.name)

    if not inspect.isfunction(candidate):
        return None

    try:
        candidate_spec = FunctionSpec.from_function(candidate)
    except Exception:
        return None

    if (
        candidate_spec.kind == "source"
        and candidate_spec.name == spec.name
        and candidate_spec.source == spec.source
        and (candidate_spec.imports or {}) == (spec.imports or {})
    ):
        return candidate

    return None

@dataclass(frozen=True, slots=True)
class FunctionSpec:
    """
    Canonical specification for reconstructing a function.

    Modes
    -----
    import
        Reconstruct by importing module + qualname.

    source
        Reconstruct by exec/eval on a source string. Since no environment is
        captured, source-backed functions should be self-contained.
    """
    kind: FunctionSpecKind

    # import mode
    module: str | None = None
    qualname: str | None = None

    # source mode
    source: str | None = None
    name: str | None = None  # required for `def ...`; None allowed for lambda expr
    imports: dict[str,str] | None = None

    def __post_init__(self):
        if self.kind not in ("import", "source"):
            raise ValueError(f"Invalid FunctionSpec kind {self.kind!r}.")

        if self.kind == "import":
            if not self.module or not self.qualname:
                raise ValueError(
                    "Import FunctionSpec requires `module` and `qualname`."
                )
            if self.source is not None or self.name is not None or self.imports is not None:
                raise ValueError(
                    "Import FunctionSpec may not define `source`, `name`, or `imports`."
                )

        elif self.kind == "source":
            if not self.source:
                raise ValueError("Source FunctionSpec requires `source`.")
            object.__setattr__(self, "source", _normalize_source(self.source))

            if self.module is not None or self.qualname is not None:
                raise ValueError(
                    "Source FunctionSpec may not define `module` or `qualname`."
                )

            imports = self.imports or {}
            object.__setattr__(self, "imports", dict(sorted(imports.items())))

    @classmethod
    def from_function(cls, fn: FunctionType) -> FunctionSpec:
        """
        Single ingest point for live functions.

        Order:
        1. stable import path
        2. source extraction
        """
        if not inspect.isfunction(fn):
            raise TypeError(
                f"Expected a Python function, got {type(fn).__name__}."
            )

        if _has_stable_import_path(fn):
            return cls(
                kind="import",
                module=fn.__module__,
                qualname=fn.__qualname__,
            )

        return _source_spec_from_function(fn)

    @classmethod
    def from_import(cls, module: str, qualname: str) -> FunctionSpec:
        return cls(kind="import", module=module, qualname=qualname)

    @classmethod
    def from_import_path(cls, path: str) -> FunctionSpec:
        """
        Parse 'package.module:qualname'
        """
        if ":" not in path:
            raise ValueError(
                "Import path must have the form 'package.module:qualname'."
            )
        module, qualname = path.split(":", 1)
        module = module.strip()
        qualname = qualname.strip()
        if not module or not qualname:
            raise ValueError(
                "Import path must have the form 'package.module:qualname'."
            )
        return cls.from_import(module, qualname)

    @classmethod
    def from_source(cls, source: str, name: str | None = None) -> FunctionSpec:
        """
        Explicit source-backed FunctionSpec.

        Examples
        --------
        FunctionSpec.from_source(
            '''
            def f(x):
                return x + 1
            ''',
            name='f',
        )

        FunctionSpec.from_source('lambda x: x + 1')
        """
        return cls(kind="source", source=source, name=name)

    def resolve(self):
        """
        Reconstruct the live function.
        """
        if self.kind == "import":
            module = importlib.import_module(self.module)
            fn = _resolve_qualname(module, self.qualname)

        elif self.kind == "source":
            live = _matching_live_function(self)
            if live is not None:
                return live

            ns: dict[str, object] = {}

            for name, path in (self.imports or {}).items():
                ns[name] = _resolve_import_path(path)

            if self.name is None:
                fn = eval(self.source, ns, ns)
            else:
                exec(self.source, ns, ns)
                if self.name not in ns:
                    raise ValueError(
                        f"Resolved namespace does not contain function {self.name!r}."
                    )
                fn = ns[self.name]

        else:
            raise RuntimeError(f"Unexpected FunctionSpec kind {self.kind!r}.")

        if not callable(fn):
            raise TypeError("Resolved object is not callable.")

        return fn

    def import_path(self) -> str:
        if self.kind != "import":
            raise ValueError("Only import FunctionSpec values have an import path.")
        return f"{self.module}:{self.qualname}"

    def __repr__(self) -> str:
        if self.kind == "import":
            return (
                f"FunctionSpec(kind='import', "
                f"module={self.module!r}, qualname={self.qualname!r})"
            )
        return (
            f"FunctionSpec(kind='source', "
            f"name={self.name!r}, source={self.source!r})"
        )

    def __stable_leaf_bytes__(self):
        payload = {
            "kind": self.kind,
            "module": self.module,
            "qualname": self.qualname,
            "source": self.source,
            "name": self.name,
            "imports": dict(sorted((self.imports or {}).items())),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def function_spec(
    obj: FunctionSpec | FunctionType | str,
    *,
    name: str | None = None,
) -> FunctionSpec:
    if isinstance(obj, FunctionSpec):
        return obj

    if inspect.isfunction(obj):
        return FunctionSpec.from_function(obj)

    if isinstance(obj, str):
        return FunctionSpec.from_source(obj, name=name)

    raise TypeError(
        f"Cannot convert object of type {type(obj).__name__} to FunctionSpec."
    )


def resolve_function(obj: FunctionSpec | FunctionType):
    """
    Resolve either a FunctionSpec or a live function.
    """
    if isinstance(obj, FunctionSpec):
        return obj.resolve()

    if inspect.isfunction(obj):
        return obj

    raise TypeError(
        f"Cannot resolve object of type {type(obj).__name__} as a function."
    )
