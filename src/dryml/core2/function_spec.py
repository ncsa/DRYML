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


class _FunctionScopeNameCollector(ast.NodeVisitor):
    def __init__(self):
        self.bound: set[str] = set()
        self.used: set[str] = set()
        self._depth = 0

    def _bind_target(self, node):
        if isinstance(node, ast.Name):
            self.bound.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._bind_target(elt)

    def visit_FunctionDef(self, node):
        if self._depth > 0:
            self.bound.add(node.name)
            return

        self._depth += 1
        self.bound.add(node.name)

        for arg in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        ):
            self.bound.add(arg.arg)

        if node.args.vararg:
            self.bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.bound.add(node.args.kwarg.arg)

        for stmt in node.body:
            self.visit(stmt)

        self._depth -= 1

    def visit_Lambda(self, node):
        if self._depth > 0:
            return

        self._depth += 1
        for arg in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        ):
            self.bound.add(arg.arg)

        if node.args.vararg:
            self.bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.bound.add(node.args.kwarg.arg)

        self.visit(node.body)
        self._depth -= 1

    def visit_ClassDef(self, node):
        self.bound.add(node.name)

    def visit_Import(self, node):
        for alias in node.names:
            self.bound.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)

    def visit_For(self, node):
        self._bind_target(node.target)
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node):
        self.visit_For(node)

    def visit_With(self, node):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node):
        self.visit_With(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_NamedExpr(self, node):
        self.visit(node.value)
        self._bind_target(node.target)

    def free_names(self) -> set[str]:
        builtins_ns = set(dir(builtins))
        return self.used - self.bound - builtins_ns


def _collect_source_imports(
    fn: FunctionType,
    source: str,
) -> dict[str, str]:
    tree = ast.parse(source)

    if len(tree.body) != 1:
        raise ValueError("Expected a single function expression or definition.")

    root = tree.body[0]
    if isinstance(root, ast.Expr) and isinstance(root.value, ast.Lambda):
        node = root.value
    elif isinstance(root, ast.FunctionDef):
        node = root
    else:
        raise ValueError("Unsupported source form for FunctionSpec.")

    collector = _FunctionScopeNameCollector()
    collector.visit(node)

    imports: dict[str, str] = {}
    missing: list[str] = []

    for name in sorted(collector.free_names()):
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
