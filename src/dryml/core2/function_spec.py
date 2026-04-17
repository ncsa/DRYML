from __future__ import annotations

from dataclasses import dataclass
import ast
import importlib
import inspect
import textwrap
from types import FunctionType
from typing import Literal


FunctionSpecKind = Literal["import", "source"]


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
        return FunctionSpec(kind="source", source=lambda_source, name=None)

    source, name = _extract_named_function_source(fn, raw_source)
    return FunctionSpec(kind="source", source=source, name=name)


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

    def __post_init__(self):
        if self.kind not in ("import", "source"):
            raise ValueError(f"Invalid FunctionSpec kind {self.kind!r}.")

        if self.kind == "import":
            if not self.module or not self.qualname:
                raise ValueError(
                    "Import FunctionSpec requires `module` and `qualname`."
                )
            if self.source is not None or self.name is not None:
                raise ValueError(
                    "Import FunctionSpec may not define `source` or `name`."
                )

        elif self.kind == "source":
            if not self.source:
                raise ValueError("Source FunctionSpec requires `source`.")
            object.__setattr__(self, "source", _normalize_source(self.source))

            if self.module is not None or self.qualname is not None:
                raise ValueError(
                    "Source FunctionSpec may not define `module` or `qualname`."
                )

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
            ns: dict[str, object] = {}

            if self.name is None:
                # treat as expression, e.g. lambda x: x + 1
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
        return str(self).encode('utf-8')


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
