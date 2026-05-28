import ast
from dataclasses import dataclass

@dataclass(frozen=True)
class AttrAccess:
    root: str
    chain: tuple[str, ...]
    ctx: str          # "load" | "store" | "del"
    lineno: int | None
    col_offset: int | None

@dataclass(frozen=True)
class MethodCall:
    root: str
    chain: tuple[str, ...]    # e.g. ("foo", "bar") for self.foo.bar(...)
    lineno: int | None
    col_offset: int | None

def _flatten_attr(node):
    chain = []
    cur = node
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        chain.reverse()
        return cur.id, tuple(chain)
    return None


class AccessCollector(ast.NodeVisitor):
    def __init__(self):
        self.attr_accesses: list[AttrAccess] = []
        self.method_calls: list[MethodCall] = []

    def visit_Attribute(self, node: ast.Attribute):
        flat = _flatten_attr(node)
        if flat is not None:
            root, chain = flat
            if isinstance(node.ctx, ast.Load):
                ctx = "load"
            elif isinstance(node.ctx, ast.Store):
                ctx = "store"
            elif isinstance(node.ctx, ast.Del):
                ctx = "del"
            else:
                ctx = type(node.ctx).__name__.lower()

            self.attr_accesses.append(
                AttrAccess(
                    root=root,
                    chain=chain,
                    ctx=ctx,
                    lineno=getattr(node, "lineno", None),
                    col_offset=getattr(node, "col_offset", None),
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        flat = _flatten_attr(node.func)
        if flat is not None:
            root, chain = flat
            self.method_calls.append(
                MethodCall(
                    root=root,
                    chain=chain,
                    lineno=getattr(node, "lineno", None),
                    col_offset=getattr(node, "col_offset", None),
                )
            )
        self.generic_visit(node)


def collect_accesses_from_source(source: str):
    tree = ast.parse(source)
    coll = AccessCollector()
    coll.visit(tree)
    return coll
