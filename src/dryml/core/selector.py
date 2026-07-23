from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Selector:
    """Query interpretation wrapper around an immutable Definition root."""

    root: Any
    strict: bool = False
    cls_policy: str = "selector"

    def __post_init__(self) -> None:
        from .definition import Definition

        if not isinstance(self.root, Definition):
            self_root = Definition(self.root) if isinstance(self.root, type) else self.root
            if not isinstance(self_root, Definition):
                raise TypeError(f"Selector root must be a Definition, got {type(self.root).__name__}.")
            object.__setattr__(self, "root", self_root)

    def compile(self, ctx=None):
        from .query.selector_graph import compile_selector_graph

        return compile_selector_graph(self, class_match=self.cls_policy)

    def matches(self, target: Any, *, verbose: bool = False) -> bool:
        from .query.query import _query_match


        return _query_match(self.root, target, strict=self.strict, class_match=self.cls_policy)


def selector(root: Any = None, **kwargs) -> Selector:
    from .definition import Definition

    if isinstance(root, Definition):
        return Selector(root, **kwargs)
    if root is None:
        return Selector(Definition(), **kwargs)
    return Selector(Definition(root), **kwargs)
