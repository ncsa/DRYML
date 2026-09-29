from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Selector:
    """Query interpretation wrapper around an immutable Definition root.

    Supplied semantic parameters are exposed through ``parameters`` and direct
    non-reserved attributes without changing selector omission semantics.
    """

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

    @property
    def parameters(self):
        """Return the immutable supplied semantic parameters of ``root``.

        Returns:
            The partial parameter record supplied to the wrapped Definition.
        """

        return self.root.parameters

    def __getattr__(self, name: str) -> Any:
        """Delegate non-reserved semantic parameter access to ``root``.

        Args:
            name: Requested Python attribute name.

        Returns:
            The supplied frozen semantic value from the wrapped Definition.

        Raises:
            AttributeError: If ``name`` is not supplied on the wrapped root.
        """

        try:
            return getattr(self.root, name)
        except AttributeError as error:
            raise AttributeError(
                f"{type(self).__name__!s} object has no attribute {name!r}"
            ) from error

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
