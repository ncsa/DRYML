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
    exact_root: Any | None = None

    def __post_init__(self) -> None:
        from .definition import ConcreteDefinition, Definition

        if not isinstance(self.root, (Definition, ConcreteDefinition)):
            self_root = Definition(self.root) if isinstance(self.root, type) else self.root
            if not isinstance(self_root, (Definition, ConcreteDefinition)):
                raise TypeError(f"Selector root must be a Definition, got {type(self.root).__name__}.")
            object.__setattr__(self, "root", self_root)

        if self.exact_root is None and isinstance(self.root, ConcreteDefinition):
            object.__setattr__(self, "exact_root", self.root)
        elif self.exact_root is not None and not isinstance(
            self.exact_root, ConcreteDefinition
        ):
            raise TypeError("Selector exact_root must be a ConcreteDefinition.")

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

    def categorical(
        self, *, path="$", recursive=False, drop=(), drop_args=False, drop_class=False
    ) -> "Selector":
        """Return a relaxed immutable selector while retaining exact authority.

        Args:
            path: Root-relative selector occurrence to project.
            recursive: Whether to project nested definitions.
            drop: Supplied parameter names to omit at the selected occurrence.
            drop_args: Whether to omit all argument constraints.
            drop_class: Whether to omit the selected class constraint.

        Returns:
            A selector whose original complete CDef, when available, is retained
            for a later :meth:`exact` or :meth:`restore` edit.

        Raises:
            QueryPathError: If the selected occurrence is unavailable.
            TypeError: If projection controls are invalid.

        Side Effects:
            None. This performs selector-only editing and never constructs an
            Object or accesses Store authority.
        """

        from .categorical import _project_categorical_definition_with_origins
        from .query.path import get_subtree, normalize_path, replace_subtree

        norm = normalize_path(path)
        subtree = get_subtree(self.root, norm)
        projected, origins = _project_categorical_definition_with_origins(
            subtree,
            recursive=recursive,
            drop=drop,
            drop_args=drop_args,
            drop_class=drop_class,
        )
        return Selector(
            replace_subtree(self.root, norm, projected, _origins=origins),
            strict=self.strict,
            cls_policy=self.cls_policy,
            exact_root=self.exact_root,
        )

    def restore(self, *, path="$") -> "Selector":
        """Restore one selector subtree from retained exact CDef authority.

        Args:
            path: Root-relative subtree to restore.

        Returns:
            A selector with the original subtree restored.

        Raises:
            QueryPathError: If no original exact CDef is available or ``path``
                cannot be resolved.

        Side Effects:
            None. Restoration is an immutable selector edit.
        """

        from .query.path import QueryPathError, get_subtree, normalize_path, replace_subtree

        if self.exact_root is None:
            raise QueryPathError("restore() requires an original ConcreteDefinition.")
        norm = normalize_path(path)
        return Selector(
            replace_subtree(self.root, norm, get_subtree(self.exact_root, norm)),
            strict=self.strict,
            cls_policy=self.cls_policy,
            exact_root=self.exact_root,
        )

    def exact(self, definition=None, *, path="$") -> "Selector":
        """Replace one selector subtree with an explicit exact CDef.

        Args:
            definition: Replacement ConcreteDefinition, or ``None`` to restore
                the corresponding subtree from retained exact authority.
            path: Root-relative selected subtree.

        Returns:
            A selector carrying the exact replacement.

        Raises:
            QueryPathError: If no exact authority exists for an implicit edit.
            TypeError: If the replacement is not a ConcreteDefinition.

        Side Effects:
            None. This does not fill defaults or construct a target.
        """

        from .definition import ConcreteDefinition
        from .query.path import QueryPathError, get_subtree, normalize_path, replace_subtree

        norm = normalize_path(path)
        if definition is None:
            if self.exact_root is None:
                raise QueryPathError("exact() requires an explicit ConcreteDefinition.")
            definition = get_subtree(self.exact_root, norm)
        if not isinstance(definition, ConcreteDefinition):
            raise TypeError("exact() requires a ConcreteDefinition.")
        return Selector(
            replace_subtree(self.root, norm, definition),
            strict=self.strict,
            cls_policy=self.cls_policy,
            exact_root=self.exact_root,
        )


def selector(root: Any = None, **kwargs) -> Selector:
    from .definition import Definition

    if isinstance(root, Definition):
        return Selector(root, **kwargs)
    if root is None:
        return Selector(Definition(), **kwargs)
    return Selector(Definition(root), **kwargs)
