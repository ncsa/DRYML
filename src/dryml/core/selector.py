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
        exact_path = _exact_authority_path(self.root, self.exact_root, norm)
        return Selector(
            replace_subtree(self.root, norm, get_subtree(self.exact_root, exact_path)),
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
            exact_path = _exact_authority_path(self.root, self.exact_root, norm)
            definition = get_subtree(self.exact_root, exact_path)
        if not isinstance(definition, ConcreteDefinition):
            raise TypeError("exact() requires a ConcreteDefinition.")
        return Selector(
            replace_subtree(self.root, norm, definition),
            strict=self.strict,
            cls_policy=self.cls_policy,
            exact_root=self.exact_root,
        )


def _exact_authority_path(selector_root, exact_root, path):
    """Translate one authored selector path to retained semantic CDef segments."""

    from .query.path import DefinitionPath, get_subtree
    from .query.selector_graph import _semantic_selector_path

    selector_value = selector_root
    exact_value = exact_root
    exact_segments = []
    for segment in path:
        step = DefinitionPath((segment,))
        semantic = _semantic_selector_path(selector_value, step)
        selector_value = get_subtree(selector_value, step)
        exact_step = step if semantic is None else semantic
        exact_value = get_subtree(exact_value, exact_step)
        exact_segments.extend(exact_step.segments)
    return DefinitionPath(tuple(exact_segments))


def selector(root: Any = None, *, scope=None, **kwargs) -> Selector:
    """Create a selector and optionally pin soft state aliases to one Repo scope.

    Args:
        root: Definition-like structural selector root. ``None`` creates an
            unconstrained Definition selector.
        scope: Repo authority used to resolve every embedded StateSelectorRef
            exactly once before the selector is returned. Omit it when no soft
            state aliases are present.
        **kwargs: Selector policies accepted by :class:`Selector`.

    Returns:
        An immutable selector. When ``scope`` is supplied, it contains exact
        StateRef leaves in place of its soft state-alias leaves.

    Raises:
        TypeError: If ``root`` or ``scope`` is unsupported.
        KeyError: If a scoped state alias is absent (without echoing its value).
        ValueError: If authority lookup fails, resolution leaves the ObjectRef
            scope, or the selector graph contains a cycle.

    Side Effects:
        Scoped preparation reads current alias authority once per shared soft
        leaf. It never constructs Objects or changes Store contents.
    """

    from .definition import ConcreteDefinition, Definition

    if root is None:
        root = Definition()
    elif not isinstance(root, (Definition, ConcreteDefinition)):
        root = Definition(root)
    if scope is not None:
        root = _resolve_state_selectors(root, scope)
    return Selector(root, **kwargs)


def _contains_state_selector(value: Any) -> bool:
    """Return whether a selector graph retains an unresolved state alias leaf."""

    return _resolve_state_selectors(value, None, require_scope=False)


def _resolve_state_selectors(source: Any, scope, *, require_scope: bool = True):
    """Replace shared soft aliases while retaining selector graph topology.

    ``require_scope=False`` is a validation walk used by V3 composition; its
    boolean return says whether a soft leaf was found without reading authority.
    """

    from .cdef_graph import EdgeKind
    from .definition import Definition, SKIP_ARGS
    from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
    from .links import DefLink
    from .reference_values import StateSelectorRef

    memo = {}
    active: set[int] = set()

    def visit(value):
        if isinstance(value, StateSelectorRef):
            if not require_scope:
                return True
            key = id(value)
            if key not in memo:
                resolver = getattr(scope, "resolve_state_selector", None)
                if not callable(resolver):
                    raise TypeError(
                        "selector(value, scope=repo) requires a Repo scope for StateSelectorRef values."
                    )
                failure = None
                try:
                    resolved = resolver(value)
                except KeyError:
                    failure = "missing"
                except Exception:
                    failure = "unavailable"
                if failure == "missing":
                    raise KeyError("State selector alias authority is missing.")
                if failure is not None:
                    raise ValueError("State selector alias authority is unavailable.")
                if resolved.object != value.object:
                    raise ValueError(
                        "StateSelectorRef selector preparation returned a StateRef outside its ObjectRef scope."
                    )
                memo[key] = resolved
            return memo[key]
        if not isinstance(
            value,
            (DefLink, Definition, dict, FrozenDict, list, FrozenList, tuple, FrozenTuple, set, FrozenSet),
        ):
            return False if not require_scope else value
        key = id(value)
        if key in memo:
            return memo[key]
        if key in active:
            raise ValueError("Cycle while preparing selector state aliases.")
        active.add(key)
        try:
            if not require_scope:
                # Check every branch; an earlier soft alias must not mask an invalid sibling.
                if isinstance(value, DefLink):
                    result = visit(value.target)
                elif isinstance(value, Definition):
                    found = [visit(item) for item in (() if value.args is None else value.args)]
                    found.extend(visit(item) for item in value.kwargs.values())
                    result = any(found)
                else:
                    items = value.values() if isinstance(value, (dict, FrozenDict)) else value
                    result = any([visit(item) for item in items])
            elif isinstance(value, DefLink):
                target = visit(value.target)
                result = target if value.kind is EdgeKind.MATERIALIZE else DefLink.finalized(value.kind, target)
            elif isinstance(value, Definition):
                args = (SKIP_ARGS,) if value.args is None else tuple(visit(item) for item in value.args)
                kwargs = {name: visit(item) for name, item in value.kwargs.items()}
                result = Definition(*args, **kwargs) if value.cls is None else Definition(value.cls, *args, **kwargs)
            elif isinstance(value, (dict, FrozenDict)):
                result = type(value)({name: visit(item) for name, item in value.items()})
            elif isinstance(value, (list, FrozenList, tuple, FrozenTuple)):
                result = type(value)(visit(item) for item in value)
            else:
                result = type(value)(visit(item) for item in value)
                if len(result) != len(value):
                    raise ValueError("Preparing selector state aliases collapsed set members.")
            memo[key] = result
            return result
        finally:
            active.remove(key)

    return visit(source)
