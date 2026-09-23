"""Internal semantic projection of definitions into categorical selectors.

This module prepares the named selector surface shared by future categorical
entry points. It preserves source definition authority and never materializes
or mutates an input graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from inspect import isclass
from typing import Any

from .bound_args import bind_partial_arguments
from .canonical import (
    NodeKind,
    is_value_container_kind,
    node_kind,
    transform_container,
)
from .cdef_identity import cdef_node_key
from .errors import CycleError
from .symbol import ImportRef, SourceSpec, resolve_symbol, symbol_ref


def project_categorical_definition(
        defn: Any,
        recursive: bool = True,
        memo: dict[Any, Any] | None = None,
        *,
        drop: Sequence[str] = (),
        drop_args: bool = False,
        drop_class: bool = False,
):
    """Project a definition graph onto a named categorical selector surface.

    Authored ``Definition`` calls are partially bound only when their supplied
    values must be named. Exact CDefs instead use their persisted parameter
    records directly, without resolving their class. The result is always an
    existing ``Definition`` with skipped positional spelling and preserves the
    source class reference unless ``drop_class`` is requested.

    Args:
        defn: An Object, Definition, or ConcreteDefinition source graph.
        recursive: Whether to apply the controls to nested definition nodes.
        memo: Optional operation-local identity memo for the transformed graph.
        drop: Semantic constructor parameter names to omit.
        drop_args: Whether to omit every argument constraint.
        drop_class: Whether to omit each transformed definition's class.

    Returns:
        A new named ``Definition`` selector graph.

    Raises:
        TypeError: If controls are malformed or authored values cannot be
            safely partially bound.
        ValueError: If a requested name is absent from the original selected
            traversal, a source has no usable signature, or a transformed set
            would lose cardinality.

    Side Effects:
        May resolve an authored symbolic class to inspect its signature. It
        never invokes constructors, applies omitted defaults, resolves CDef
        classes, or mutates source definitions.
    """

    return _project_categorical_definition_with_origins(
        defn,
        recursive=recursive,
        memo=memo,
        drop=drop,
        drop_args=drop_args,
        drop_class=drop_class,
    )[0]


def _project_categorical_definition_with_origins(
        defn: Any,
        recursive: bool = True,
        memo: dict[Any, Any] | None = None,
        *,
        drop: Sequence[str] = (),
        drop_args: bool = False,
        drop_class: bool = False,
) -> tuple[Any, "_ProjectionOrigins"]:
    """Project a definition and retain private source values for query composition.

    The returned records exist only while an immutable query transformation is
    composed. They neither alter selector syntax nor become persisted identity.
    """

    defn, names = _validated_projection_input(
        defn,
        recursive=recursive,
        drop=drop,
        drop_args=drop_args,
        drop_class=drop_class,
    )
    return _project_categorical_definition(
        defn,
        recursive=recursive,
        names=names,
        drop_args=drop_args,
        drop_class=drop_class,
        memo=memo,
    )


def _validated_projection_input(
        defn: Any,
        *,
        recursive: bool,
        drop: Sequence[str],
        drop_args: bool,
        drop_class: bool,
) -> tuple[Any, frozenset[str]]:
    """Validate controls and return the definition selected for projection."""

    from .definition import ConcreteDefinition, Definition
    from .object import Object

    if type(recursive) is not bool:
        raise TypeError("recursive must be a bool.")
    names = _validate_controls(drop, drop_args, drop_class)
    if isinstance(defn, Object):
        defn = defn.definition
    if not isinstance(defn, (Definition, ConcreteDefinition)):
        raise TypeError(
            "Categorical projection requires Object, Definition, or "
            f"ConcreteDefinition; got {type(defn).__name__}."
        )
    if names:
        encountered = _collect_parameter_names(defn, recursive=recursive)
        missing = names - encountered
        if missing:
            raise ValueError(
                "Categorical projection names were not found in the selected "
                f"traversal: {sorted(missing)!r}."
            )
    return defn, names


def _project_categorical_definition(
        defn: Any,
        *,
        recursive: bool,
        names: frozenset[str],
        drop_args: bool,
        drop_class: bool,
        memo: dict[Any, Any] | None,
) -> tuple[Any, _ProjectionOrigins]:
    if memo is None:
        memo = {}
    projection = _Projection(
        recursive=recursive,
        names=names,
        drop_args=drop_args,
        drop_class=drop_class,
        memo=memo,
    )
    result = projection.transform(defn, selected_root=True)
    return result, projection.origins


def _validate_controls(
        drop: Sequence[str], drop_args: bool, drop_class: bool
) -> frozenset[str]:
    if type(drop_args) is not bool:
        raise TypeError("drop_args must be a bool.")
    if type(drop_class) is not bool:
        raise TypeError("drop_class must be a bool.")
    if isinstance(drop, str) or not isinstance(drop, Sequence):
        raise TypeError("drop must be a sequence of strings, not a bare string.")
    drop = tuple(drop)
    if any(not isinstance(name, str) for name in drop):
        raise TypeError("drop entries must be strings.")
    return frozenset(drop)


def _definition_key(value: Any) -> tuple[str, Any] | None:
    from .definition import ConcreteDefinition, Definition

    if isinstance(value, ConcreteDefinition):
        return ("cdef", cdef_node_key(value))
    if isinstance(value, Definition):
        return ("definition", id(value))
    return None


def _semantic_parameters(value: Any):
    """Return source parameter names without applying defaults or projection."""

    from .definition import ConcreteDefinition, Definition

    if isinstance(value, ConcreteDefinition):
        return value.parameters
    if not isinstance(value, Definition):
        raise TypeError(f"Expected a definition, got {type(value).__name__}.")

    # A skipped classless or symbolic Definition is the prepared named form.
    if value.skip_args and (
            value.cls is None or isinstance(value.cls, (ImportRef, SourceSpec))
    ):
        return value.kwargs
    if value.cls is None:
        if value.args:
            raise TypeError(
                "Semantic parameters for a classless definition require "
                "SKIP_ARGS or keyword-only spelling."
            )
        return value.kwargs

    try:
        live_cls = resolve_symbol(value.cls)
    except (ImportError, AttributeError) as error:
        raise TypeError(
            "Categorical projection cannot inspect the authored class authority "
            f"{value.cls!r}; retired mixin references cannot be projected."
        ) from error
    if not isclass(live_cls):
        raise TypeError(
            "Categorical projection class target must resolve to a class; "
            f"got {type(live_cls).__name__}."
        )
    args = () if value.skip_args else tuple(value.args)
    return bind_partial_arguments(live_cls, args, value.kwargs).as_frozen_dict()


def _is_prepared_selector_definition(value: Any) -> bool:
    """Return whether a Definition stores named selector parameters directly."""

    from .definition import Definition

    return (
        isinstance(value, Definition)
        and value.skip_args
        and (value.cls is None or isinstance(value.cls, (ImportRef, SourceSpec)))
    )


def _collect_parameter_names(root: Any, *, recursive: bool) -> set[str]:
    names: set[str] = set()
    seen: set[tuple[str, Any]] = set()
    active: set[tuple[str, Any]] = set()
    active_values: set[int] = set()

    def visit(value: Any, *, selected_root: bool = False) -> None:
        key = _definition_key(value)
        if key is not None:
            if key in active:
                raise CycleError("categorical definition traversal")
            if key in seen:
                return
            active.add(key)
            try:
                parameters = _semantic_parameters(value)
                names.update(parameters)
                if recursive:
                    for child in parameters.values():
                        visit(child)
            finally:
                active.remove(key)
            seen.add(key)
            return

        if not recursive and not selected_root:
            return
        kind = node_kind(value)
        if is_value_container_kind(kind):
            oid = id(value)
            if oid in active_values:
                raise CycleError("categorical value traversal")
            active_values.add(oid)
            try:
                for _, child in _value_children(value):
                    visit(child)
            finally:
                active_values.remove(oid)
            return
        if kind is NodeKind.DEFLINK and value.is_finalized:
            visit(value.target)

    visit(root, selected_root=True)
    return names


def _value_children(value: Any):
    from .canonical import iter_value_children

    return iter_value_children(value)


class _ProjectionOrigins:
    """Private exact source correspondence for transformed unordered members."""

    def __init__(self) -> None:
        self.set_members: dict[int, tuple[tuple[Any, Any], ...]] = {}

    def record_set_members(
            self, result: Any, members: list[tuple[Any, Any]]) -> None:
        """Record output-to-source members for one transformed set occurrence."""

        self.set_members[id(result)] = tuple(members)


class _Projection:
    """One identity-preserving, cycle-aware categorical graph projection."""

    def __init__(
            self,
            *,
            recursive: bool,
            names: frozenset[str],
            drop_args: bool,
            drop_class: bool,
            memo: dict[Any, Any],
    ) -> None:
        self.recursive = recursive
        self.names = names
        self.drop_args = drop_args
        self.drop_class = drop_class
        self.memo = memo
        self.origins = _ProjectionOrigins()
        self.active: set[tuple[str, Any]] = set()
        self.active_values: set[int] = set()

    def transform(self, value: Any, *, selected_root: bool = False) -> Any:
        from .definition import Definition
        from .links import DefLink

        key = _definition_key(value)
        if key is not None:
            if key in self.active:
                raise CycleError("categorical definition projection")
            if key in self.memo:
                return self.memo[key]
            self.active.add(key)
            try:
                source_parameters = {} if self.drop_args else _semantic_parameters(value)
                parameters = dict(source_parameters.items())
                if self.names:
                    for name in self.names:
                        parameters.pop(name, None)
                if self.recursive and parameters:
                    parameters = {
                        name: self.transform(child) for name, child in parameters.items()
                    }
                cls = None if self.drop_class else _selector_class(value.cls)
                result = Definition._from_prepared_parameters(cls, parameters)
                self.memo[key] = result
                return result
            finally:
                self.active.remove(key)

        if not self.recursive and not selected_root:
            return value
        kind = node_kind(value)
        if is_value_container_kind(kind):
            key = ("container", id(value))
            if key in self.memo:
                return self.memo[key]
            if id(value) in self.active_values:
                raise CycleError("categorical value projection")
            self.active_values.add(id(value))
            try:
                members: list[tuple[Any, Any]] = []

                def transform_child(_, child):
                    transformed = self.transform(child)
                    members.append((transformed, child))
                    return transformed

                result = transform_container(value, transform_child, target="same")
                if kind in {NodeKind.SET, NodeKind.FROZEN_SET} and len(result) != len(value):
                    raise ValueError(
                        "Categorical projection would collapse set cardinality."
                    )
                self.memo[key] = result
                if kind in {NodeKind.SET, NodeKind.FROZEN_SET}:
                    self.origins.record_set_members(result, members)
                return result
            finally:
                self.active_values.remove(id(value))
        if kind is NodeKind.DEFLINK and value.is_finalized:
            key = ("deflink", id(value))
            if key in self.memo:
                return self.memo[key]
            if id(value) in self.active_values:
                raise CycleError("categorical link projection")
            self.active_values.add(id(value))
            try:
                result = DefLink.finalized(value.kind, self.transform(value.target))
                self.memo[key] = result
                return result
            finally:
                self.active_values.remove(id(value))
        return value


def _selector_class(value: Any) -> ImportRef | SourceSpec | None:
    """Return stable selector class authority without replacing source specs."""

    if value is None or isinstance(value, (ImportRef, SourceSpec)):
        return value
    return symbol_ref(value)
