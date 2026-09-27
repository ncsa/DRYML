"""Inert definition-template authoring and symbolic expression syntax."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any

from .errors import TemplateError, UnresolvedTemplateError
from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from .utils.graph.path import GraphPath, Parameter, normalize_path


_COMPONENT = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")


def _validate_number(value: object) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _operand(value: object) -> object:
    if isinstance(value, Expr) or _validate_number(value):
        return value
    return NotImplemented


def _stable_operand(value: object) -> object:
    return value._stable_value() if isinstance(value, Expr) else value


class Expr:
    """Base class for immutable unresolved template arithmetic and repetition.

    Symbolic expressions record supported operations but never implement index
    conversion or call user constructors. Unsupported operands return
    ``NotImplemented`` so Python can apply ordinary reflected dispatch.
    """

    def __mul__(self, other: object, /) -> "Expr":
        """Return an inert multiplication expression when ``other`` is supported.

        Args:
            other: Another expression or finite exact built-in number.

        Returns:
            A binary expression, or ``NotImplemented`` for Python dispatch.
        """
        operand = _operand(other)
        return NotImplemented if operand is NotImplemented else _BinaryExpr("mul", self, operand)

    def __rmul__(self, other: object, /) -> "Expr":
        """Return reflected multiplication or list/tuple repetition syntax.

        Args:
            other: A supported scalar, expression, list, or tuple.

        Returns:
            An inert arithmetic or repetition expression, or ``NotImplemented``.
        """
        if isinstance(other, (list, tuple)):
            return repeat(other, self)
        operand = _operand(other)
        return NotImplemented if operand is NotImplemented else _BinaryExpr("mul", operand, self)

    def __truediv__(self, other: object, /) -> "Expr":
        """Return an inert true-division expression for a supported operand.

        Args:
            other: Another expression or finite exact built-in number.

        Returns:
            A binary expression, or ``NotImplemented`` for Python dispatch.
        """
        operand = _operand(other)
        return NotImplemented if operand is NotImplemented else _BinaryExpr("truediv", self, operand)

    def __rtruediv__(self, other: object, /) -> "Expr":
        """Return an inert reflected true-division expression.

        Args:
            other: Another expression or finite exact built-in number.

        Returns:
            A binary expression, or ``NotImplemented`` for Python dispatch.
        """
        operand = _operand(other)
        return NotImplemented if operand is NotImplemented else _BinaryExpr("truediv", operand, self)

    def __floordiv__(self, other: object, /) -> "Expr":
        """Return an inert floor-division expression for a supported operand.

        Args:
            other: Another expression or finite exact built-in number.

        Returns:
            A binary expression, or ``NotImplemented`` for Python dispatch.
        """
        operand = _operand(other)
        return NotImplemented if operand is NotImplemented else _BinaryExpr("floordiv", self, operand)

    def __rfloordiv__(self, other: object, /) -> "Expr":
        """Return an inert reflected floor-division expression.

        Args:
            other: Another expression or finite exact built-in number.

        Returns:
            A binary expression, or ``NotImplemented`` for Python dispatch.
        """
        operand = _operand(other)
        return NotImplemented if operand is NotImplemented else _BinaryExpr("floordiv", operand, self)

    def _stable_value(self) -> object:
        raise NotImplementedError

    def __stable_leaf_bytes__(self) -> bytes:
        from .utils.stable_hash import stable_hash_function

        return b"dryml-template-expr:" + stable_hash_function(self._stable_value()).encode("ascii")


@dataclass(frozen=True, slots=True, init=False)
class Par(Expr):
    """Identify one named template binding root and optional semantic path.

    Args:
        name: ASCII qualified root, optionally followed by dot-separated
            semantic parameter components when ``path`` is omitted.
        path: Explicit root-relative graph path. It cannot accompany dotted
            ``name`` syntax.

    Raises:
        TemplateError: If the root, path spelling, or components are malformed.
    """

    name: str
    path: GraphPath

    def __init__(self, name: str, /, *, path: object | None = None) -> None:
        if not isinstance(name, str):
            raise TemplateError("template parameter name must be a string")
        root, dot, suffix = name.partition(".")
        if path is not None and dot:
            raise TemplateError("dotted parameter names cannot be combined with an explicit path")
        components = root.split("/")
        if not root or any(not _COMPONENT.fullmatch(component) for component in components):
            raise TemplateError("template parameter root contains an invalid qualified-name component")
        if path is None:
            if suffix:
                path_components = suffix.split(".")
                if any(not _COMPONENT.fullmatch(component) for component in path_components):
                    raise TemplateError("template parameter dotted path contains an invalid component")
                normalized_path = GraphPath(tuple(Parameter(component) for component in path_components))
            else:
                normalized_path = GraphPath()
        else:
            try:
                normalized_path = normalize_path(path)
            except Exception as error:
                raise TemplateError("template parameter path is invalid") from error
        object.__setattr__(self, "name", root)
        object.__setattr__(self, "path", normalized_path)

    def _stable_value(self) -> object:
        return ("par", self.name, tuple((type(segment).__name__, str(segment)) for segment in self.path))


@dataclass(frozen=True, slots=True)
class _BinaryExpr(Expr):
    operation: str
    left: object
    right: object

    def _stable_value(self) -> object:
        return ("binary", self.operation, _stable_operand(self.left), _stable_operand(self.right))


@dataclass(frozen=True, slots=True)
class Shared:
    """Request graph-sharing semantics for a repeated template group.

    Args:
        count: Exact nonnegative integer or unresolved template expression.

    Raises:
        TemplateError: If ``count`` is not a permitted repetition count.
    """

    count: int | Expr

    def __post_init__(self) -> None:
        _validate_count(self.count)

    def __rmul__(self, group: object, /) -> Expr:
        """Return a shared repetition expression for a list or tuple group.

        Args:
            group: The ordered recipe group supplied through reflected
                multiplication.

        Returns:
            A shared repetition expression, or ``NotImplemented`` for other
            left operands.
        """
        if not isinstance(group, (list, tuple)):
            return NotImplemented
        return _RepeatExpr(group, self.count, shared=True)


def _validate_count(count: object) -> None:
    if isinstance(count, Expr):
        return
    if type(count) is not int or count < 0:
        raise TemplateError("template repetition count must be a nonnegative exact int or Expr")


@dataclass(frozen=True, slots=True)
class _RepeatExpr(Expr):
    group: list | tuple
    count: int | Expr
    shared: bool = False

    def _stable_value(self) -> object:
        return ("repeat", tuple(_stable_operand(value) for value in self.group), _stable_operand(self.count), self.shared)


def repeat(group: list | tuple, count: int | Expr | Shared, /) -> Expr:
    """Capture a list or tuple repetition that remains inert until evaluation.

    Args:
        group: Ordered list or tuple recipe group.
        count: Exact nonnegative count, symbolic expression, or ``Shared``
            sharing request.

    Returns:
        An immutable repetition expression.

    Raises:
        TemplateError: If the group or count is unsupported.
    """
    if not isinstance(group, (list, tuple)):
        raise TemplateError("template repetition requires a list or tuple group")
    if isinstance(count, Shared):
        return _RepeatExpr(group, count.count, shared=True)
    _validate_count(count)
    return _RepeatExpr(group, count)


def _template_children(value: object) -> tuple[object, ...]:
    from .definition import ConcreteDefinition, Definition
    from .factory import FactorySpec

    if isinstance(value, FactorySpec):
        return (*value.args, *value.kwargs.values())
    if isinstance(value, Definition):
        return (() if value.args is None else tuple(value.args)) + tuple(value.kwargs.values())
    if isinstance(value, ConcreteDefinition):
        return tuple(value.parameters.values())
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, (list, tuple, set, frozenset, FrozenList, FrozenTuple, FrozenSet, FrozenDict)):
        return tuple(value.values()) if isinstance(value, FrozenDict) else tuple(value)
    if isinstance(value, _BinaryExpr):
        return value.left, value.right
    if isinstance(value, _RepeatExpr):
        return (*value.group, value.count)
    return ()


def _active_parameters(value: object) -> tuple[Par, ...]:
    seen_values: set[int] = set()
    seen_names: set[str] = set()
    found: list[Par] = []

    def visit(current: object) -> None:
        if isinstance(current, Par):
            if current.name not in seen_names:
                seen_names.add(current.name)
                found.append(current)
            return
        if isinstance(current, Expr):
            marker = id(current)
            if marker in seen_values:
                return
            seen_values.add(marker)
        for child in _template_children(current):
            visit(child)

    visit(value)
    return tuple(found)


@dataclass(frozen=True, slots=True, init=False)
class Template:
    """Capture an immutable definition recipe without resolving its target.

    Args:
        target: Class or symbolic class authority for a direct Definition root.
        *args: Inert positional recipe values.
        **kwargs: Inert keyword recipe values.

    Raises:
        TypeError: If ``target`` is not a class, ImportRef, or SourceSpec.

    Direct construction never calls, imports, or otherwise resolves ``target``.
    Use :meth:`from_value` for an existing Definition or a generic value root.
    """

    _root: object

    def __init__(self, target: object, /, *args: object, **kwargs: object) -> None:
        from .symbol import ImportRef, SourceSpec

        if not isinstance(target, (type, ImportRef, SourceSpec)):
            raise TypeError("Template target must be a class, ImportRef, or SourceSpec; use Template.from_value for existing values.")
        from .definition import Definition

        object.__setattr__(self, "_root", Definition(target, *args, **kwargs))

    @classmethod
    def from_value(cls, value: object, /) -> "Template":
        """Convert an existing supported frozen value into an inert template.

        Args:
            value: Existing Definition, FactorySpec, expression-bearing value,
                or supported nested container root.

        Returns:
            A template retaining an immutable root value.

        Raises:
            TypeError: If ``value`` cannot be represented as a definition value.
        """
        if isinstance(value, cls):
            return value
        from .canonical import freeze_def_value

        result = object.__new__(cls)
        object.__setattr__(result, "_root", freeze_def_value(value))
        return result

    @property
    def root(self) -> object:
        """Return the immutable inert recipe root without resolving it."""
        return self._root

    @property
    def names(self) -> tuple[str, ...]:
        """Return active binding roots once each in lexical traversal order."""
        return tuple(parameter.name for parameter in _active_parameters(self._root))

    @property
    def is_resolved(self) -> bool:
        """Return whether the active recipe contains no template expressions."""
        return not _active_parameters(self._root) and not _contains_expression(self._root)

    def resolve(self) -> object:
        """Return the root when no active expression remains.

        Raises:
            UnresolvedTemplateError: If a parameter, arithmetic, or repetition
                expression still requires a later template operation.
        """
        if not self.is_resolved:
            raise UnresolvedTemplateError("template contains unresolved expressions")
        return self._root

    def to_definition(self):
        """Return a resolved soft Definition root without concretizing it.

        Returns:
            The retained Definition recipe.

        Raises:
            UnresolvedTemplateError: If active expressions remain.
            TemplateError: If the template root is not a soft Definition.
        """
        from .definition import Definition

        root = self.resolve()
        if not isinstance(root, Definition):
            raise TemplateError("Template.to_definition requires a Definition root")
        return root

    def stable_hash(self) -> str:
        """Return a deterministic hash for this frozen recipe root."""
        from .utils.stable_hash import stable_hash_function

        return stable_hash_function(self._root)


def _contains_expression(value: object) -> bool:
    seen: set[int] = set()

    def visit(current: object) -> bool:
        if isinstance(current, Expr):
            return True
        children = _template_children(current)
        if not children:
            return False
        marker = id(current)
        if marker in seen:
            return False
        seen.add(marker)
        return any(visit(child) for child in children)

    return visit(value)


__all__ = ["Expr", "Par", "Shared", "Template", "repeat"]
