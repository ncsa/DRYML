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

    @classmethod
    def _from_root(cls, root: object) -> "Template":
        """Wrap an already-frozen root produced by template rewriting."""

        result = object.__new__(cls)
        object.__setattr__(result, "_root", root)
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

    def sub(
        self,
        *,
        sub_dict: Mapping[str, object] | None = None,
        namespace: object = (),
        traverse_refs: bool = False,
        **bindings: object,
    ) -> "Template":
        """Bind supplied roots once without sampling, construction, or mutation.

        Args:
            sub_dict: Fully qualified root-to-static-value bindings.
            namespace: Optional namespace prepended only to keyword bindings.
            traverse_refs: Whether to enter pre-existing ``Ref(Template)`` data.
            **bindings: Unqualified static root bindings.

        Returns:
            A rewritten template. Parameters introduced by a replacement template
            remain unresolved until a later call.

        Raises:
            TemplateError: If names, values, paths, or traversal controls are
                invalid, including any nested Distribution value.
        """

        if type(traverse_refs) is not bool:
            raise TemplateError("traverse_refs must be a bool")
        occurrences = _snapshot_parameters(self._root, traverse_refs=traverse_refs)
        roots = {parameter.name for parameter in occurrences}
        supplied = _normalize_bindings(sub_dict, namespace, bindings)
        _validate_distribution_free(supplied)
        unknown = sorted(set(supplied) - roots)
        if unknown:
            raise TemplateError(f"unknown template binding roots: {unknown!r}")

        projected: dict[tuple[str, GraphPath], object] = {}
        replacements = {}
        for parameter in occurrences:
            if parameter.name not in supplied:
                continue
            key = parameter.name, parameter.path
            if key not in projected:
                projected[key] = _project_binding(
                    supplied[parameter.name], parameter.path, parameter.name
                )
            replacements[id(parameter)] = projected[key]
        return type(self)._from_root(
            _rewrite_template_value(
                self._root,
                replacements=replacements,
                remap=None,
                traverse_refs=traverse_refs,
            )
        )

    def remap(
        self,
        mapping: Mapping[str, str] | None = None,
        *,
        prefix: object = (),
        strip: object = (),
        traverse_refs: bool = False,
    ) -> "Template":
        """Simultaneously rename, strip, and prefix active binding roots.

        Args:
            mapping: Explicit fully qualified old-to-new root names.
            prefix: Namespace appended to every transformed root.
            strip: Namespace removed from matching roots after explicit renames.
            traverse_refs: Whether to enter pre-existing ``Ref(Template)`` data.

        Returns:
            A new template with unchanged relative graph paths.

        Raises:
            TemplateError: If roots, namespaces, mappings, or traversal controls
                are malformed or refer to absent roots.
        """

        if type(traverse_refs) is not bool:
            raise TemplateError("traverse_refs must be a bool")
        occurrences = _snapshot_parameters(self._root, traverse_refs=traverse_refs)
        roots = {parameter.name for parameter in occurrences}
        renames = _normalize_remap(mapping, roots)
        prefix_parts = _normalize_namespace(prefix, "prefix")
        strip_parts = _normalize_namespace(strip, "strip")
        mapped_roots = {root: renames.get(root, root) for root in roots}
        if strip_parts and not any(_has_prefix(_root_parts(root), strip_parts) for root in mapped_roots.values()):
            raise TemplateError("strip namespace does not match any active template root")

        result_names = {
            root: _remap_root(name, prefix_parts, strip_parts)
            for root, name in mapped_roots.items()
        }
        return type(self)._from_root(
            _rewrite_template_value(
                self._root,
                replacements=None,
                remap=result_names,
                traverse_refs=traverse_refs,
            )
        )

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


def _validate_root(name: object, label: str) -> str:
    if not isinstance(name, str) or "." in name:
        raise TemplateError(f"{label} must be a fully qualified template root")
    parts = name.split("/")
    if not name or any(not _COMPONENT.fullmatch(part) for part in parts):
        raise TemplateError(f"{label} contains an invalid qualified-name component")
    return name


def _root_parts(name: str) -> tuple[str, ...]:
    return tuple(name.split("/"))


def _normalize_namespace(value: object, label: str) -> tuple[str, ...]:
    if value == ():
        return ()
    if isinstance(value, str):
        parts = tuple(value.split("/"))
    elif isinstance(value, tuple):
        parts = value
    else:
        raise TemplateError(f"template {label} must be a string or tuple of components")
    if any(not isinstance(part, str) or not _COMPONENT.fullmatch(part) for part in parts):
        raise TemplateError(f"template {label} contains an invalid component")
    return parts


def _normalize_bindings(
    sub_dict: Mapping[str, object] | None,
    namespace: object,
    bindings: Mapping[str, object],
) -> dict[str, object]:
    namespace_parts = _normalize_namespace(namespace, "namespace")
    if sub_dict is not None and not isinstance(sub_dict, Mapping):
        raise TemplateError("sub_dict must be a mapping")
    result: dict[str, object] = {}
    for name, value in (() if sub_dict is None else sub_dict.items()):
        normalized = _validate_root(name, "sub_dict key")
        if normalized in result:
            raise TemplateError(f"duplicate template binding for {normalized!r}")
        result[normalized] = value
    for name, value in bindings.items():
        if not _COMPONENT.fullmatch(name):
            raise TemplateError(f"keyword binding {name!r} is not a valid root component")
        normalized = "/".join((*namespace_parts, name))
        if normalized in result:
            raise TemplateError(f"duplicate template binding for {normalized!r}")
        result[normalized] = value
    return result


def _normalize_remap(mapping: Mapping[str, str] | None, roots: set[str]) -> dict[str, str]:
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise TemplateError("template remap mapping must be a mapping")
    result: dict[str, str] = {}
    for source, target in mapping.items():
        source = _validate_root(source, "remap source")
        target = _validate_root(target, "remap target")
        if source not in roots:
            raise TemplateError(f"remap source {source!r} is not an active template root")
        result[source] = target
    return result


def _has_prefix(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return parts[:len(prefix)] == prefix


def _remap_root(name: str, prefix: tuple[str, ...], strip: tuple[str, ...]) -> str:
    parts = _root_parts(name)
    if strip and _has_prefix(parts, strip):
        parts = parts[len(strip):]
    parts = (*prefix, *parts)
    if not parts:
        raise TemplateError("template remap produced an empty root")
    return _validate_root("/".join(parts), "remapped root")


def _snapshot_parameters(value: object, *, traverse_refs: bool) -> tuple[Par, ...]:
    """Collect the pre-operation parameters allowed by one traversal boundary."""

    from .links import DefLink

    found: list[Par] = []
    seen: set[int] = set()

    def visit(current: object) -> None:
        if isinstance(current, Par):
            found.append(current)
            return
        if isinstance(current, Template):
            return
        if isinstance(current, DefLink):
            if (
                traverse_refs
                and current.kind.name == "REF"
                and isinstance(current.target, Template)
            ):
                visit(current.target.root)
            return
        children = _template_children(current)
        if not children:
            return
        marker = id(current)
        if marker in seen:
            return
        seen.add(marker)
        for child in children:
            visit(child)

    visit(value)
    return tuple(found)


def _validate_distribution_free(value: object) -> None:
    """Reject providers structurally before any projection or rewrite work starts."""

    from .definition import ConcreteDefinition, Definition
    from .domains import Distribution
    from .factory import FactorySpec
    from .links import DefLink

    seen: set[int] = set()

    def visit(current: object) -> None:
        if isinstance(current, Distribution):
            raise TemplateError("Template.sub does not accept Distribution values")
        if isinstance(current, Template):
            visit(current.root)
            return
        if isinstance(current, DefLink):
            visit(current.target)
            return
        if isinstance(current, FactorySpec):
            children = (*current.args, *current.kwargs.values())
        elif isinstance(current, Definition):
            children = (() if current.args is None else tuple(current.args)) + tuple(current.kwargs.values())
        elif isinstance(current, ConcreteDefinition):
            children = tuple(current.parameters.values())
        elif isinstance(current, Mapping):
            children = tuple(current.values())
        elif isinstance(current, (list, tuple, set, frozenset, FrozenList, FrozenTuple, FrozenSet)):
            children = tuple(current)
        else:
            return
        marker = id(current)
        if marker in seen:
            return
        seen.add(marker)
        for child in children:
            visit(child)

    for binding in value.values():
        visit(binding)


def _project_binding(value: object, path: GraphPath, root: str) -> object:
    """Select a supplied root path before lowering any live Object leaf."""

    from .definition import ConcreteDefinition, Definition
    from .object import Object
    from .reference_values import ObjectRef, StateRef
    from .utils.graph.value import get_subtree

    try:
        if path:
            if isinstance(value, Object):
                value = value.graph_at(path)
            elif isinstance(value, (ObjectRef, StateRef)):
                value = value.at(path)
            elif isinstance(value, ConcreteDefinition):
                value = value.graph_path(path)
            elif isinstance(value, Definition) and any(isinstance(segment, Parameter) for segment in path):
                if value.cls is None or not isinstance(value.cls, type):
                    raise TemplateError("semantic path requires an available soft Definition class")
                first, *rest = tuple(path)
                if not isinstance(first, Parameter):
                    raise TemplateError("semantic Definition path must begin with a Parameter")
                value = value.parameters[first.name]
                if rest:
                    value = get_subtree(value, GraphPath(tuple(rest)))
            else:
                value = get_subtree(value, path)
    except TemplateError:
        raise
    except Exception as error:
        raise TemplateError(f"template binding path is invalid for root {root!r}", path=path, root=root) from error
    return _freeze_binding_value(value)


def _freeze_binding_value(value: object) -> object:
    """Detach containers and lower live Object leaves without traversing their graphs."""

    from .canonical import freeze_def_value
    from .object import Object

    memo: dict[int, object] = {}

    def lower(current: object) -> object:
        if isinstance(current, Object):
            return current.object_ref
        if isinstance(current, Template):
            return current
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            items = [(key, lower(item)) for key, item in current.items()]
            if isinstance(current, FrozenDict):
                result = FrozenDict(items)
            else:
                result = dict(items)
            memo[marker] = result
            return result
        if isinstance(current, (list, FrozenList)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            result = FrozenList(lower(item) for item in current)
            memo[marker] = result
            return result
        if isinstance(current, (tuple, FrozenTuple)):
            return FrozenTuple(lower(item) for item in current)
        if isinstance(current, (set, frozenset, FrozenSet)):
            return FrozenSet(lower(item) for item in current)
        return current

    try:
        lowered = lower(value)
        return lowered if isinstance(lowered, Template) else freeze_def_value(lowered)
    except Exception as error:
        raise TemplateError("template binding value is unsupported") from error


def _rewrite_template_value(
    value: object,
    *,
    replacements: Mapping[int, object] | None,
    remap: Mapping[str, str] | None,
    traverse_refs: bool,
) -> object:
    """Perform one memoized copy-on-write traversal over pre-selected values."""

    from .definition import Definition
    from .factory import FactorySpec
    from .links import DefLink

    memo: dict[int, object] = {}

    def rewrite(current: object) -> object:
        if isinstance(current, Par):
            if replacements is not None and id(current) in replacements:
                replacement = replacements[id(current)]
                return replacement.root if isinstance(replacement, Template) else replacement
            if remap is not None and current.name in remap:
                return Par(remap[current.name], path=current.path)
            return current
        if isinstance(current, Template):
            return current
        if isinstance(current, DefLink):
            if not (
                traverse_refs
                and current.kind.name == "REF"
                and isinstance(current.target, Template)
            ):
                return current
            marker = id(current)
            if marker in memo:
                return memo[marker]
            target = Template._from_root(rewrite(current.target.root))
            result = DefLink.assertion(current.kind, target)
            memo[marker] = result
            return result
        if isinstance(current, _BinaryExpr):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            left, right = rewrite(current.left), rewrite(current.right)
            result = current if left is current.left and right is current.right else _BinaryExpr(current.operation, left, right)
            memo[marker] = result
            return result
        if isinstance(current, _RepeatExpr):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            group = tuple(rewrite(item) for item in current.group)
            count = rewrite(current.count)
            unchanged = all(new is old for new, old in zip(group, current.group)) and count is current.count
            result = current if unchanged else _RepeatExpr(
                list(group) if isinstance(current.group, list) else group, count, current.shared
            )
            memo[marker] = result
            return result
        if isinstance(current, FactorySpec):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            args = tuple(rewrite(item) for item in current.args)
            kwargs = FrozenDict((key, rewrite(item)) for key, item in current.kwargs.items())
            unchanged = (
                all(new is old for new, old in zip(args, current.args))
                and all(kwargs[key] is value for key, value in current.kwargs.items())
            )
            result = current if unchanged else FactorySpec._from_template_parts(current.target, args, kwargs)
            memo[marker] = result
            return result
        if isinstance(current, Definition):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            args = None if current.args is None else FrozenTuple(rewrite(item) for item in current.args)
            kwargs = FrozenDict((key, rewrite(item)) for key, item in current.kwargs.items())
            unchanged = (
                (args is None and current.args is None)
                or (args is not None and all(new is old for new, old in zip(args, current.args)))
            ) and all(kwargs[key] is value for key, value in current.kwargs.items())
            result = current if unchanged else Definition._from_template_parts(current.cls, args, kwargs)
            memo[marker] = result
            return result
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            items = [(key, rewrite(item)) for key, item in current.items()]
            result = current if all(new is current[key] for key, new in items) else (
                FrozenDict(items) if isinstance(current, FrozenDict) else dict(items)
            )
            memo[marker] = result
            return result
        if isinstance(current, (list, FrozenList)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            items = tuple(rewrite(item) for item in current)
            result = current if all(new is old for new, old in zip(items, current)) else FrozenList(items)
            memo[marker] = result
            return result
        if isinstance(current, (tuple, FrozenTuple)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            items = tuple(rewrite(item) for item in current)
            result = current if all(new is old for new, old in zip(items, current)) else FrozenTuple(items)
            memo[marker] = result
            return result
        if isinstance(current, (set, frozenset, FrozenSet)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            items = tuple(rewrite(item) for item in current)
            result = current if all(new is old for new, old in zip(items, current)) else FrozenSet(items)
            memo[marker] = result
            return result
        return current

    return rewrite(value)


__all__ = ["Expr", "Par", "Shared", "Template", "repeat"]
