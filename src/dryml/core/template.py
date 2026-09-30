"""Inert definition-template authoring and symbolic expression syntax."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re

from .errors import ParameterizationError, ParameterizationLimitError, UnresolvedDefinitionError
from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from .utils.graph.path import GraphPath, Parameter, normalize_path


_COMPONENT = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")
_MAX_EXPRESSION_DEPTH = 128
_MAX_EXPRESSION_VALUES = 65_536
_MAX_INTEGER_BITS = 1_024
_MAX_REPEAT_COUNT = 1_024
_MAX_NAMESPACE_COMPONENTS = 16
_MAX_COMPONENT_LENGTH = 64
_MAX_QUALIFIED_ROOT_LENGTH = 256


def _validate_number(value: object) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _validate_integer_limit(value: object) -> None:
    if type(value) is int and value.bit_length() > _MAX_INTEGER_BITS:
        raise ParameterizationLimitError("template integer bit-length limit exceeded")


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
        ParameterizationError: If the root, path spelling, or components are malformed.
    """

    name: str
    path: GraphPath

    def __init__(self, name: str, /, *, path: object | None = None) -> None:
        if not isinstance(name, str):
            raise ParameterizationError("template parameter name must be a string")
        root, dot, suffix = name.partition(".")
        if path is not None and dot:
            raise ParameterizationError("dotted parameter names cannot be combined with an explicit path")
        _validate_root(root, "template parameter root")
        if path is None:
            if suffix:
                path_components = suffix.split(".")
                if any(
                    not isinstance(component, str)
                    or len(component) > _MAX_COMPONENT_LENGTH
                    or not _COMPONENT.fullmatch(component)
                    for component in path_components
                ):
                    raise ParameterizationError("template parameter dotted path contains an invalid component")
                normalized_path = GraphPath(tuple(Parameter(component) for component in path_components))
            else:
                normalized_path = GraphPath()
        else:
            try:
                normalized_path = normalize_path(path)
            except Exception as error:
                raise ParameterizationError("template parameter path is invalid") from error
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
        ParameterizationError: If ``count`` is not a permitted repetition count.
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
        raise ParameterizationError("template repetition count must be a nonnegative exact int or Expr")


@dataclass(frozen=True, slots=True)
class _RepeatExpr(Expr):
    group: list | tuple
    count: int | Expr
    shared: bool = False

    def __post_init__(self) -> None:
        """Freeze the ordered group while retaining its list or tuple family."""

        _validate_count(self.count)
        from .canonical import freeze_def_value

        object.__setattr__(self, "group", freeze_def_value(self.group))

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
        ParameterizationError: If the group or count is unsupported.
    """
    if not isinstance(group, (list, tuple)):
        raise ParameterizationError("template repetition requires a list or tuple group")
    if isinstance(count, Shared):
        return _RepeatExpr(group, count.count, shared=True)
    _validate_count(count)
    return _RepeatExpr(group, count)


def _template_children(value: object) -> tuple[object, ...]:
    from .cdef_graph import EdgeKind
    from .definition import ConcreteDefinition, Definition
    from .factory import FactorySpec
    from .links import DefLink

    if isinstance(value, TemplateBundle):
        return tuple(value.recipes.values())
    if isinstance(value, DefLink):
        return (value.target,) if value.kind is EdgeKind.MATERIALIZE else ()
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


@dataclass(frozen=True, slots=True, init=False, eq=False)
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

        root = Definition(target, *args, **kwargs)
        _validate_template_set_members(root)
        object.__setattr__(self, "_root", root)

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
        root = freeze_def_value(value)
        _validate_template_set_members(root)
        object.__setattr__(result, "_root", root)
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
        return not _contains_expression(self._root)

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
            A rewritten template with fully bound arithmetic and repetition
            expressions evaluated. Parameters introduced by a replacement
            template remain unresolved until a later call.

        Raises:
            ParameterizationError: If names, values, paths, traversal controls, or a
                bound expression are invalid, including any nested Distribution
                value. ParameterizationLimitError: If expression depth or expansion
                exceeds a hard limit.
        """

        if type(traverse_refs) is not bool:
            raise ParameterizationError("traverse_refs must be a bool")
        occurrences = _snapshot_parameters(self._root, traverse_refs=traverse_refs)
        roots = {parameter.name for parameter in occurrences}
        supplied = _normalize_bindings(sub_dict, namespace, bindings)
        _validate_distribution_free(supplied)
        unknown = sorted(set(supplied) - roots)
        if unknown:
            raise ParameterizationError(f"unknown template binding roots: {unknown!r}")

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
            _evaluate_template_value(
                _rewrite_template_value(
                    self._root,
                    replacements=replacements,
                    remap=None,
                    traverse_refs=traverse_refs,
                ),
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
            ParameterizationError: If roots, namespaces, mappings, or traversal controls
                are malformed or refer to absent roots.
        """

        if type(traverse_refs) is not bool:
            raise ParameterizationError("traverse_refs must be a bool")
        occurrences = _snapshot_parameters(self._root, traverse_refs=traverse_refs)
        roots = {parameter.name for parameter in occurrences}
        renames = _normalize_remap(mapping, roots)
        prefix_parts = _normalize_namespace(prefix, "prefix")
        strip_parts = _normalize_namespace(strip, "strip")
        mapped_roots = {root: renames.get(root, root) for root in roots}
        if strip_parts and not any(_has_prefix(_root_parts(root), strip_parts) for root in mapped_roots.values()):
            raise ParameterizationError("strip namespace does not match any active template root")

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
        """Evaluate and return the root when no active expression remains.

        Raises:
            UnresolvedDefinitionError: If a parameter, arithmetic, or repetition
                expression still requires a later template operation.
            ParameterizationError: If a fully bound expression is invalid.
            ParameterizationLimitError: If expression depth or expansion exceeds a hard
                limit.
        """
        root = _evaluate_template_value(self._root)
        if _contains_expression(root):
            raise UnresolvedDefinitionError("template contains unresolved expressions")
        return root

    def as_selector(self, *, strict: bool = False):
        """Project known structure into a deliberately loose ordinary Selector.

        Args:
            strict: Ordinary Selector strictness for retained known structure.

        Returns:
            A symbolic-class-exact Selector that omits unknown relationships.

        Raises:
            ParameterizationError: If this template does not have a soft Definition root.

        This projection never samples domains, resolves targets, or asserts exact
        parameter linkage, arithmetic, repetition topology, or factory builds.
        """

        from .template_selector import _loose_selector

        selector = _loose_selector(self)
        if strict:
            from .selector import Selector

            return Selector(selector.root, strict=True, cls_policy="exact")
        return selector

    def to_definition(self):
        """Return a resolved soft Definition root without concretizing it.

        Returns:
            The retained Definition recipe.

        Raises:
            UnresolvedDefinitionError: If active expressions remain.
            ParameterizationError: If the template root is not a soft Definition.
        """
        from .definition import Definition

        root = self.resolve()
        if not isinstance(root, Definition):
            raise ParameterizationError("Template.to_definition requires a Definition root")
        return root

    def stable_hash(self) -> str:
        """Return a topology-sensitive deterministic digest for this recipe.

        Raises:
            ParameterizationError: If the recipe contains a nonportable value.
        """
        from .template_codec import stable_hash

        return stable_hash(self)

    def __stable_leaf_bytes__(self) -> bytes:
        """Return a closed-codec leaf projection for enclosing CDef identity."""

        return b"dryml-template:" + self.stable_hash().encode("ascii")

    def __eq__(self, other: object) -> bool:
        """Compare portable recipe meaning, including graph-local aliases.

        Args:
            other: Value to compare with this Template.

        Returns:
            ``True`` only for the same closed-codec recipe projection.

        Raises:
            ParameterizationError: If either Template is nonportable.
        """
        return isinstance(other, Template) and self.to_data() == other.to_data()

    def __hash__(self) -> int:
        """Return the portable topology-sensitive hash used by frozen values.

        Raises:
            ParameterizationError: If this Template is nonportable.
        """
        return int(self.stable_hash(), 16)

    def to_data(self) -> dict[str, object]:
        """Return the closed portable template payload without target resolution.

        Returns:
            Canonical ``dryml-template`` v1 data.

        Raises:
            ParameterizationError: If the recipe is nonportable or exceeds codec limits.
        """
        from .template_codec import template_to_data

        return template_to_data(self)

    @classmethod
    def from_data(cls, data: Mapping[str, object], /) -> "Template":
        """Decode a closed portable recipe without importing its target.

        Args:
            data: A canonical ``dryml-template`` v1 mapping.

        Returns:
            The decoded immutable Template.

        Raises:
            ParameterizationError: If data is malformed, unsupported, or noncanonical.
        """
        from .template_codec import template_from_data

        return template_from_data(data)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class TemplateBundle:
    """Retain named inert Template recipes as one ordered quotation value.

    Args:
        recipes: One Template, an ordered list or tuple of Templates, or a
            string-keyed mapping of Templates. Positional recipes are named
            ``artifact_N`` in order; mapping insertion order is retained.

    The bundle validates and freezes names and recipes only. It never resolves
    parameters, invokes factories, constructs artifacts, or materializes inputs.
    A bundle is admitted to a concrete definition only through
    ``Ref[TemplateBundle]``.
    """

    _recipes: FrozenDict

    def __init__(self, recipes: "Template | list[Template] | tuple[Template, ...] | Mapping[str, Template]", /) -> None:
        if isinstance(recipes, Template):
            items = (("artifact_0", recipes),)
        elif isinstance(recipes, (list, tuple)):
            items = tuple((f"artifact_{index}", recipe) for index, recipe in enumerate(recipes))
        elif isinstance(recipes, Mapping):
            items = tuple(recipes.items())
        else:
            raise TypeError("TemplateBundle requires a Template, ordered Template sequence, or string-keyed Template mapping.")
        object.__setattr__(self, "_recipes", self._validated_recipes(items))

    @staticmethod
    def _validated_recipes(items: object) -> FrozenDict:
        try:
            pairs = tuple(items)
        except TypeError as error:
            raise TypeError("TemplateBundle recipes must be ordered name/Template pairs.") from error
        if len(pairs) > 4_096:
            raise ParameterizationLimitError("template bundle entry limit exceeded")
        result = []
        names = set()
        for item in pairs:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("TemplateBundle recipes must be ordered name/Template pairs.")
            name, recipe = item
            if not isinstance(name, str) or not name:
                raise TypeError("TemplateBundle recipe names must be nonempty strings.")
            if name in names:
                raise ParameterizationError("TemplateBundle recipe names must be unique.")
            if not isinstance(recipe, Template):
                raise TypeError("TemplateBundle recipes must be Template instances.")
            names.add(name)
            result.append((name, recipe))
        return FrozenDict(result)

    @classmethod
    def _from_recipes(cls, recipes: Mapping[str, Template]) -> "TemplateBundle":
        """Wrap codec- or rewrite-owned frozen recipes after invariant checks."""

        result = object.__new__(cls)
        object.__setattr__(result, "_recipes", cls._validated_recipes(tuple(recipes.items())))
        return result

    @property
    def recipes(self) -> FrozenDict:
        """Return the frozen ordered mapping from artifact name to Template.

        Returns:
            The immutable name-to-recipe mapping in declared order.

        Side Effects:
            None. Access does not resolve or copy a recipe.
        """

        return self._recipes

    @property
    def names(self) -> tuple[str, ...]:
        """Return artifact names in declared processing order.

        Returns:
            The immutable ordered tuple of bundle names.

        Side Effects:
            None.
        """

        return tuple(self._recipes)

    def to_data(self) -> dict[str, object]:
        """Return one aggregate closed v1 quotation payload for all recipes.

        Returns:
            Canonical ``dryml-template`` v1 aggregate data.

        Raises:
            ParameterizationError: If a recipe is nonportable or the aggregate exceeds
                a codec limit.

        Side Effects:
            None. Encoding never resolves a recipe target.
        """

        from .template_codec import template_bundle_to_data

        return template_bundle_to_data(self)

    @classmethod
    def from_data(cls, data: Mapping[str, object], /) -> "TemplateBundle":
        """Decode one canonical bundle payload without resolving any recipe.

        Args:
            data: Canonical aggregate ``dryml-template`` v1 data.

        Returns:
            The decoded immutable bundle.

        Raises:
            ParameterizationError: If data is malformed, unsupported, noncanonical, or
                exceeds an aggregate codec limit.

        Side Effects:
            None. Decoding does not import or invoke recipe targets.
        """

        from .template_codec import template_bundle_from_data

        return template_bundle_from_data(data)

    def __stable_leaf_bytes__(self) -> bytes:
        """Return the closed aggregate quotation identity for enclosing CDefs."""

        from .template_codec import _canonical_bytes

        return b"dryml-template-bundle:" + _canonical_bytes(self.to_data())

    def __eq__(self, other: object) -> bool:
        """Compare canonical aggregate quotation meaning."""

        return isinstance(other, TemplateBundle) and self.to_data() == other.to_data()

    def __hash__(self) -> int:
        """Return the canonical aggregate quotation hash."""

        return hash(self.__stable_leaf_bytes__())


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
        raise ParameterizationError(f"{label} must be a fully qualified template root")
    parts = name.split("/")
    if (
        not name
        or len(parts) > _MAX_NAMESPACE_COMPONENTS
        or len(name) > _MAX_QUALIFIED_ROOT_LENGTH
        or any(
            len(part) > _MAX_COMPONENT_LENGTH or not _COMPONENT.fullmatch(part)
            for part in parts
        )
    ):
        raise ParameterizationError(f"{label} contains an invalid qualified-name component")
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
        raise ParameterizationError(f"template {label} must be a string or tuple of components")
    if (
        len(parts) > _MAX_NAMESPACE_COMPONENTS
        or sum(len(part) for part in parts) + max(len(parts) - 1, 0) > _MAX_QUALIFIED_ROOT_LENGTH
        or any(
            not isinstance(part, str)
            or len(part) > _MAX_COMPONENT_LENGTH
            or not _COMPONENT.fullmatch(part)
            for part in parts
        )
    ):
        raise ParameterizationError(f"template {label} contains an invalid component")
    return parts


def _validate_template_set_members(value: object) -> None:
    """Reject symbolic or structural values whose unordered positions are unstable.

    Template expressions carry addressable construction structure.  Allowing them
    in a set would make canonical ordering part of that structure, so only
    literal scalar/symbol members are admitted at public template construction.
    """

    from .definition import ConcreteDefinition, Definition
    from .factory import FactorySpec
    from .links import DefLink
    from .quoted import QuotedDef, SelectorSpec
    from .reference_values import ObjectRef, StateRef
    from .selector import Selector
    from .symbol import ImportRef, SourceSpec

    allowed = (type(None), bool, int, float, str, bytes, ImportRef, SourceSpec)
    structural = (
        Expr, Template, Definition, ConcreteDefinition, FactorySpec, DefLink,
        QuotedDef, SelectorSpec, Selector, ObjectRef, StateRef,
    )
    seen: set[int] = set()

    def visit(current: object) -> None:
        if isinstance(current, (set, frozenset, FrozenSet)):
            for member in current:
                if isinstance(member, structural) or not isinstance(member, allowed):
                    raise ParameterizationError("template set members must be literal portable values")
            return
        if isinstance(current, (Template, TemplateBundle)):
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


def _normalize_bindings(
    sub_dict: Mapping[str, object] | None,
    namespace: object,
    bindings: Mapping[str, object],
) -> dict[str, object]:
    namespace_parts = _normalize_namespace(namespace, "namespace")
    if sub_dict is not None and not isinstance(sub_dict, Mapping):
        raise ParameterizationError("sub_dict must be a mapping")
    result: dict[str, object] = {}
    for name, value in (() if sub_dict is None else sub_dict.items()):
        normalized = _validate_root(name, "sub_dict key")
        if normalized in result:
            raise ParameterizationError(f"duplicate template binding for {normalized!r}")
        result[normalized] = value
    for name, value in bindings.items():
        if not _COMPONENT.fullmatch(name):
            raise ParameterizationError(f"keyword binding {name!r} is not a valid root component")
        normalized = _validate_root("/".join((*namespace_parts, name)), "keyword binding")
        if normalized in result:
            raise ParameterizationError(f"duplicate template binding for {normalized!r}")
        result[normalized] = value
    return result


def _normalize_remap(mapping: Mapping[str, str] | None, roots: set[str]) -> dict[str, str]:
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ParameterizationError("template remap mapping must be a mapping")
    result: dict[str, str] = {}
    for source, target in mapping.items():
        source = _validate_root(source, "remap source")
        target = _validate_root(target, "remap target")
        if source not in roots:
            raise ParameterizationError(f"remap source {source!r} is not an active template root")
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
        raise ParameterizationError("template remap produced an empty root")
    return _validate_root("/".join(parts), "remapped root")


def _snapshot_parameters(value: object, *, traverse_refs: bool) -> tuple[Par, ...]:
    """Collect the pre-operation parameters allowed by one traversal boundary."""

    from .cdef_graph import EdgeKind
    from .definition import Definition
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
                and current.kind is EdgeKind.REF
                and isinstance(current.target, (Definition, Template, TemplateBundle))
            ):
                if isinstance(current.target, Definition):
                    visit(current.target)
                elif isinstance(current.target, Template):
                    visit(current.target.root)
                else:
                    for recipe in current.target.recipes.values():
                        visit(recipe.root)
            elif current.kind is EdgeKind.MATERIALIZE:
                visit(current.target)
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
            raise ParameterizationError("Template.sub does not accept Distribution values")
        if isinstance(current, Template):
            visit(current.root)
            return
        if isinstance(current, TemplateBundle):
            for recipe in current.recipes.values():
                visit(recipe.root)
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
            elif isinstance(value, StateRef):
                try:
                    value = value.at(path)
                except ValueError as materializing_error:
                    try:
                        value = value.reference_value_at(path)
                    except ValueError:
                        raise materializing_error
            elif isinstance(value, ObjectRef):
                value = value.at(path)
            elif isinstance(value, ConcreteDefinition):
                value = value.graph_path(path)
            elif isinstance(value, Definition) and any(isinstance(segment, Parameter) for segment in path):
                if value.cls is None or not isinstance(value.cls, type):
                    raise ParameterizationError("semantic path requires an available soft Definition class")
                first, *rest = tuple(path)
                if not isinstance(first, Parameter):
                    raise ParameterizationError("semantic Definition path must begin with a Parameter")
                value = value.parameters[first.name]
                if rest:
                    value = get_subtree(value, GraphPath(tuple(rest)))
            else:
                value = get_subtree(value, path)
    except ParameterizationError:
        raise
    except Exception as error:
        raise ParameterizationError(f"template binding path is invalid for root {root!r}", path=path, root=root) from error
    return _freeze_binding_value(value)


def _freeze_binding_value(value: object) -> object:
    """Detach containers and lower live Object leaves without traversing their graphs."""

    from .canonical import freeze_def_value
    from .object import Object

    memo: dict[int, object] = {}

    def lower(current: object) -> object:
        if isinstance(current, Object):
            return current.object_ref
        if isinstance(current, (Template, TemplateBundle)):
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
        raise ParameterizationError("template binding value is unsupported") from error


def _rewrite_template_value(
    value: object,
    *,
    replacements: Mapping[int, object] | None,
    remap: Mapping[str, str] | None,
    traverse_refs: bool,
) -> object:
    """Perform one memoized copy-on-write traversal over pre-selected values."""

    from .bound_args import BoundArguments
    from .cdef_graph import EdgeKind
    from .definition import ConcreteDefinition, Definition
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
            marker = id(current)
            if marker in memo:
                return memo[marker]
            if current.kind is EdgeKind.REF:
                if not (
                        traverse_refs
                        and isinstance(current.target, (Definition, Template, TemplateBundle))):
                    return current
                if isinstance(current.target, Definition):
                    target = rewrite(current.target)
                elif isinstance(current.target, Template):
                    target = Template._from_root(rewrite(current.target.root))
                else:
                    target = TemplateBundle._from_recipes(FrozenDict(
                        (name, Template._from_root(rewrite(recipe.root)))
                        for name, recipe in current.target.recipes.items()
                    ))
            elif current.kind is EdgeKind.MATERIALIZE:
                target = rewrite(current.target)
            else:
                return current
            if target is current.target:
                result = current
            elif current.is_finalized:
                result = DefLink.finalized(current.kind, target)
            else:
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
                list(group) if isinstance(current.group, (list, FrozenList)) else group,
                count,
                current.shared,
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
            result = current if unchanged else FactorySpec._from_symbolic_parts(current.target, args, kwargs)
            memo[marker] = result
            return result
        if isinstance(current, ConcreteDefinition):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            parameters = FrozenDict(
                (key, rewrite(item)) for key, item in current.parameters.items()
            )
            unchanged = all(
                parameters[key] is item for key, item in current.parameters.items()
            )
            result = current if unchanged else ConcreteDefinition._from_bound_record(
                current.cls,
                BoundArguments(parameters.items()),
                stateful_role=current._stateful_role,
            )
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
            result = current if unchanged else Definition._from_symbolic_parts(current.cls, args, kwargs)
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


@dataclass(slots=True)
class _ExpressionBudget:
    """Track one template evaluation's cumulative repeat expansion allowance."""

    expansions: int = 0

    def charge_expansion(self, count: int) -> None:
        """Record generated ordered positions or fail before exceeding the hard cap."""

        if self.expansions + count > _MAX_EXPRESSION_VALUES:
            raise ParameterizationLimitError("template expansion limit exceeded")
        self.expansions += count


def _evaluate_template_value(value: object, *, traverse_refs: bool = False) -> object:
    """Evaluate resolved expression leaves with one cumulative expansion budget."""

    from .bound_args import BoundArguments
    from .definition import ConcreteDefinition, Definition
    from .cdef_graph import EdgeKind
    from .factory import FactorySpec
    from .links import DefLink

    budget = _ExpressionBudget()
    memo: dict[int, object] = {}

    def evaluate(current: object, depth: int = 0) -> object:
        if isinstance(current, Par):
            return current
        if isinstance(current, _BinaryExpr):
            depth += 1
            if depth > _MAX_EXPRESSION_DEPTH:
                raise ParameterizationLimitError("template expression depth limit exceeded")
            marker = id(current)
            if marker in memo:
                return memo[marker]
            left = evaluate(current.left, depth)
            right = evaluate(current.right, depth)
            if isinstance(left, Expr) or isinstance(right, Expr):
                result = current if left is current.left and right is current.right else _BinaryExpr(
                    current.operation, left, right
                )
            else:
                if not _validate_number(left) or not _validate_number(right):
                    raise ParameterizationError("template arithmetic operands must be exact finite built-in numbers")
                _validate_integer_limit(left)
                _validate_integer_limit(right)
                try:
                    if current.operation == "mul":
                        result = left * right
                    elif current.operation == "truediv":
                        result = left / right
                    elif current.operation == "floordiv":
                        result = left // right
                    else:
                        raise ParameterizationError("template arithmetic operation is unsupported")
                except ZeroDivisionError as error:
                    raise ParameterizationError("template arithmetic division by zero") from error
                except OverflowError as error:
                    raise ParameterizationError("template arithmetic result is not finite") from error
                if not _validate_number(result):
                    raise ParameterizationError("template arithmetic result must be a finite built-in number")
                _validate_integer_limit(result)
            memo[marker] = result
            return result
        if isinstance(current, _RepeatExpr):
            depth += 1
            if depth > _MAX_EXPRESSION_DEPTH:
                raise ParameterizationLimitError("template expression depth limit exceeded")
            marker = id(current)
            if marker in memo:
                return memo[marker]
            count = evaluate(current.count, depth)
            if isinstance(count, Expr):
                group = tuple(evaluate(item, depth) for item in current.group)
                result = current if (
                    count is current.count
                    and all(new is old for new, old in zip(group, current.group))
                ) else _RepeatExpr(
                    list(group) if isinstance(current.group, FrozenList) else group,
                    count,
                    current.shared,
                )
                memo[marker] = result
                return result
            if type(count) is not int or count < 0:
                raise ParameterizationError("template repetition count must resolve to a nonnegative exact int")
            if count > _MAX_REPEAT_COUNT:
                raise ParameterizationLimitError("template repetition count limit exceeded")
            budget.charge_expansion(count * len(current.group))
            if current.shared:
                group = tuple(evaluate(item, depth) for item in current.group)
                items = group * count
            else:
                items = []
                copied_groups = []
                for _ in range(count):
                    copy_memo: dict[int, object] = {}
                    copied_groups.append(tuple(
                        _copy_template_construction_value(item, copy_memo)
                        for item in current.group
                    ))
                for group in copied_groups:
                    items.extend(
                        evaluate(item, depth) for item in group
                    )
                items = tuple(items)
            result = FrozenList(items) if isinstance(current.group, FrozenList) else FrozenTuple(items)
            memo[marker] = result
            return result
        if isinstance(current, (Template, TemplateBundle)):
            return current
        if isinstance(current, DefLink):
            if current.kind is EdgeKind.REF:
                if not (
                        traverse_refs
                        and isinstance(current.target, (Definition, Template, TemplateBundle))):
                    return current
                if isinstance(current.target, Definition):
                    target = evaluate(current.target, depth)
                elif isinstance(current.target, Template):
                    target = Template._from_root(evaluate(current.target.root, depth))
                else:
                    target = TemplateBundle._from_recipes(FrozenDict(
                        (name, Template._from_root(evaluate(recipe.root, depth)))
                        for name, recipe in current.target.recipes.items()
                    ))
            elif current.kind is EdgeKind.MATERIALIZE:
                target = evaluate(current.target, depth)
            else:
                return current
            if target is current.target:
                return current
            if current.is_finalized:
                return DefLink.finalized(current.kind, target)
            return DefLink.assertion(current.kind, target)
        if isinstance(current, ConcreteDefinition):
            parameters = FrozenDict(
                (key, evaluate(item, depth))
                for key, item in current.parameters.items()
            )
            if all(parameters[key] is item for key, item in current.parameters.items()):
                return current
            return ConcreteDefinition._from_bound_record(
                current.cls,
                BoundArguments(parameters.items()),
                stateful_role=current._stateful_role,
            )
        if isinstance(current, FactorySpec):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            args = tuple(evaluate(item, depth) for item in current.args)
            kwargs = FrozenDict((key, evaluate(item, depth)) for key, item in current.kwargs.items())
            result = current if (
                all(new is old for new, old in zip(args, current.args))
                and all(kwargs[key] is item for key, item in current.kwargs.items())
            ) else FactorySpec._from_symbolic_parts(current.target, args, kwargs)
            memo[marker] = result
            return result
        if isinstance(current, Definition):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            args = None if current.args is None else FrozenTuple(evaluate(item, depth) for item in current.args)
            kwargs = FrozenDict((key, evaluate(item, depth)) for key, item in current.kwargs.items())
            result = current if (
                (args is None and current.args is None)
                or (args is not None and all(new is old for new, old in zip(args, current.args)))
            ) and all(kwargs[key] is item for key, item in current.kwargs.items()) else Definition._from_symbolic_parts(
                current.cls, args, kwargs
            )
            memo[marker] = result
            return result
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            items = [(key, evaluate(item, depth)) for key, item in current.items()]
            result = current if all(new is current[key] for key, new in items) else (
                FrozenDict(items) if isinstance(current, FrozenDict) else dict(items)
            )
            memo[marker] = result
            return result
        if isinstance(current, (list, FrozenList)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            items = tuple(evaluate(item, depth) for item in current)
            result = current if all(new is old for new, old in zip(items, current)) else FrozenList(items)
            memo[marker] = result
            return result
        if isinstance(current, (tuple, FrozenTuple)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            items = tuple(evaluate(item, depth) for item in current)
            result = current if all(new is old for new, old in zip(items, current)) else FrozenTuple(items)
            memo[marker] = result
            return result
        if isinstance(current, (set, frozenset, FrozenSet)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            items = tuple(evaluate(item, depth) for item in current)
            result = current if all(new is old for new, old in zip(items, current)) else FrozenSet(items)
            memo[marker] = result
            return result
        return current

    return evaluate(value)


def _copy_template_construction_value(value: object, memo: dict[int, object]) -> object:
    """Copy owned construction nodes while leaving explicit reference boundaries exact."""

    from .bound_args import BoundArguments
    from .definition import ConcreteDefinition, Definition
    from .cdef_graph import EdgeKind
    from .factory import FactorySpec
    from .links import DefLink

    if isinstance(value, Template):
        return value
    if isinstance(value, DefLink):
        if value.kind is EdgeKind.REF:
            return value
        marker = id(value)
        if marker in memo:
            return memo[marker]
        target = _copy_template_construction_value(value.target, memo)
        if value.is_finalized:
            result = DefLink.finalized(value.kind, target)
        else:
            result = DefLink.assertion(value.kind, target)
        memo[marker] = result
        return result
    marker = id(value)
    if marker in memo:
        return memo[marker]
    if isinstance(value, ConcreteDefinition):
        parameters = FrozenDict(
            (key, _copy_template_construction_value(item, memo))
            for key, item in value.parameters.items()
        )
        result = ConcreteDefinition._from_bound_record(
            value.cls,
            BoundArguments(parameters.items()),
            stateful_role=value._stateful_role,
        )
        memo[marker] = result
        return result
    if isinstance(value, Definition):
        args = None if value.args is None else FrozenTuple(
            _copy_template_construction_value(item, memo) for item in value.args
        )
        kwargs = FrozenDict(
            (key, _copy_template_construction_value(item, memo))
            for key, item in value.kwargs.items()
        )
        result = Definition._from_symbolic_parts(value.cls, args, kwargs)
        memo[marker] = result
        return result
    if isinstance(value, FactorySpec):
        args = tuple(_copy_template_construction_value(item, memo) for item in value.args)
        kwargs = FrozenDict(
            (key, _copy_template_construction_value(item, memo))
            for key, item in value.kwargs.items()
        )
        result = FactorySpec._from_symbolic_parts(value.target, args, kwargs)
        memo[marker] = result
        return result
    if isinstance(value, _BinaryExpr):
        result = _BinaryExpr(
            value.operation,
            _copy_template_construction_value(value.left, memo),
            _copy_template_construction_value(value.right, memo),
        )
        memo[marker] = result
        return result
    if isinstance(value, _RepeatExpr):
        group = tuple(_copy_template_construction_value(item, memo) for item in value.group)
        result = _RepeatExpr(
            list(group) if isinstance(value.group, FrozenList) else group,
            _copy_template_construction_value(value.count, memo),
            value.shared,
        )
        memo[marker] = result
        return result
    if isinstance(value, Mapping):
        items = [(key, _copy_template_construction_value(item, memo)) for key, item in value.items()]
        result = FrozenDict(items) if isinstance(value, FrozenDict) else dict(items)
        memo[marker] = result
        return result
    if isinstance(value, (list, FrozenList)):
        result = FrozenList(_copy_template_construction_value(item, memo) for item in value)
        memo[marker] = result
        return result
    if isinstance(value, (tuple, FrozenTuple)):
        result = FrozenTuple(_copy_template_construction_value(item, memo) for item in value)
        memo[marker] = result
        return result
    if isinstance(value, (set, frozenset, FrozenSet)):
        result = FrozenSet(_copy_template_construction_value(item, memo) for item in value)
        memo[marker] = result
        return result
    return value


__all__ = ["Expr", "Par", "Shared", "Template", "TemplateBundle", "repeat"]
