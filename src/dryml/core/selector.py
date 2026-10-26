"""Explicit query predicates and structural selectors for Node metadata.

Ordinary Node patterns need no wrapper. QueryPredicate owns boolean, explicit
absence and captured-support syntax; shared query preparation owns interpretation.
Authoring these values performs no construction, capture or Store IO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True, init=False, eq=False)
class QueryPredicate:
    """Explicit boolean, absence, or correlated-support query value.

    Args:
        value (Node | GeneratorSelector | QueryPredicate): Supplied pattern or
            captured exact support. Ordinary Node selection needs no wrapper.
    Raises:
        TypeError: For an unsupported value or boolean operand.
    Side Effects:
        None. Authoring never binds defaults, constructs, captures, allocates
        identities, or accesses a Store. Runtime providers remain runtime-only.
    """

    kind: str
    value: Any
    operands: tuple

    def __init__(self, value):
        from .node import Node
        from .generator import GeneratorSelector
        if isinstance(value, QueryPredicate):
            kind, value, operands = value.kind, value.value, value.operands
        elif isinstance(value, Node):
            kind, operands = "node", ()
        elif isinstance(value, GeneratorSelector):
            kind, operands = "support", ()
        else:
            raise TypeError("QueryPredicate requires a Node or captured Generator support.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "operands", operands)

    @classmethod
    def _make(cls, kind, value=None, operands=()):
        result = object.__new__(cls)
        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "value", value)
        object.__setattr__(result, "operands", operands)
        return result

    @classmethod
    def absent(cls, facet, *, path="$"):
        """Require an existing Node occurrence to lack one exact facet.

        Args:
            facet (str): ``identity`` or ``state``; other facets stay unspecified.
            path (GraphPathLike): Metadata path to the constrained Node.
        Returns:
            QueryPredicate: Immutable explicit absence, distinct from omission.
        Raises:
            ValueError: For an unknown facet.
            GraphPathError: For a malformed path. A missing endpoint does not match.
        Side Effects:
            None. Reference and quotation paths are inspected only as metadata.
        """
        from .utils.graph.path import normalize_path
        if facet not in {"identity", "state"}:
            raise ValueError("Absence facet must be identity or state.")
        return cls._make("absent", (facet, normalize_path(path)))

    @classmethod
    def support(cls, generator, *, max_assignments=65_536):
        """Capture exact correlated Generator support, retaining its proof cap.

        Args:
            generator (Generator): Description and provider capture to verify.
            max_assignments (int): Positive proof bound, at most 65,536.
        Returns:
            QueryPredicate: Conservative prefilter plus exact assignment witness.
        Raises:
            TypeError: If ``generator`` is not a Generator.
            ParameterizationError: For invalid input, providers or proof bound.
        Side Effects:
            None until matching; matching may call provider proof methods, never
            sampling. Unsupported proofs and exceeded caps raise, not approximate.
        """
        from .generator import Generator
        if not isinstance(generator, Generator):
            raise TypeError("support requires a Generator.")
        return cls(generator.support_selector(max_assignments=max_assignments))

    def __and__(self, other):
        """Return conjunction with a Node/predicate; invalid operands raise TypeError."""
        return self._make("and", operands=(self, QueryPredicate(other)))

    def __or__(self, other):
        """Return disjunction with a Node/predicate; invalid operands raise TypeError."""
        return self._make("or", operands=(self, QueryPredicate(other)))

    def __invert__(self):
        """Return explicit logical negation; omission itself is never negation."""
        return self._make("not", operands=(self,))

    def __bool__(self):
        raise TypeError("Compose QueryPredicate values with &, | and ~, not Python and/or/not.")

    def compile(self, *, domain="node"):
        """Prepare this predicate once against detached metadata semantics.

        Args:
            domain (str): ``node`` for recipe/identified/saved graph metadata,
                ``local_state`` for descriptors, or ``occurrence`` for a caller's
                explicitly selected relationship endpoint. These modes do not
                assert independent Store membership in any of those domains.
        Returns:
            An operation-owned prepared matcher with ``matches(target)``. Its
            NodePreparation owns graph facts; no Store coverage is implied.
        Raises:
            QueryDomainError: For invalid domains or donor-identity constraints in
                local-state queries, including masked boolean branches.
            TypeError: For unsupported graph leaves or unavailable parameter names.
        Side Effects:
            Pure metadata preparation only. No defaults, target construction,
            identity allocation, capture or state IO.
        """
        from .query.node_pattern import _PreparedPredicate
        return _PreparedPredicate(self, domain=domain)

    def matches(self, target, *, domain="node"):
        """Return exact predicate acceptance of detached Node/LocalState metadata.

        Args:
            target (Node | LocalState): Metadata candidate, never a live Object.
            domain (str): Explicit domain accepted by :meth:`compile`.
        Returns:
            bool: Whether every supplied constraint and exact witness holds.
        Raises:
            QueryDomainError: For an inapplicable domain/identity constraint.
            TypeError: For unsupported metadata values.
            ParameterizationError: For provider failures or unavailable/bounded proof.
        Side Effects:
            As for ``compile``; support proof can invoke trusted provider methods.
        """
        return self.compile(domain=domain).matches(target)

    def to_data(self):
        """Return closed ``dryml-node-query-predicate`` v1 portable metadata.

        Returns a mapping using the shared Node codec, preserving aliases, roles,
        exact facets, Match leaves and built-in domains. Runtime providers raise
        ParameterizationError; unsupported values raise TypeError/ValueError.
        Encoding never invokes a provider, resolves targets or accesses a Store.
        """
        from .query.node_pattern import _predicate_to_data
        return _predicate_to_data(self)

    @classmethod
    def from_data(cls, data):
        """Decode closed predicate data without target resolution or execution.

        Args:
            data (Mapping): Canonical v1 predicate envelope from ``to_data``.
        Returns:
            QueryPredicate: Fresh handles retaining the encoded alias partition.
        Raises:
            ValueError, ParameterizationError: For malformed/unsupported data or
                exceeded codec bounds. Opaque provider deserialization is forbidden.
        Side Effects:
            None beyond pure metadata decoding and validation.
        """
        from .query.node_pattern import _predicate_from_data
        return _predicate_from_data(data)


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
            for a later :meth:`exact` or :meth:`restore` edit. Node endpoints are
            edited at every alias; independent equal nodes and quoted data remain
            separate.

        Raises:
            QueryPathError: If the selected occurrence is unavailable.
            TypeError: If projection controls are invalid.

        Side Effects:
            None. This performs selector-only editing and never constructs an
            Object or accesses Store authority.
        """

        from .categorical import _project_categorical_definition_with_origins
        from .query.path import get_subtree, normalize_path

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
            _replace_selector_subtree(self.root, norm, projected, _origins=origins),
            strict=self.strict,
            cls_policy=self.cls_policy,
            exact_root=self.exact_root,
        )

    def restore(self, *, path="$") -> "Selector":
        """Restore one selector subtree from retained exact CDef authority.

        Args:
            path: Root-relative subtree to restore.

        Returns:
            A selector with the original subtree restored at every alias of a
            selected Node. Independent equal nodes and quoted data stay separate;
            non-Node endpoints retain structural-slot replacement semantics.

        Raises:
            QueryPathError: If no original exact CDef is available or ``path``
                cannot be resolved.

        Side Effects:
            None. Restoration is an immutable selector edit.
        """

        from .query.path import QueryPathError, get_subtree, normalize_path

        if self.exact_root is None:
            raise QueryPathError("restore() requires an original ConcreteDefinition.")
        norm = normalize_path(path)
        exact_path = _exact_authority_path(self.root, self.exact_root, norm)
        return Selector(
            _replace_selector_subtree(self.root, norm, get_subtree(self.exact_root, exact_path)),
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
            A selector carrying the exact replacement at every alias of a selected
            Node. Independent equal nodes and quoted data remain separate.

        Raises:
            QueryPathError: If no exact authority exists for an implicit edit.
            TypeError: If the replacement is not a ConcreteDefinition.

        Side Effects:
            None. This does not fill defaults or construct a target.
        """

        from .definition import ConcreteDefinition
        from .query.path import QueryPathError, get_subtree, normalize_path

        norm = normalize_path(path)
        if definition is None:
            if self.exact_root is None:
                raise QueryPathError("exact() requires an explicit ConcreteDefinition.")
            exact_path = _exact_authority_path(self.root, self.exact_root, norm)
            definition = get_subtree(self.exact_root, exact_path)
        if not isinstance(definition, ConcreteDefinition):
            raise TypeError("exact() requires a ConcreteDefinition.")
        return Selector(
            _replace_selector_subtree(self.root, norm, definition),
            strict=self.strict,
            cls_policy=self.cls_policy,
            exact_root=self.exact_root,
        )


def _replace_selector_subtree(root, path, replacement, *, _origins=None):
    """Use occurrence-coherent Node edits while keeping literal slot edits structural."""
    from .node import Node, _coalesce, _rewrite
    from .query.path import get_subtree, replace_subtree

    selected = get_subtree(root, path)
    if isinstance(selected, Node) and isinstance(replacement, Node):
        return _coalesce(_rewrite(root, {selected._occurrence: replacement},
                                  reset_construction=True, enter_quoted=False))
    return replace_subtree(root, path, replacement, _origins=_origins)


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


def selector(root: Any = None, **kwargs) -> Selector:
    """Create an immutable structural selector from supplied metadata only.

    Args:
        root: Definition-like structural selector root. ``None`` creates an
            unconstrained Definition selector.
        **kwargs: Selector policies accepted by :class:`Selector`.

    Returns:
        An immutable Selector retaining the supplied root and policies.

    Raises:
        TypeError: If the root or keyword policies are unsupported.
        ValueError: If the supplied graph or policies are invalid.

    Side Effects:
        None. This neither reads Store authority nor constructs Objects.
    """

    from .definition import ConcreteDefinition, Definition

    if root is None:
        root = Definition()
    elif not isinstance(root, (Definition, ConcreteDefinition)):
        root = Definition(root)
    return Selector(root, **kwargs)
