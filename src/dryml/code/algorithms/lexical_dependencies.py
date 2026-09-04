"""Static lexical free-name evidence built solely from program-graph syntax."""

from __future__ import annotations

import builtins
from dataclasses import dataclass

from ..analysis import analyze
from ..facts import FactValue, SourceLocation
from ..graph import ProgramGraph, ProgramNode, _source_sort_key as _source_key
from ..kernels import KernelCall, KernelContext, TraversalKernel
from ..targets import CodeTargetInput, SourceTarget


_FUNCTION_KINDS = frozenset({"FunctionDef", "AsyncFunctionDef"})
_COMPREHENSION_KINDS = frozenset({"ListComp", "SetComp", "DictComp", "GeneratorExp"})
_TYPE_PARAMETER_KINDS = frozenset({"TypeVar", "ParamSpec", "TypeVarTuple"})
_STATEMENT_KINDS = frozenset({
    "Assert", "Assign", "AnnAssign", "AugAssign", "AsyncFor", "AsyncWith",
    "Break", "ClassDef", "Continue", "Delete", "Expr", "For", "FunctionDef",
    "AsyncFunctionDef", "Global", "If", "Import", "ImportFrom", "Match",
    "Nonlocal", "Pass", "Raise", "Return", "Try", "TryStar", "TypeAlias",
    "While", "With", "Yield", "YieldFrom",
})


@dataclass(frozen=True, slots=True)
class LexicalDependency:
    """One statically free name and the location of its first visible use.

    Args:
        name: Exact Python identifier used without a lexical binding.
        source: Sanitized location of the first free load, if graph syntax has
            source coordinates.

    Raises:
        ValueError: If the name or source carrier is invalid.

    Side Effects:
        None. This value retains neither source text nor a live Python value.
    """

    name: str
    source: SourceLocation | None

    def __post_init__(self) -> None:
        """Validate immutable, dependency-light lexical evidence."""

        if type(self.name) is not str or not self.name:
            raise ValueError("lexical dependency name is invalid")
        if self.source is not None and type(self.source) is not SourceLocation:
            raise ValueError("lexical dependency source is invalid")


@dataclass(frozen=True, slots=True)
class LexicalDependencies:
    """Deterministically ordered lexical free-name evidence for one target.

    Args:
        dependencies: Exact tuple of one entry per free name, ordered by the
            first free use's sanitized source coordinate.

    Raises:
        ValueError: If entries are not exact :class:`LexicalDependency` values.

    Side Effects:
        None. The aggregate contains no target, import, or live-value handles.
    """

    dependencies: tuple[LexicalDependency, ...]

    def __post_init__(self) -> None:
        """Reject mutable containers and subclassed evidence entries."""

        if type(self.dependencies) is not tuple or any(type(item) is not LexicalDependency for item in self.dependencies):
            raise ValueError("lexical dependencies are invalid")


@dataclass(frozen=True, slots=True)
class _SyntaxNode:
    """Private graph-only syntax projection used by the scope walker."""

    node: ProgramNode
    kind: str
    fields: dict[str, FactValue]
    children: tuple["_SyntaxNode", ...]
    name: tuple[str, str] | None

    @property
    def source(self) -> SourceLocation | None:
        """Return the graph's already sanitized source evidence."""

        return self.node.source


@dataclass(frozen=True, slots=True)
class _DependencyUse:
    """Private candidate free-name occurrence before deterministic deduplication."""

    name: str
    source: SourceLocation | None


@dataclass(frozen=True, slots=True)
class _TraversalDependencies(LexicalDependencies):
    """Private mutable-node carrier satisfying the public traversal state type."""

    nodes: list[ProgramNode]


def _mapping(value: FactValue) -> dict[str, FactValue]:
    """Return one graph-owned closed mapping as a private convenience view."""

    return dict(value)  # type: ignore[arg-type]


def _node_key(node: _SyntaxNode) -> tuple[object, ...]:
    """Order private syntax projections as their public graph evidence does."""

    return (_source_key(node.source), node.kind, node.node.id)


def _is_statement(node: _SyntaxNode) -> bool:
    """Return whether a syntax kind can occur directly in a definition body."""

    return node.kind in _STATEMENT_KINDS


def _name_field(node: _SyntaxNode) -> str | None:
    """Return a scalar binding field only when graph syntax carries one."""

    value = node.fields.get("name")
    return value if type(value) is str else None


def _syntax_roots(nodes: list[ProgramNode], graph: ProgramGraph) -> tuple[_SyntaxNode, ...]:
    """Rebuild a private syntax forest from public containment and name evidence."""

    by_id = {node.id: node for node in nodes if node.kind == "syntax"}
    graph_nodes = {node.id: node for node in graph.nodes}
    children: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    names: dict[str, tuple[str, str]] = {}
    for edge in graph.edges:
        if edge.kind == "containment" and edge.source in by_id and edge.target in by_id:
            children[edge.source].append(edge.target)
        elif edge.kind == "lexical_reference" and edge.source in by_id:
            symbol = graph_nodes.get(edge.target)
            if symbol is not None and symbol.kind == "lexical_symbol":
                payload = _mapping(symbol.value)
                name, role = payload.get("name"), payload.get("role")
                if type(name) is str and type(role) is str:
                    names[edge.source] = (name, role)

    def build(node_id: str) -> _SyntaxNode:
        """Build one private syntax projection recursively from immutable graph data."""

        node = by_id[node_id]
        payload = _mapping(node.value)
        fields = {
            field[1]: field[2]
            for field in payload["fields"]  # type: ignore[index, union-attr]
            if type(field) is tuple and len(field) == 3 and field[0] == "field" and type(field[1]) is str
        }
        child_nodes = tuple(sorted((build(child_id) for child_id in children[node_id]), key=_node_key))
        return _SyntaxNode(node, payload["type"], fields, child_nodes, names.get(node_id))  # type: ignore[index, arg-type]

    contained = {child_id for child_ids in children.values() for child_id in child_ids}
    return tuple(sorted((build(node_id) for node_id in by_id if node_id not in contained), key=_node_key))


def _bind_target(node: _SyntaxNode) -> set[str]:
    """Return lexical store names in one target without entering nested scopes."""

    if node.kind in _FUNCTION_KINDS or node.kind in {"ClassDef", "Lambda"}:
        return set()
    if node.kind in _COMPREHENSION_KINDS:
        return _namedexpr_bindings(node)
    if node.name is not None and node.name[1] == "bind":
        return {node.name[0]}
    names = set()
    if node.kind in {"Import", "ImportFrom"}:
        for child in node.children:
            if child.kind == "alias":
                imported = child.fields.get("name")
                asname = child.fields.get("asname")
                if type(imported) is str:
                    names.add(asname if type(asname) is str else imported.split(".", 1)[0])
        return names
    if node.kind == "ExceptHandler":
        name = _name_field(node)
        if name is not None:
            names.add(name)
    if node.kind.startswith("Match"):
        for field_name in ("name", "rest"):
            value = node.fields.get(field_name)
            if type(value) is str:
                names.add(value)
    for child in node.children:
        names |= _bind_target(child)
    return names


def _namedexpr_bindings(node: _SyntaxNode) -> set[str]:
    """Return assignment-expression targets that escape a comprehension scope."""

    if node.kind in _FUNCTION_KINDS or node.kind in {"ClassDef", "Lambda"}:
        return set()
    if node.kind == "NamedExpr":
        return set().union(*(_bind_target(child) for child in node.children))
    return set().union(*(_namedexpr_bindings(child) for child in node.children)) if node.children else set()


def _argument_names(node: _SyntaxNode) -> set[str]:
    """Return all graph-carried argument bindings from one arguments syntax node."""

    return {
        value
        for child in node.children
        if child.kind == "arg"
        for value in (child.fields.get("arg"),)
        if type(value) is str
    }


def _type_parameter_names(nodes: tuple[_SyntaxNode, ...]) -> set[str]:
    """Return maintained-version type parameter names without version-specific AST imports."""

    return {
        value
        for node in nodes
        if node.kind in _TYPE_PARAMETER_KINDS
        for value in (node.fields.get("name"),)
        if type(value) is str
    }


def _declared_globals(nodes: tuple[_SyntaxNode, ...]) -> set[str]:
    """Return global declarations while excluding nested lexical scopes."""

    names: set[str] = set()
    for node in nodes:
        if node.kind in _FUNCTION_KINDS or node.kind in {"ClassDef", "Lambda"}:
            continue
        if node.kind == "Global":
            values = node.fields.get("names")
            if type(values) is tuple:
                names |= {value for value in values if type(value) is str}
            continue
        names |= _declared_globals(node.children)
    return names


def _scope_bindings(body: tuple[_SyntaxNode, ...], globals_: set[str]) -> set[str]:
    """Return all function-local bindings, independent of assignment order."""

    bound: set[str] = set()
    for statement in body:
        bound |= _statement_bindings(statement)
    return bound - globals_


def _statement_bindings(statement: _SyntaxNode) -> set[str]:
    """Return names introduced after one statement executes."""

    if statement.kind in _FUNCTION_KINDS or statement.kind == "ClassDef":
        name = _name_field(statement)
        return {name} if name is not None else set()
    return _bind_target(statement)


class _ScopeWalker:
    """Private lexical scope interpreter over public graph syntax nodes."""

    def __init__(self, roots: tuple[_SyntaxNode, ...]) -> None:
        """Initialize the private syntax forest and deterministic built-in names."""

        self._roots = roots
        self._builtins = frozenset(dir(builtins))

    def collect(self) -> LexicalDependencies:
        """Return deduplicated graph-only free-name evidence for supported roots."""

        uses: list[_DependencyUse] = []
        pending = list(self._roots)
        while pending:
            root = pending.pop()
            if root.kind in _FUNCTION_KINDS or root.kind in {"ClassDef", "Lambda"}:
                uses.extend(self._collect_scope(root, set()))
            else:
                pending.extend(reversed(root.children))
        first: dict[str, SourceLocation | None] = {}
        for use in uses:
            first.setdefault(use.name, use.source)
        return LexicalDependencies(tuple(
            LexicalDependency(name, source)
            for name, source in sorted(first.items(), key=lambda item: (_source_key(item[1]), item[0]))
        ))

    def _unresolved(
        self,
        uses: list[_DependencyUse],
        bound: set[str],
        available: set[str],
    ) -> list[_DependencyUse]:
        """Filter collected loads using only lexical bindings and built-in names."""

        return [
            use for use in uses
            if use.name not in bound and use.name not in available and use.name not in self._builtins
        ]

    def _collect_scope(
        self,
        node: _SyntaxNode,
        available: set[str],
        body_available: set[str] | None = None,
    ) -> list[_DependencyUse]:
        """Collect free uses for one function, class, or lambda lexical scope."""

        lexical_available = available if body_available is None else body_available
        if node.kind == "Lambda":
            arguments = next((child for child in node.children if child.kind == "arguments"), None)
            body = tuple(child for child in node.children if child is not arguments)
            header_uses: list[_DependencyUse] = []
            header_nested: list[_DependencyUse] = []
            if arguments is not None:
                self._walk(arguments, set(), header_uses, available, nested_uses=header_nested)
            bound = _argument_names(arguments) if arguments is not None else set()
            body_uses: list[_DependencyUse] = []
            body_nested: list[_DependencyUse] = []
            for child in body:
                self._walk(child, bound, body_uses, lexical_available | bound, nested_uses=body_nested)
            return (
                self._unresolved(header_uses, set(), available)
                + header_nested
                + self._unresolved(body_uses, bound, lexical_available)
                + body_nested
            )

        arguments = next((child for child in node.children if child.kind == "arguments"), None)
        type_parameters = tuple(child for child in node.children if child.kind in _TYPE_PARAMETER_KINDS)
        body = tuple(child for child in node.children if _is_statement(child))
        excluded = {child.node.id for child in type_parameters + body}
        if arguments is not None:
            excluded.add(arguments.node.id)
        header = tuple(child for child in node.children if child.node.id not in excluded)
        type_names = _type_parameter_names(type_parameters)
        header_uses: list[_DependencyUse] = []
        header_nested: list[_DependencyUse] = []
        for child in type_parameters + header + ((arguments,) if arguments is not None else ()):
            self._walk(child, type_names, header_uses, available | type_names, nested_uses=header_nested)
        if node.kind in _FUNCTION_KINDS:
            global_names = _declared_globals(body)
            bound = _scope_bindings(body, global_names) | type_names
            if arguments is not None:
                bound |= _argument_names(arguments)
            name = _name_field(node)
            if name is not None:
                bound.add(name)
            body_uses: list[_DependencyUse] = []
            body_nested: list[_DependencyUse] = []
            for child in body:
                self._walk(child, bound, body_uses, lexical_available | bound, nested_uses=body_nested)
            return (
                self._unresolved(header_uses, type_names, available)
                + header_nested
                + self._unresolved(body_uses, bound, lexical_available)
                + body_nested
            )

        global_names = _declared_globals(body)
        class_bound = set(type_names)
        body_uses = []
        body_nested = []
        for child in body:
            statement_uses: list[_DependencyUse] = []
            self._walk(
                child,
                class_bound,
                statement_uses,
                lexical_available | class_bound,
                nested_available=lexical_available,
                nested_uses=body_nested,
            )
            body_uses.extend(self._unresolved(statement_uses, class_bound, lexical_available))
            class_bound |= _statement_bindings(child) - global_names
        return self._unresolved(header_uses, type_names, available) + header_nested + body_uses + body_nested

    def _walk(
        self,
        node: _SyntaxNode,
        bound: set[str],
        uses: list[_DependencyUse],
        available: set[str],
        namedexpr_bound: set[str] | None = None,
        nested_available: set[str] | None = None,
        nested_uses: list[_DependencyUse] | None = None,
    ) -> None:
        """Collect loads from one syntax node while honoring lexical boundaries."""

        if node.kind in _FUNCTION_KINDS or node.kind in {"ClassDef", "Lambda"}:
            destination = uses if nested_uses is None else nested_uses
            destination.extend(
                self._collect_scope(
                    node,
                    available | bound,
                    available | bound if nested_available is None else nested_available,
                )
            )
            return
        if node.kind in _COMPREHENSION_KINDS:
            self._walk_comprehension(
                node,
                bound,
                uses,
                available,
                namedexpr_bound,
                nested_available,
                nested_uses,
            )
            return
        if node.kind in {"Import", "ImportFrom"}:
            return
        if node.kind == "Name" and node.name is not None:
            if node.name[1] == "load":
                uses.append(_DependencyUse(node.name[0], node.source))
            return
        if node.kind == "NamedExpr":
            targets = [child for child in node.children if _bind_target(child)]
            for child in node.children:
                if child not in targets:
                    self._walk(child, bound, uses, available, namedexpr_bound, nested_available, nested_uses)
            target_bound = namedexpr_bound if namedexpr_bound is not None else bound
            for target in targets:
                target_bound |= _bind_target(target)
            return
        if node.kind in {"For", "AsyncFor"}:
            self._walk_targeted(node, bound, uses, available, namedexpr_bound, nested_available, nested_uses)
            return
        if node.kind == "ExceptHandler":
            exception = tuple(child for child in node.children if not _is_statement(child))
            body = tuple(child for child in node.children if _is_statement(child))
            for child in exception:
                self._walk(child, bound, uses, available, namedexpr_bound, nested_available, nested_uses)
            for child in body:
                self._walk(child, bound, uses, available, namedexpr_bound, nested_available, nested_uses)
            return
        if node.kind == "Match":
            for child in node.children:
                self._walk(child, bound, uses, available, namedexpr_bound, nested_available, nested_uses)
            return
        if node.kind == "match_case":
            patterns = tuple(child for child in node.children if child.kind.startswith("Match"))
            for pattern in patterns:
                self._walk(pattern, bound, uses, available, namedexpr_bound, nested_available, nested_uses)
            pattern_bound = set().union(*(_bind_target(pattern) for pattern in patterns)) if patterns else set()
            for child in node.children:
                if child not in patterns:
                    self._walk(child, bound | pattern_bound, uses, available | pattern_bound, namedexpr_bound, nested_available, nested_uses)
            return
        for child in node.children:
            self._walk(child, bound, uses, available, namedexpr_bound, nested_available, nested_uses)

    def _walk_targeted(
        self,
        node: _SyntaxNode,
        bound: set[str],
        uses: list[_DependencyUse],
        available: set[str],
        namedexpr_bound: set[str] | None,
        nested_available: set[str] | None,
        nested_uses: list[_DependencyUse] | None,
    ) -> None:
        """Walk a for-loop target after its iterable expression has been read."""

        targets = [child for child in node.children if _bind_target(child)]
        first_target = targets[0] if targets else None
        for child in node.children:
            if child is first_target:
                continue
            self._walk(child, bound, uses, available, namedexpr_bound, nested_available, nested_uses)

    def _walk_comprehension(
        self,
        node: _SyntaxNode,
        bound: set[str],
        uses: list[_DependencyUse],
        available: set[str],
        namedexpr_bound: set[str] | None,
        nested_available: set[str] | None,
        nested_uses: list[_DependencyUse] | None,
    ) -> None:
        """Interpret comprehension generators as their own lexical scope."""

        generators = tuple(child for child in node.children if child.kind == "comprehension")
        expressions = tuple(child for child in node.children if child not in generators)
        local_bound: set[str] = set()
        outer_named = namedexpr_bound if namedexpr_bound is not None else bound
        scope_available = available | bound if nested_available is None else nested_available
        for index, generator in enumerate(generators):
            targets = [child for child in generator.children if _bind_target(child)]
            target = targets[0] if targets else None
            expressions_in_generator = tuple(child for child in generator.children if child is not target)
            iterator = expressions_in_generator[0] if expressions_in_generator else None
            if iterator is not None:
                iterator_uses: list[_DependencyUse] = []
                iterator_nested: list[_DependencyUse] = []
                iterator_available = available | bound | local_bound if index == 0 else scope_available | local_bound
                iterator_bound = bound | local_bound if index == 0 else local_bound
                self._walk(
                    iterator,
                    iterator_bound,
                    iterator_uses,
                    iterator_available,
                    outer_named,
                    nested_available if index == 0 else None,
                    iterator_nested,
                )
                uses.extend(self._unresolved(iterator_uses, iterator_bound, iterator_available))
                (uses if nested_uses is None else nested_uses).extend(iterator_nested)
            if target is not None:
                local_bound |= _bind_target(target)
            for child in expressions_in_generator[1:]:
                filter_uses: list[_DependencyUse] = []
                filter_nested: list[_DependencyUse] = []
                self._walk(child, local_bound, filter_uses, scope_available | local_bound, outer_named, nested_uses=filter_nested)
                uses.extend(self._unresolved(filter_uses, local_bound, scope_available))
                (uses if nested_uses is None else nested_uses).extend(filter_nested)
        for expression in expressions:
            expression_uses: list[_DependencyUse] = []
            expression_nested: list[_DependencyUse] = []
            self._walk(expression, local_bound, expression_uses, scope_available | local_bound, outer_named, nested_uses=expression_nested)
            uses.extend(self._unresolved(expression_uses, local_bound, scope_available))
            (uses if nested_uses is None else nested_uses).extend(expression_nested)


class LexicalDependencyKernel(TraversalKernel[None, LexicalDependencies, LexicalDependencies]):
    """Collect deterministic free-name evidence from one immutable program graph.

    The kernel uses the inherited unfused traversal template and private local
    state only, so its ``fusion_safe`` declaration is valid for the conservative
    fused executor. It neither resolves names nor applies import policy.
    """

    input_type = type(None)
    output_type = LexicalDependencies
    fusion_safe = True

    def begin(self, value: None, context: KernelContext) -> LexicalDependencies:
        """Create private graph-node accumulation state.

        Args:
            value: Required ``None`` input; it carries no lexical policy.
            context: Read-only request context, not retained by this kernel.

        Returns:
            Empty private state for inherited canonical traversal.

        Raises:
            None.

        Side Effects:
            None.
        """

        return _TraversalDependencies((), [])

    def visit(self, node: ProgramNode, state: LexicalDependencies, context: KernelContext) -> LexicalDependencies:
        """Accumulate one public graph node without mutating graph data.

        Args:
            node: Current immutable canonical program node.
            state: Private state for this kernel execution.
            context: Read-only request context, not retained by this kernel.

        Returns:
            The same private state with one node reference appended.

        Raises:
            None.

        Side Effects:
            Mutates only this execution's private traversal state.
        """

        if type(state) is not _TraversalDependencies:
            raise TypeError("lexical traversal state is invalid")
        state.nodes.append(node)
        return state

    def finish(self, state: LexicalDependencies, context: KernelContext) -> LexicalDependencies:
        """Convert public syntax and evidence nodes into lexical dependencies.

        Args:
            state: Completed private canonical graph-node sequence.
            context: Read-only request context supplying the immutable graph.

        Returns:
            Ordered free-name evidence with sanitized locations only.

        Raises:
            None. Incomplete graphs without source syntax yield an empty result.

        Side Effects:
            None. No source is executed and no name is resolved or imported.
        """

        if type(state) is not _TraversalDependencies:
            raise TypeError("lexical traversal state is invalid")
        return _ScopeWalker(_syntax_roots(state.nodes, context.graph)).collect()


def collect_lexical_dependencies(target: CodeTargetInput) -> LexicalDependencies:
    """Collect static free-name evidence with the built-in lexical kernel.

    Args:
        target: Supported static code target or target wrapper.

    Returns:
        The same :class:`LexicalDependencies` artifact produced by a direct
        :class:`KernelCall` for :class:`LexicalDependencyKernel`.

    Raises:
        CodeAnalysisError: If target normalization or graph construction fails.
        MissingOutputError: If the conforming built-in output is unexpectedly
            unavailable from the analysis result.

    Side Effects:
        May read admitted source or perform an explicit ``ImportTarget`` module
        import during target normalization. It never executes target source,
        resolves lexical names, searches caller frames, or imports dependencies.
    """

    return analyze(target, (KernelCall(LexicalDependencyKernel(), None),)).require(LexicalDependencyKernel)


def _collect_source_dependencies(source: str) -> LexicalDependencies:
    """Keep core's source-only use inside the approved lexical leaf."""

    return collect_lexical_dependencies(SourceTarget(source))


__all__ = [
    "LexicalDependencies",
    "LexicalDependency",
    "LexicalDependencyKernel",
    "collect_lexical_dependencies",
]
