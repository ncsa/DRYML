"""Deterministic immutable foundational program graphs.

This module derives static syntax, lexical-name, attribute-access, and call
evidence without executing a target or retaining its request-local handles.
"""

from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .ast_tools import parse_source
from .errors import CodeAnalysisError
from .facts import Diagnostic, FactValue, SourceLocation
from .targets import CodeTargetInput, TargetInfo, normalize_target


ProgramNodeKind: TypeAlias = Literal[
    "target", "syntax", "lexical_symbol", "attribute_access", "static_call", "trace_event"
]
ProgramEdgeKind: TypeAlias = Literal[
    "containment", "lexical_reference", "access", "call",
    "trace_sequence", "frame_descent", "observed_code",
]

_NODE_KINDS = frozenset({"target", "syntax", "lexical_symbol", "attribute_access", "static_call", "trace_event"})
_EDGE_KINDS = frozenset({"containment", "lexical_reference", "access", "call", "trace_sequence", "frame_descent", "observed_code"})


def _graph_error() -> CodeAnalysisError:
    """Create the fixed public error used for invalid graph inputs and queries."""

    return CodeAnalysisError("invalid program graph", code="graph.invalid")


def _pack(tag: bytes, payload: bytes) -> bytes:
    """Encode one domain-separated component with an unambiguous byte length."""

    return tag + len(payload).to_bytes(8, "big") + payload


def _is_mapping(value: tuple[object, ...]) -> bool:
    """Return whether a tuple is the canonical closed mapping representation."""

    return all(type(item) is tuple and len(item) == 2 and type(item[0]) is str for item in value)


def _encode(value: FactValue) -> bytes:
    """Return private canonical bytes for one closed immutable graph value."""

    value_type = type(value)
    if value is None:
        return b"n"
    if value_type is bool:
        return b"b1" if value else b"b0"
    if value_type is int:
        return _pack(b"i", str(value).encode("ascii"))
    if value_type is float:
        if not math.isfinite(value):
            raise _graph_error()
        return _pack(b"f", format(value, ".17g").encode("ascii"))
    if value_type is str:
        return _pack(b"s", value.encode("utf-8"))
    if value_type is bytes:
        return _pack(b"y", value)
    if value_type is not tuple:
        raise _graph_error()
    if _is_mapping(value):
        keys = tuple(item[0] for item in value)
        if len(keys) != len(set(keys)):
            raise _graph_error()
        return b"m" + _pack(b"l", b"".join(_encode(item[0]) + _encode(item[1]) for item in sorted(value)))
    return b"t" + _pack(b"l", b"".join(_encode(item) for item in value))


def _mapping(value: FactValue, keys: tuple[str, ...]) -> dict[str, FactValue]:
    """Validate and unpack one exact canonical payload mapping."""

    if type(value) is not tuple or not _is_mapping(value):
        raise _graph_error()
    result = dict(value)
    if set(result) != set(keys) or len(result) != len(value):
        raise _graph_error()
    _encode(value)
    return result


def _canonical_value(value: FactValue) -> FactValue:
    """Return a recursively canonical closed value independent of map ordering."""

    value_type = type(value)
    if value is None or value_type in (bool, int, str, bytes):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise _graph_error()
        return value
    if value_type is not tuple:
        raise _graph_error()
    if _is_mapping(value):
        keys = tuple(item[0] for item in value)
        if len(keys) != len(set(keys)):
            raise _graph_error()
        return tuple((key, _canonical_value(item_value)) for key, item_value in sorted(value))
    return tuple(_canonical_value(item) for item in value)


def _source_value(source: SourceLocation | None) -> FactValue:
    """Project sanitized source coordinates into the closed digest vocabulary."""

    if source is None:
        return None
    return (
        ("column", source.column),
        ("filename", source.filename),
        ("line", source.line),
    )


def _target_value(target: TargetInfo) -> FactValue:
    """Return the exact target-root payload retained by foundational graphs."""

    return (
        ("descriptor_kind", target.descriptor_kind),
        ("filename", target.filename),
        ("import_path", target.import_path),
        ("kind", target.kind),
        ("module", target.module),
        ("name", target.name),
        ("owner_module", target.owner_module),
        ("owner_qualname", target.owner_qualname),
        ("qualname", target.qualname),
        ("start_line", target.start_line),
    )


def _validate_payload(kind: ProgramNodeKind, value: FactValue) -> None:
    """Validate the closed payload shape assigned to one node vocabulary kind."""

    if kind == "target":
        values = _mapping(value, (
            "descriptor_kind", "filename", "import_path", "kind", "module", "name",
            "owner_module", "owner_qualname", "qualname", "start_line",
        ))
        if values["kind"] not in {"function", "bound_method", "callable_instance", "descriptor", "class", "import", "source"}:
            raise _graph_error()
        return
    if kind == "syntax":
        values = _mapping(value, ("fields", "type"))
        if type(values["type"]) is not str or not values["type"].isidentifier() or type(values["fields"]) is not tuple:
            raise _graph_error()
        field_names: list[str] = []
        for field in values["fields"]:
            if type(field) is not tuple or len(field) != 3 or field[0] != "field" or type(field[1]) is not str:
                raise _graph_error()
            _encode(field[2])
            field_names.append(field[1])
        if len(field_names) != len(set(field_names)):
            raise _graph_error()
        return
    if kind == "lexical_symbol":
        values = _mapping(value, ("name", "role"))
        if type(values["name"]) is not str or values["role"] not in ("bind", "load"):
            raise _graph_error()
        return
    if kind == "attribute_access":
        values = _mapping(value, ("chain", "ctx", "root"))
        if (
            type(values["root"]) is not str
            or type(values["chain"]) is not tuple
            or any(type(item) is not str for item in values["chain"])
            or values["ctx"] not in ("del", "load", "store")
        ):
            raise _graph_error()
        return
    if kind == "static_call":
        values = _mapping(value, ("chain", "root"))
        if type(values["root"]) is not str or type(values["chain"]) is not tuple or any(type(item) is not str for item in values["chain"]):
            raise _graph_error()
        return
    values = _mapping(value, (
        "code_id", "current_thread_only", "depth", "event", "location",
        "python_only", "root", "sequence",
    ))
    if (
        type(values["code_id"]) is not str
        or type(values["depth"]) is not int or values["depth"] < 0
        or values["event"] not in ("call", "exception", "line", "return")
        or type(values["sequence"]) is not int or values["sequence"] < 0
        or any(type(values[name]) is not bool for name in ("current_thread_only", "python_only", "root"))
    ):
        raise _graph_error()
    _mapping(values["location"], ("column", "filename", "line"))


def _node_id(kind: ProgramNodeKind, value: FactValue, source: SourceLocation | None, occurrence: int) -> str:
    """Hash one deterministic node identity without using live Python identity."""

    payload: FactValue = (
        ("kind", kind),
        ("occurrence", occurrence),
        ("source", _source_value(source)),
        ("value", value),
    )
    return hashlib.sha256(_pack(b"dryml.code.node.v1", _encode(payload))).hexdigest()


def _source_sort_key(source: SourceLocation | None) -> tuple[int, str, int, int]:
    """Return a total canonical source-order key for one node location."""

    if source is None:
        return (0, "", 0, 0)
    return (1, source.filename or "", source.line or 0, source.column or 0)


def _node_sort_key(node: "ProgramNode") -> tuple[object, ...]:
    """Order target first and all remaining nodes by source then closed content."""

    if node.kind == "trace_event":
        return (2, dict(node.value)["sequence"])

    return (
        0 if node.kind == "target" else 1,
        _source_sort_key(node.source),
        node.kind,
        _encode(node.value),
        node.id,
    )


@dataclass(frozen=True, slots=True)
class ProgramNode:
    """One immutable typed program-graph node.

    Args:
        id: Canonical domain-tagged SHA-256 node identity.
        kind: Closed program-evidence vocabulary kind.
        value: Closed recursively immutable payload for ``kind``.
        source: Optional sanitized source provenance.

    Returns:
        An immutable node value; graph construction validates all fields.

    Raises:
        CodeAnalysisError: Raised by :class:`ProgramGraph` when this node is not
            a valid graph member.

    Side Effects:
        None.
    """

    id: str
    kind: ProgramNodeKind
    value: FactValue
    source: SourceLocation | None = None


@dataclass(frozen=True, slots=True)
class ProgramEdge:
    """One immutable typed relationship between program-graph node IDs.

    Args:
        source: Canonical source node ID.
        target: Canonical target node ID.
        kind: Closed relationship vocabulary kind.

    Returns:
        An immutable edge value; graph construction validates endpoints and the
        relationship allowed by ``kind``.

    Raises:
        CodeAnalysisError: Raised by :class:`ProgramGraph` for invalid edges.

    Side Effects:
        None.
    """

    source: str
    target: str
    kind: ProgramEdgeKind


@dataclass(frozen=True, slots=True)
class ProgramGraph:
    """An immutable deterministic static or trace-derived program graph.

    Args:
        target: Metadata-only target provenance; no live target handle is kept.
        nodes: Immutable node tuple for the closed graph vocabulary.
        edges: Immutable edge tuple for the closed relationship vocabulary.
        diagnostics: Immutable redacted diagnostics for incomplete evidence.

    Raises:
        CodeAnalysisError: If a member, payload, identity, edge, query, or
            endpoint violates the graph contract.

    Side Effects:
        Input tuples are replaced with deterministic canonical tuple order.
    """

    target: TargetInfo
    nodes: tuple[ProgramNode, ...]
    edges: tuple[ProgramEdge, ...]
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        """Validate graph authority and freeze its deterministic member order."""

        if type(self.target) is not TargetInfo or type(self.nodes) is not tuple or type(self.edges) is not tuple or type(self.diagnostics) is not tuple:
            raise _graph_error()
        if any(type(node) is not ProgramNode for node in self.nodes) or any(type(edge) is not ProgramEdge for edge in self.edges):
            raise _graph_error()
        if any(type(diagnostic) is not Diagnostic or diagnostic.kernel is not None for diagnostic in self.diagnostics):
            raise _graph_error()
        normalized_nodes = tuple(ProgramNode(node.id, node.kind, _canonical_value(node.value), node.source) for node in self.nodes)
        identity_groups: dict[tuple[ProgramNodeKind, bytes, bytes], list[str]] = {}
        for node in normalized_nodes:
            if type(node.id) is not str or len(node.id) != 64 or any(character not in "0123456789abcdef" for character in node.id):
                raise _graph_error()
            if node.kind not in _NODE_KINDS or (node.source is not None and type(node.source) is not SourceLocation):
                raise _graph_error()
            _validate_payload(node.kind, node.value)
            identity_groups.setdefault(
                (node.kind, _encode(node.value), _encode(_source_value(node.source))), []
            ).append(node.id)
        for (kind, _, _), identifiers in identity_groups.items():
            sample = next(
                node
                for node in normalized_nodes
                if node.kind == kind and node.id in identifiers
            )
            expected = {
                _node_id(kind, sample.value, sample.source, occurrence)
                for occurrence in range(len(identifiers))
            }
            if set(identifiers) != expected:
                raise _graph_error()
        canonical_nodes = tuple(sorted(normalized_nodes, key=_node_sort_key))
        if not canonical_nodes or canonical_nodes[0].kind != "target" or sum(node.kind == "target" for node in canonical_nodes) != 1:
            raise _graph_error()
        if canonical_nodes[0].value != _target_value(self.target) or canonical_nodes[0].source is not None:
            raise _graph_error()
        identifiers = tuple(node.id for node in canonical_nodes)
        if len(identifiers) != len(set(identifiers)):
            raise _graph_error()
        by_id = {node.id: node for node in canonical_nodes}
        edge_keys: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            if type(edge.source) is not str or type(edge.target) is not str or edge.kind not in _EDGE_KINDS:
                raise _graph_error()
            if edge.source not in by_id or edge.target not in by_id:
                raise _graph_error()
            key = (edge.source, edge.target, edge.kind)
            if key in edge_keys:
                raise _graph_error()
            edge_keys.add(key)
            if not _edge_is_valid(edge, by_id):
                raise _graph_error()
        index = {node.id: position for position, node in enumerate(canonical_nodes)}
        canonical_edges = tuple(sorted(self.edges, key=lambda edge: (index[edge.source], edge.kind, index[edge.target])))
        diagnostic_order = tuple(sorted(self.diagnostics, key=lambda item: _encode(_diagnostic_value(item))))
        object.__setattr__(self, "nodes", canonical_nodes)
        object.__setattr__(self, "edges", canonical_edges)
        object.__setattr__(self, "diagnostics", diagnostic_order)

    @property
    def digest(self) -> str:
        """Return the canonical domain-tagged SHA-256 identity for this graph.

        Returns:
            A lowercase hexadecimal SHA-256 digest of target, nodes, edges, and
            diagnostics.

        Raises:
            None. Graph construction has already validated all encoded fields.

        Side Effects:
            None.
        """

        payload: FactValue = (
            ("diagnostics", tuple(_diagnostic_value(item) for item in self.diagnostics)),
            ("edges", tuple((("kind", edge.kind), ("source", edge.source), ("target", edge.target)) for edge in self.edges)),
            ("nodes", tuple((("id", node.id), ("kind", node.kind), ("source", _source_value(node.source)), ("value", node.value)) for node in self.nodes)),
            ("target", _target_value(self.target)),
        )
        return hashlib.sha256(_pack(b"dryml.code.graph.v1", _encode(payload))).hexdigest()

    def nodes_of_kind(self, kind: ProgramNodeKind) -> tuple[ProgramNode, ...]:
        """Return canonical nodes of one closed kind.

        Args:
            kind: Requested closed node kind.

        Returns:
            Nodes in graph canonical order.

        Raises:
            CodeAnalysisError: If ``kind`` is outside the closed vocabulary.

        Side Effects:
            None.
        """

        if kind not in _NODE_KINDS:
            raise _graph_error()
        return tuple(node for node in self.nodes if node.kind == kind)

    def edges_of_kind(self, kind: ProgramEdgeKind) -> tuple[ProgramEdge, ...]:
        """Return canonical edges of one closed kind.

        Args:
            kind: Requested closed edge kind.

        Returns:
            Edges in graph canonical order.

        Raises:
            CodeAnalysisError: If ``kind`` is outside the closed vocabulary.

        Side Effects:
            None.
        """

        if kind not in _EDGE_KINDS:
            raise _graph_error()
        return tuple(edge for edge in self.edges if edge.kind == kind)

    def successors(self, node_id: str, *, kind: ProgramEdgeKind | None = None) -> tuple[ProgramNode, ...]:
        """Return deterministic direct successors for one graph node.

        Args:
            node_id: Existing canonical source node ID.
            kind: Optional closed edge-kind filter.

        Returns:
            Target nodes in canonical edge order.

        Raises:
            CodeAnalysisError: If the node ID or optional kind is unknown.

        Side Effects:
            None.
        """

        by_id = self._node_index(node_id, kind)
        return tuple(by_id[edge.target] for edge in self.edges if edge.source == node_id and (kind is None or edge.kind == kind))

    def predecessors(self, node_id: str, *, kind: ProgramEdgeKind | None = None) -> tuple[ProgramNode, ...]:
        """Return deterministic direct predecessors for one graph node.

        Args:
            node_id: Existing canonical target node ID.
            kind: Optional closed edge-kind filter.

        Returns:
            Source nodes in canonical edge order.

        Raises:
            CodeAnalysisError: If the node ID or optional kind is unknown.

        Side Effects:
            None.
        """

        by_id = self._node_index(node_id, kind)
        return tuple(by_id[edge.source] for edge in self.edges if edge.target == node_id and (kind is None or edge.kind == kind))

    def _node_index(self, node_id: str, kind: ProgramEdgeKind | None) -> dict[str, ProgramNode]:
        """Validate a query boundary and return the transient node lookup table."""

        if type(node_id) is not str or (kind is not None and kind not in _EDGE_KINDS):
            raise _graph_error()
        by_id = {node.id: node for node in self.nodes}
        if node_id not in by_id:
            raise _graph_error()
        return by_id


def _diagnostic_value(diagnostic: Diagnostic) -> FactValue:
    """Return the graph-owned canonical projection of one diagnostic."""

    return (
        ("code", diagnostic.code),
        ("message", diagnostic.message),
        ("severity", diagnostic.severity),
        ("source", _source_value(diagnostic.source)),
    )


def _edge_is_valid(edge: ProgramEdge, by_id: dict[str, ProgramNode]) -> bool:
    """Return whether one edge expresses an admitted typed relationship."""

    source = by_id[edge.source].kind
    target = by_id[edge.target].kind
    if edge.kind == "containment":
        return source in ("target", "syntax") and target in ("syntax", "lexical_symbol", "attribute_access", "static_call")
    if edge.kind == "lexical_reference":
        return source == "syntax" and target == "lexical_symbol"
    if edge.kind == "access":
        return source == "syntax" and target == "attribute_access"
    if edge.kind == "call":
        return source == "syntax" and target == "static_call"
    if edge.kind in ("trace_sequence", "frame_descent"):
        return source == "trace_event" and target == "trace_event"
    return source == "trace_event" and target == "syntax"


def _ast_scalar(value: object) -> FactValue:
    """Canonicalize one non-child AST field without using AST presentation text."""

    value_type = type(value)
    if value is Ellipsis:
        return ("ellipsis",)
    if value_type in (type(None), bool, int, str, bytes):
        return value  # type: ignore[return-value]
    if value_type is float:
        if math.isfinite(value):
            return value
        return ("float", "nan" if math.isnan(value) else "inf" if value > 0 else "-inf")
    if value_type is complex:
        return ("complex", _ast_scalar(value.real), _ast_scalar(value.imag))
    raise _graph_error()


def _location(node: ast.AST, filename: str | None, line_offset: int) -> SourceLocation | None:
    """Return absolute sanitized AST coordinates while preserving byte columns."""

    line = getattr(node, "lineno", None)
    column = getattr(node, "col_offset", None)
    if type(line) is not int or line < 1:
        return None
    return SourceLocation(filename, line + line_offset, column if type(column) is int else None)


def _flatten_attribute(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    """Return a static name-rooted attribute path without evaluation."""

    chain: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        chain.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        chain.reverse()
        return current.id, tuple(chain)
    return None


def build_program_graph(target: CodeTargetInput) -> ProgramGraph:
    """Build immutable foundational evidence for one admitted static target.

    Args:
        target: A supported target wrapper or live static target accepted by
            :func:`normalize_target`.

    Returns:
        A target-rooted immutable graph. Unavailable non-required live source
        yields only target evidence and a redacted diagnostic.

    Raises:
        CodeAnalysisError: If graph construction encounters invalid graph data.
        SourceUnavailableError: If explicit source is malformed before graph
            construction. Target normalization errors propagate unchanged.

    Side Effects:
        Reads admitted ordinary source files and may import an explicit
        ``ImportTarget`` module through target normalization; it never executes
        target bodies, compiles source, or retains live target handles.
    """

    normalized = normalize_target(target)
    root_value = _target_value(normalized.info)
    root = ProgramNode(_node_id("target", root_value, None, 0), "target", root_value)
    if normalized.source is None:
        return ProgramGraph(
            normalized.info,
            (root,),
            (),
            (Diagnostic("source.unavailable", "source is unavailable", severity="warning"),),
        )
    source = normalized.source
    tree = parse_source(source)
    line_offset = source.start_line - 1 if source.start_line is not None else 0
    drafts: list[tuple[int, ProgramNodeKind, FactValue, SourceLocation | None]] = [(0, "target", root_value, None)]
    draft_edges: list[tuple[int, int, ProgramEdgeKind]] = []
    syntax_tokens: dict[int, int] = {}
    symbols: dict[tuple[str, str], int] = {}

    def add(kind: ProgramNodeKind, value: FactValue, location: SourceLocation | None) -> int:
        """Register one pending node using deterministic source traversal order."""

        token = len(drafts)
        drafts.append((token, kind, value, location))
        return token

    def visit(node: ast.AST, parent: int) -> int:
        """Collect one syntax node, its evidence, and ordered AST children."""

        location = _location(node, source.filename, line_offset)
        fields: list[tuple[str, str, FactValue]] = []
        for field_name, field_value in ast.iter_fields(node):
            if isinstance(field_value, ast.AST):
                continue
            if type(field_value) is list:
                if any(isinstance(child, ast.AST) for child in field_value):
                    continue
                fields.append(("field", field_name, tuple(_ast_scalar(child) for child in field_value)))
                continue
            fields.append(("field", field_name, _ast_scalar(field_value)))
        value: FactValue = (("fields", tuple(fields)), ("type", type(node).__name__))
        token = add("syntax", value, location)
        syntax_tokens[id(node)] = token
        draft_edges.append((parent, token, "containment"))
        if isinstance(node, ast.Name):
            role = "load" if isinstance(node.ctx, ast.Load) else "bind"
            symbol_key = (node.id, role)
            symbol = symbols.get(symbol_key)
            if symbol is None:
                symbol = add("lexical_symbol", (("name", node.id), ("role", role)), location)
                symbols[symbol_key] = symbol
            draft_edges.append((token, symbol, "lexical_reference"))
        if isinstance(node, ast.Attribute):
            flattened = _flatten_attribute(node)
            if flattened is not None:
                root_name, chain = flattened
                context = "load" if isinstance(node.ctx, ast.Load) else "store" if isinstance(node.ctx, ast.Store) else "del"
                access = add("attribute_access", (("chain", chain), ("ctx", context), ("root", root_name)), location)
                draft_edges.append((token, access, "access"))
        if isinstance(node, ast.Call):
            flattened = _flatten_attribute(node.func)
            if flattened is not None:
                root_name, chain = flattened
                call = add("static_call", (("chain", chain), ("root", root_name)), _location(node.func, source.filename, line_offset))
                draft_edges.append((token, call, "call"))
        for _, field_value in ast.iter_fields(node):
            if isinstance(field_value, ast.AST):
                visit(field_value, token)
            elif type(field_value) is list:
                for child in field_value:
                    if isinstance(child, ast.AST):
                        visit(child, token)
        return token

    visit(tree, 0)
    ordered = sorted(drafts, key=lambda item: (0 if item[1] == "target" else 1, _source_sort_key(item[3]), item[1], _encode(item[2]), item[0]))
    occurrences: dict[tuple[ProgramNodeKind, bytes, bytes], int] = {}
    nodes: list[ProgramNode] = []
    identifiers: dict[int, str] = {}
    for token, kind, value, location in ordered:
        key = (kind, _encode(value), _encode(_source_value(location)))
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        identifier = _node_id(kind, value, location, occurrence)
        identifiers[token] = identifier
        nodes.append(ProgramNode(identifier, kind, value, location))
    edges = tuple(ProgramEdge(identifiers[source_token], identifiers[target_token], kind) for source_token, target_token, kind in draft_edges)
    return ProgramGraph(normalized.info, tuple(nodes), edges)


__all__ = [
    "ProgramEdge",
    "ProgramEdgeKind",
    "ProgramGraph",
    "ProgramNode",
    "ProgramNodeKind",
    "build_program_graph",
]
