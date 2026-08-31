"""Internal deterministic codec and identity projection for CDef graphs.

This module is deliberately dependency-light: it never resolves a class or
symbol while inspecting or decoding graph authority. Private process tokens are
replaced by deterministic labels derived from canonical graph paths.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .bound_args import BoundArguments
from .cdef_identity import cdef_node_key
from .definition import ConcreteDefinition
from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from .links import DefLink
from .reference_values import ObjectRef, StateRef
from .utils.graph.path import GraphPath, graph_path_sort_key
from .utils.graph.value import iter_value_edges

CDEF_GRAPH_CODEC_VERSION = 2


class CDefGraphCodecError(ValueError):
    """Raised when graph authority data is malformed or inconsistent."""


def encode_cdef_graph(root: ConcreteDefinition) -> dict[str, Any]:
    """Encode a rooted CDef graph with deterministic, token-free labels.

    Args:
        root: Root exact CDef to encode.

    Returns:
        A versioned graph record containing labels, structural payloads, and
        identity-neutral stateful-role bits.

    Raises:
        TypeError: If ``root`` is not a concrete definition.
        CDefGraphCodecError: If the graph cannot be represented as current
            acyclic CDef authority.
    """

    if not isinstance(root, ConcreteDefinition):
        raise TypeError(
            f"Expected ConcreteDefinition, got {type(root).__name__}."
        )
    graph, labels = _graph_and_labels(root)
    nodes = []
    for node in graph.nodes():
        cdef = node.definition
        nodes.append(
            {
                "label": labels[cdef_node_key(cdef)],
                "cls": cdef.cls,
                "parameters": _encode_value(cdef.parameters, labels),
                "stateful_role": cdef._stateful_role,
            }
        )
    return {
        "codec_version": CDEF_GRAPH_CODEC_VERSION,
        "root": labels[cdef_node_key(root)],
        "nodes": nodes,
    }


def decode_cdef_graph(data: Any) -> ConcreteDefinition:
    """Decode current graph authority and regenerate all private node tokens.

    Args:
        data: A versioned mapping produced by :func:`encode_cdef_graph`.

    Returns:
        A graph-isomorphic CDef root with fresh private tokens.

    Raises:
        CDefGraphCodecError: If labels, references, payloads, roles, or the
            normalized deterministic record are malformed.
    """

    if not isinstance(data, dict):
        raise CDefGraphCodecError("CDef graph record must be a mapping.")
    _require_exact_keys(
        data, {"codec_version", "root", "nodes"}, "graph record"
    )
    if data["codec_version"] != CDEF_GRAPH_CODEC_VERSION:
        raise CDefGraphCodecError(
            f"Unsupported CDef graph codec version {data['codec_version']!r}."
        )
    if not isinstance(data["root"], str) or not isinstance(
        data["nodes"], list
    ):
        raise CDefGraphCodecError(
            "CDef graph root must be a label and nodes must be a list."
        )
    payloads: dict[str, dict[str, Any]] = {}
    for node in data["nodes"]:
        if not isinstance(node, dict):
            raise CDefGraphCodecError("CDef graph nodes must be mappings.")
        _require_exact_keys(
            node, {"label", "cls", "parameters", "stateful_role"}, "graph node"
        )
        label = node["label"]
        if (
            not isinstance(label, str)
            or not label.startswith("n")
            or not label[1:].isdigit()
        ):
            raise CDefGraphCodecError(f"Malformed CDef graph label {label!r}.")
        if label in payloads:
            raise CDefGraphCodecError(f"Duplicate CDef graph label {label!r}.")
        if type(node["stateful_role"]) is not bool:
            raise CDefGraphCodecError(
                f"CDef graph role for {label!r} must be bool."
            )
        payloads[label] = node
    if data["root"] not in payloads:
        raise CDefGraphCodecError(
            f"CDef graph root label {data['root']!r} is undeclared."
        )
    built: dict[str, ConcreteDefinition] = {}
    active: set[str] = set()

    def build(label: str) -> ConcreteDefinition:
        if label in built:
            return built[label]
        if label in active:
            raise CDefGraphCodecError(
                f"CDef graph contains a cyclic label reference at {label!r}."
            )
        active.add(label)
        try:
            node = payloads[label]
            parameters = _decode_value(node["parameters"], build, payloads)
            if not isinstance(parameters, FrozenDict):
                raise CDefGraphCodecError(
                    f"CDef graph parameters for {label!r} must be a frozen mapping payload."
                )
            result = ConcreteDefinition._from_bound_record(
                node["cls"],
                BoundArguments(parameters.items()),
                stateful_role=node["stateful_role"],
            )
            built[label] = result
            return result
        finally:
            active.remove(label)

    root = build(data["root"])
    if len(built) != len(payloads):
        raise CDefGraphCodecError(
            "CDef graph contains unreachable node declarations."
        )
    normalized = encode_cdef_graph(root)
    if not _codec_data_equal(normalized, data):
        raise CDefGraphCodecError(
            "CDef graph labels or payload ordering are not canonical."
        )
    return root


def cdef_graph_equal(left: ConcreteDefinition, right: object) -> bool:
    """Compare rooted CDef topology without comparing raw private tokens."""

    if not isinstance(right, ConcreteDefinition):
        return False
    return _topology_projection(left) == _topology_projection(right)


def cdef_graph_hash(root: ConcreteDefinition) -> str:
    """Return a deterministic hash of CDef topology and structural node data."""

    projection = _topology_projection(root)
    payload = json.dumps(
        projection, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(b"dryml-cdef-graph-v1\x00" + payload).hexdigest()


def copy_cdef_graph(root: ConcreteDefinition) -> ConcreteDefinition:
    """Return a graph-isomorphic CDef copy with regenerated private tokens."""

    return decode_cdef_graph(encode_cdef_graph(root))


def validate_cdef_stateful_role(cdef: ConcreteDefinition, cls: type) -> None:
    """Validate recorded role authority against an already-resolved class.

    Args:
        cdef: CDef whose graph authority recorded a stateful role.
        cls: The current class resolved by an admitted live-operation boundary.

    Raises:
        CDefGraphCodecError: If the target is not a class or its current
            ``Serializable`` role differs from the recorded authority.
    """

    from .object import Serializable

    if not isinstance(cls, type):
        raise CDefGraphCodecError(
            f"CDef class target must resolve to a class, got {type(cls).__name__}."
        )
    actual = issubclass(cls, Serializable)
    if actual != cdef._stateful_role:
        raise CDefGraphCodecError(
            f"CDef stateful role mismatch: authority recorded {cdef._stateful_role!r}, "
            f"but {cls.__module__}.{cls.__qualname__} resolves to {actual!r}."
        )


def render_cdef_repr(root: ConcreteDefinition) -> str:
    """Render a deterministic rooted repr without exposing private tokens."""

    graph, labels = _graph_and_labels(root)
    by_hash: dict[str, list[ConcreteDefinition]] = {}
    incoming: dict[object, int] = {}
    for node in graph.nodes():
        by_hash.setdefault(node.stable_hash, []).append(node.definition)
        incoming[cdef_node_key(node.definition)] = 0
    for edge in graph.edges():
        incoming[cdef_node_key(edge.child)] += 1
    moniker_nodes = {
        cdef_node_key(node.definition)
        for node in graph.nodes()
        if incoming[cdef_node_key(node.definition)] > 1
        or len(by_hash[node.stable_hash]) > 1
    }

    declared_monikers: set[object] = set()

    def render_value(value: Any, *, root_node: bool = False) -> str:
        if isinstance(value, ConcreteDefinition):
            key = cdef_node_key(value)
            if not root_node and key in moniker_nodes:
                moniker = f"@{labels[key]}"
                if key in declared_monikers:
                    return moniker
                declared_monikers.add(key)
                return f"{moniker}={render_node(value)}"
            return render_node(value)
        if isinstance(value, DefLink):
            target = render_value(value.target)
            return f"{value.kind.value}({target})"
        if isinstance(value, FrozenDict):
            return (
                "{"
                + ", ".join(
                    f"{key!r}: {render_value(item)}"
                    for key, item in value.items()
                )
                + "}"
            )
        if isinstance(value, (FrozenList, list)):
            return "[" + ", ".join(render_value(item) for item in value) + "]"
        if isinstance(value, (FrozenTuple, tuple)):
            body = ", ".join(render_value(item) for item in value)
            return f"({body}{',' if len(value) == 1 else ''})"
        if isinstance(value, (FrozenSet, set, frozenset)):
            return (
                "{"
                + ", ".join(sorted((render_value(item) for item in value)))
                + "}"
            )
        return repr(value)

    def render_node(cdef: ConcreteDefinition) -> str:
        fields = ", ".join(
            f"{name}={render_value(value)}"
            for name, value in cdef.parameters.items()
        )
        return (
            f"ConcreteDefinition({cdef.cls!r}{', ' if fields else ''}{fields})"
        )

    return render_node(root)


def _topology_projection(root: ConcreteDefinition) -> dict[str, Any]:
    graph, labels = _graph_and_labels(root)
    return {
        "root": labels[cdef_node_key(root)],
        "nodes": [
            {
                "label": labels[cdef_node_key(node.definition)],
                "stable_hash": node.stable_hash,
                "edges": [
                    (
                        edge.kind.value,
                        edge.path.to_bytes().hex(),
                        labels[cdef_node_key(edge.child)],
                    )
                    for edge in graph.outgoing(node.definition)
                ],
            }
            for node in graph.nodes()
        ],
    }


def _graph_and_labels(root: ConcreteDefinition):
    from .cdef_graph import ConcreteDefinitionGraph

    graph = ConcreteDefinitionGraph.from_root(root, expand_ref_targets=True)
    minimum_paths: dict[object, GraphPath] = {cdef_node_key(root): GraphPath()}
    stack = [(root, GraphPath())]
    while stack:
        parent, parent_path = stack.pop()
        for edge in reversed(graph.outgoing(parent)):
            path = parent_path.join(edge.path)
            key = cdef_node_key(edge.child)
            previous = minimum_paths.get(key)
            if previous is None or graph_path_sort_key(
                path
            ) < graph_path_sort_key(previous):
                minimum_paths[key] = path
                stack.append((edge.child, path))
    if len(minimum_paths) != len(graph.nodes()):
        raise CDefGraphCodecError(
            "CDef graph contains a node with no deterministic path."
        )
    ordered = sorted(
        graph.nodes(),
        key=lambda node: (
            graph_path_sort_key(minimum_paths[cdef_node_key(node.definition)]),
            node.stable_hash,
        ),
    )
    labels = {
        cdef_node_key(node.definition): f"n{index}"
        for index, node in enumerate(ordered)
    }
    return graph, labels


def _encode_value(value: Any, labels: dict[object, str]) -> dict[str, Any]:
    if isinstance(value, ObjectRef):
        return {"kind": "object_ref", "value": value.to_data()}
    if isinstance(value, StateRef):
        return {"kind": "state_ref", "value": value.to_data()}
    if isinstance(value, ConcreteDefinition):
        return {"kind": "cdef", "label": labels[cdef_node_key(value)]}
    if isinstance(value, DefLink):
        return {
            "kind": "link",
            "edge_kind": value.kind.value,
            "target": _encode_value(value.target, labels),
        }
    if isinstance(value, FrozenDict):
        return {
            "kind": "dict",
            "items": [
                [edge.segment.key, _encode_value(edge.value, labels)]
                for edge in iter_value_edges(value)
            ],
        }
    if isinstance(value, FrozenList):
        return {
            "kind": "list",
            "items": [_encode_value(item, labels) for item in value],
        }
    if isinstance(value, FrozenTuple):
        return {
            "kind": "tuple",
            "items": [_encode_value(item, labels) for item in value],
        }
    if isinstance(value, FrozenSet):
        return {
            "kind": "set",
            "items": [
                _encode_value(edge.value, labels)
                for edge in iter_value_edges(value)
            ],
        }
    return {"kind": "atom", "value": value}


def _decode_value(
    data: Any, build, payloads: dict[str, dict[str, Any]]
) -> Any:
    if not isinstance(data, dict) or "kind" not in data:
        raise CDefGraphCodecError(
            "CDef graph value payload must be a tagged mapping."
        )
    kind = data["kind"]
    if kind == "object_ref":
        _require_exact_keys(data, {"kind", "value"}, "ObjectRef")
        return ObjectRef.from_data(data["value"])
    if kind == "state_ref":
        _require_exact_keys(data, {"kind", "value"}, "StateRef")
        return StateRef.from_data(data["value"])
    if kind == "cdef":
        _require_exact_keys(data, {"kind", "label"}, "CDef reference")
        label = data["label"]
        if not isinstance(label, str) or label not in payloads:
            raise CDefGraphCodecError(
                f"CDef graph reference {label!r} is undeclared."
            )
        return build(label)
    if kind == "link":
        _require_exact_keys(data, {"kind", "edge_kind", "target"}, "CDef link")
        from .cdef_graph import EdgeKind

        try:
            edge_kind = EdgeKind(data["edge_kind"])
        except (TypeError, ValueError) as error:
            raise CDefGraphCodecError(
                f"Invalid CDef link kind {data['edge_kind']!r}."
            ) from error
        target = _decode_value(data["target"], build, payloads)
        if not isinstance(target, ConcreteDefinition):
            raise CDefGraphCodecError(
                "CDef link target must be a CDef reference."
            )
        return DefLink(edge_kind, target)
    if kind == "dict":
        _require_exact_keys(data, {"kind", "items"}, "CDef dict")
        if not isinstance(data["items"], list):
            raise CDefGraphCodecError("CDef dict items must be a list.")
        items = []
        seen_keys = set()
        for item in data["items"]:
            if not isinstance(item, list) or len(item) != 2:
                raise CDefGraphCodecError(
                    "CDef dict item must be a key/value pair."
                )
            from .canonical import validate_canonical_key

            try:
                validate_canonical_key(item[0], where="CDef graph mapping key")
            except TypeError as error:
                raise CDefGraphCodecError(str(error)) from error
            if item[0] in seen_keys:
                raise CDefGraphCodecError(
                    f"Duplicate CDef graph mapping key {item[0]!r}."
                )
            seen_keys.add(item[0])
            items.append((item[0], _decode_value(item[1], build, payloads)))
        try:
            return FrozenDict(dict(items))
        except Exception as error:
            raise CDefGraphCodecError("Invalid CDef dict payload.") from error
    if kind in {"list", "tuple", "set"}:
        _require_exact_keys(data, {"kind", "items"}, f"CDef {kind}")
        if not isinstance(data["items"], list):
            raise CDefGraphCodecError(f"CDef {kind} items must be a list.")
        items = [
            _decode_value(item, build, payloads) for item in data["items"]
        ]
        if kind == "list":
            return FrozenList(items)
        if kind == "tuple":
            return FrozenTuple(items)
        try:
            return FrozenSet(items)
        except TypeError as error:
            raise CDefGraphCodecError(
                "CDef set payload contains an unhashable item."
            ) from error
    if kind == "atom":
        _require_exact_keys(data, {"kind", "value"}, "CDef atom")
        return data["value"]
    raise CDefGraphCodecError(f"Unknown CDef graph value kind {kind!r}.")


def _require_exact_keys(
    data: dict[str, Any], expected: set[str], what: str
) -> None:
    if set(data) != expected:
        raise CDefGraphCodecError(
            f"{what} fields must be exactly {sorted(expected)!r}."
        )


def _codec_data_equal(left: Any, right: Any) -> bool:
    """Compare codec records while preserving canonical array semantics."""

    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _codec_data_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _codec_data_equal(a, b) for a, b in zip(left, right)
        )
    from .definition import _structural_value_equal

    return _structural_value_equal(left, right)
