"""Internal deterministic codec and identity projection for CDef graphs.

This module is deliberately dependency-light: it never resolves a class or
symbol while inspecting or decoding graph authority. Private process tokens are
replaced by deterministic labels derived from canonical graph paths.
Complete public validation and private exact-matched fresh realization share the
same node builder. Optional owned preparation uses only lossless primitive content;
this module neither owns a cache nor stores tokens, classes or validity verdicts.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from typing import Any

from .bound_args import BoundArguments
from .cdef_identity import cdef_node_key
from .definition import ConcreteDefinition
from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from .links import DefLink
from .quoted import QuotedDef, SelectorSpec
from .reference_values import ObjectRef, StateRef
from .utils.graph.path import GraphPath, graph_path_sort_key
from .utils.graph.value import iter_value_edges

CDEF_GRAPH_CODEC_VERSION = 2
_MAX_QUOTED_DEFINITION_BYTES = 16 * 1024 * 1024


class CDefGraphCodecError(ValueError):
    """Raised when graph authority data is malformed or inconsistent."""


def encode_cdef_graph(root: ConcreteDefinition) -> dict[str, Any]:
    """Encode legacy V2 construction metadata with token-free labels.

    Args:
        root: Root exact CDef to encode.

    Returns:
        A versioned graph record containing labels, structural payloads, and
        identity-neutral stateful-role bits.

    Raises:
        TypeError: If ``root`` is not a concrete definition.
        CDefGraphCodecError: If the graph cannot be represented as current
            acyclic V2 authority, or contains Node facets/retained construction
            evidence. Use ``node_codec`` for the new identity-3 family.

    Side Effects:
        Metadata allocation only; no target resolution, materialization, payload
        IO, publication or conversion of existing source authority.
    """

    if not isinstance(root, ConcreteDefinition):
        raise TypeError(
            f"Expected ConcreteDefinition, got {type(root).__name__}."
        )
    _require_v2_evidence(root)
    graph, labels = _graph_and_labels(root)
    quotation_budget = _QuotationBudget()
    nodes = []
    for node in graph.nodes():
        cdef = node.definition
        nodes.append(
            {
                "label": labels[cdef_node_key(cdef)],
                "cls": cdef.cls,
                "parameters": _encode_value(
                    cdef.parameters, labels, quotation_budget
                ),
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

    Side Effects:
        Allocates fresh graph nodes without resolving classes. Public calls always
        perform complete canonical validation; only private exact-matched owning
        record preparation may use the separate fresh-realization stage.
    """

    payloads = _graph_payloads(data)
    root = _build_graph_nodes(data, payloads)
    normalized = encode_cdef_graph(root)
    if not _codec_data_equal(normalized, data):
        raise CDefGraphCodecError(
            "CDef graph labels or payload ordering are not canonical."
        )
    return root


def _graph_payloads(data):
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
    return payloads


def _build_graph_nodes(data, payloads):
    """Construct fresh nodes, preserving only graph-local label sharing."""
    built: dict[str, ConcreteDefinition] = {}
    active: set[str] = set()
    quotation_budget = _QuotationBudget()

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
            parameters = _decode_value(
                node["parameters"], build, payloads, quotation_budget
            )
            if not isinstance(parameters, FrozenDict):
                raise CDefGraphCodecError(
                    f"CDef graph parameters for {label!r} must be a frozen mapping payload."
                )
            try:
                result = ConcreteDefinition._from_bound_record(
                    node["cls"],
                    BoundArguments(parameters.items()),
                    stateful_role=node["stateful_role"],
                )
            except (TypeError, ValueError) as error:
                raise CDefGraphCodecError(
                    f"CDef graph parameters for {label!r} are invalid."
                ) from error
            built[label] = result
            return result
        finally:
            active.remove(label)

    root = build(data["root"])
    if len(built) != len(payloads):
        raise CDefGraphCodecError(
            "CDef graph contains unreachable node declarations."
        )
    return root


def _realize_prepared_graph(content):
    """Realize owned exact-matched authority already validated by this codec."""
    data = _thaw_graph_content(content)
    return _build_graph_nodes(data, {node["label"]: node for node in data["nodes"]})


_GRAPH_PREPARATION_POLICY = 1


def _eligible_graph_size(data):
    """Reject unsupported graph contexts before optional retention allocation."""
    _check_graph_eligibility(data)
    return _graph_content_size(data)


def _check_graph_eligibility(data):
    """Check the closed retention policy without computing optional owned sizes."""
    from .symbol import ImportRef

    if type(data) is not dict or type(data.get("nodes")) is not list:
        raise TypeError("Unsupported graph representation.")
    def value(payload):
        if type(payload) is not dict:
            raise TypeError("Unsupported tagged value.")
        kind = payload.get("kind")
        if kind == "atom":
            atom = payload.get("value")
            if type(atom) not in (str, bytes, int, float, complex, bool, type(None)):
                _array_content(atom)
        elif kind == "dict":
            for key, item in payload["items"]:
                from .canonical import validate_canonical_key
                if type(key) not in (str, bytes, int, float, bool, type(None), tuple):
                    raise TypeError("Unsupported mapping key.")
                validate_canonical_key(key, where="retained graph key")
                value(item)
        elif kind in ("list", "tuple", "set"):
            for item in payload["items"]:
                value(item)
        elif kind == "link":
            value(payload["target"])
        elif kind != "cdef":
            raise TypeError("Context-dependent graph leaf.")
    for node in data["nodes"]:
        if type(node) is not dict or type(node.get("cls")) is not ImportRef:
            raise TypeError("Only exact ImportRef graph classes are retainable.")
        value(node["parameters"])


def _array_content(value):
    """Return lossless primitive array descriptors without copying its buffer."""
    import numpy as np
    from .freeze import FrozenNDArray

    if type(value) not in (np.ndarray, FrozenNDArray):
        raise TypeError("Unsupported graph atom.")
    dtype = value.dtype
    if (dtype.kind not in "buifc" or dtype.fields is not None
            or dtype.subdtype is not None or dtype.metadata is not None
            or type(dtype) is not type(np.dtype(dtype.str))
            or dtype != np.dtype(dtype.str)):
        raise TypeError("Unsupported array dtype.")
    return (type(value) is FrozenNDArray, dtype.str, tuple(value.shape),
            tuple(value.strides), bool(value.flags.writeable),
            getattr(value, "_frozen_writeable", None))


def _graph_content_size(value):
    """Conservatively charge immutable tagged copies; never canonicalize input."""
    from .symbol import ImportRef
    if type(value) in (dict, list, tuple):
        items = value.items() if type(value) is dict else enumerate(value)
        return 256 + sum(128 + _graph_content_size(key) + _graph_content_size(item)
                         for key, item in items)
    if type(value) is ImportRef:
        if type(value.module) is not str or type(value.qualname) is not str:
            raise TypeError("Unsupported import fields.")
        return 256 + sys.getsizeof(value.module) + sys.getsizeof(value.qualname)
    if type(value) in (str, bytes, int, float, complex, bool, type(None)):
        return 128 + sys.getsizeof(value)
    descriptor = _array_content(value)
    return 512 + _graph_content_size(descriptor) + value.nbytes


def _freeze_graph_content(value):
    """Own token-neutral content with type/order/bit-exact scalar information."""
    from .symbol import ImportRef
    if type(value) is dict:
        return ("dict", tuple((_freeze_graph_content(key), _freeze_graph_content(item))
                              for key, item in value.items()))
    if type(value) in (list, tuple):
        return ("list" if type(value) is list else "tuple",
                tuple(_freeze_graph_content(item) for item in value))
    if type(value) is ImportRef:
        return ("import", value.module, value.qualname)
    if type(value) is float:
        return ("float", struct.pack("!d", value))
    if type(value) is complex:
        return ("complex", struct.pack("!dd", value.real, value.imag))
    if type(value) in (str, bytes, int, bool, type(None)):
        return (type(value).__name__, value)
    return ("array", _array_content(value), value.tobytes(order="C"))


def _graph_content_matches(value, content):
    """Lossless fresh-content tie-break without graph equality or serialization."""
    from .symbol import ImportRef
    tag = content[0]
    if tag == "dict":
        return type(value) is dict and len(value) == len(content[1]) and all(
            _graph_content_matches(key, stored_key) and _graph_content_matches(item, stored_item)
            for (key, item), (stored_key, stored_item) in zip(value.items(), content[1]))
    if tag in ("list", "tuple"):
        return type(value) is (list if tag == "list" else tuple) and len(value) == len(content[1]) and all(
            _graph_content_matches(item, stored) for item, stored in zip(value, content[1]))
    if tag == "import":
        return type(value) is ImportRef and (value.module, value.qualname) == content[1:]
    if tag == "float":
        return type(value) is float and struct.pack("!d", value) == content[1]
    if tag == "complex":
        return type(value) is complex and struct.pack("!dd", value.real, value.imag) == content[1]
    if tag == "array":
        try:
            descriptor = _array_content(value)
        except TypeError:
            return False
        return descriptor == content[1] and value.tobytes(order="C") == content[2]
    return type(value).__name__ == tag and type(value) in (str, bytes, int, bool, type(None)) and value == content[1]


def _thaw_graph_content(content):
    from .symbol import ImportRef
    tag = content[0]
    if tag == "dict":
        return {_thaw_graph_content(key): _thaw_graph_content(item) for key, item in content[1]}
    if tag in ("list", "tuple"):
        values = (_thaw_graph_content(item) for item in content[1])
        return list(values) if tag == "list" else tuple(values)
    if tag == "import":
        return ImportRef(content[1], content[2])
    if tag == "float":
        return struct.unpack("!d", content[1])[0]
    if tag == "complex":
        return complex(*struct.unpack("!dd", content[1]))
    if tag == "array":
        import numpy as np
        from .freeze import FrozenNDArray
        frozen, dtype, shape, strides, writeable, frozen_writeable = content[1]
        dtype = np.dtype(dtype)
        order = "F" if len(shape) > 1 and strides[0] == dtype.itemsize else "C"
        array = np.frombuffer(content[2], dtype=dtype).reshape(shape).copy(order=order)
        if frozen:
            array = array.view(FrozenNDArray)
            if frozen_writeable is not None:
                array._frozen_writeable = frozen_writeable
        array.setflags(write=writeable)
        return array
    return content[1]


def cdef_graph_equal(left: ConcreteDefinition, right: object) -> bool:
    """Compare exact rooted evidence without raw tokens or hash-only authority.

    Args:
        left (ConcreteDefinition): Rooted construction graph.
        right (object): Candidate graph.
    Returns:
        bool: Exact normalized graph equality, including independent occurrences.
            Legacy CDef pairs retain V2 reference/quotation value scopes; Node
            pairs and mixed pairs compare the complete identity-3 closure.
    Raises:
        TypeError, ValueError: For unsupported graph values or cycles.
    Side Effects:
        Pure owned preparation; no class resolution, construction or IO.
    """

    if not isinstance(right, ConcreteDefinition):
        return False
    from .node import Node
    from .node_prepare import NodePreparation
    a, b = NodePreparation((left,)), NodePreparation((right,))
    if not isinstance(left, Node) and not isinstance(right, Node):
        return _scoped_cdef_equal(a, left, b, right)
    return a.exact_equal(left, b, right)


def _scoped_cdef_equal(left, left_root, right, right_root):
    """Compare V2 value scopes using full normalized evidence, never old hashes.

    V2 transports references and quotations independently, so aliases crossing
    those boundaries cannot be authority. Each scope still checks a complete
    occurrence bijection, typed fields, bindings and any embedded Node facets.
    Backtracking carries unordered choices through later ordered aliases.
    """
    scopes = ("object-ref", "object-selector", "quote")

    def sequence(pairs, bindings):
        if not pairs:
            yield bindings
            return
        first, *rest = pairs
        for updated in match(*first, bindings):
            yield from sequence(rest, updated)

    def unordered(items, candidates, bindings):
        if not items:
            yield bindings
            return
        for index, candidate in enumerate(candidates):
            for updated in match(items[0], candidate, bindings):
                yield from unordered(items[1:], candidates[:index] + candidates[index + 1:], updated)

    def match(a, b, bindings, *, row=False, scoped=True):
        if type(a) is not type(b):
            return
        if isinstance(a, dict):
            tag = a.get("tag")
            if tag != b.get("tag"):
                return
            if scoped and tag in scopes:
                if next(match(a, b, ({}, {}), scoped=False), None) is not None:
                    yield bindings
                return
            if tag == "node":
                source, target = a["slot"], b["slot"]
                forward, reverse = bindings
                if source in forward or target in reverse:
                    if forward.get(source) == target and reverse.get(target) == source:
                        yield bindings
                    return
                updated = ({**forward, source: target}, {**reverse, target: source})
                yield from match(left._rows[source], right._rows[target], updated, row=True)
                return
            ignored = ({"stateful_role"} if row else set()) | ({"finalized"} if tag == "link" else set())
            if tag in ("set", "frozenset"):
                if len(a["items"]) != len(b["items"]):
                    return
                ignored.add("items")
            keys = sorted(set(a) - ignored)
            if keys != sorted(set(b) - ignored):
                return
            for updated in sequence([(a[key], b[key]) for key in keys], bindings):
                if tag in ("set", "frozenset"):
                    yield from unordered(a["items"], b["items"], updated)
                else:
                    yield updated
        elif isinstance(a, (list, tuple)):
            if len(a) == len(b):
                yield from sequence(list(zip(a, b)), bindings)
        elif a == b:
            yield bindings

    return next(match({"tag": "node", "slot": left._slot(left_root)},
                      {"tag": "node", "slot": right._slot(right_root)}, ({}, {})), None) is not None


def _v2_cdef_graph_equal(left, right):
    """Compare V2 boundary evidence without requiring cross-value Node aliases.

    Both graphs must be V2-representable; facets and retained construction are
    rejected rather than discarded. Within each transported value scope, exact
    topology, roles, configuration and reference bindings remain authoritative.
    """
    _require_v2_evidence(left)
    _require_v2_evidence(right)
    return _reference_cdef_graph_equal(left, right)


def _reference_cdef_graph_equal(left, right):
    """Compare standalone reference values in their transported topology scopes."""
    from .node_prepare import NodePreparation

    a, b = NodePreparation((left,)), NodePreparation((right,))
    return _scoped_cdef_equal(a, left, b, right)


def cdef_graph_hash(root: ConcreteDefinition) -> str:
    """Return the legacy V2 graph digest for existing consumers awaiting cutover.

    Args:
        root (ConcreteDefinition): A V2-representable construction graph.
    Returns:
        str: Its unchanged legacy digest, not an identity-3 Node exact key.
    Raises:
        CDefGraphCodecError: If new Node facets/construction evidence would be lost.
    Side Effects:
        Metadata inspection only. Use Node.exact_digest for the new key domain.
    """

    projection = _topology_projection(root)
    payload = json.dumps(
        projection, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(b"dryml-cdef-graph-v1\x00" + payload).hexdigest()


def copy_cdef_graph(root: ConcreteDefinition) -> ConcreteDefinition:
    """Return a graph-isomorphic CDef copy with regenerated private tokens."""

    return decode_cdef_graph(encode_cdef_graph(root))


def object_projection_cdef(
        root: ConcreteDefinition, *, traverse_refs: bool = False) -> ConcreteDefinition:
    """Recursively replace exact references in a CDef graph with ObjectSelectors.

    Args:
        root: Exact CDef graph to rewrite without resolving its class authority.
        traverse_refs: Whether to enter Ref-held quoted Definition data. Ordinary
            Ref-held exact references are always rewritten.

    Returns:
        A graph-isomorphic query CDef whose supported exact references are
        ObjectSelectors. Raw references subsequently added by a caller remain exact.

    Raises:
        TypeError: If ``root`` is not a CDef or ``traverse_refs`` is not bool.
        ValueError: If an unsupported cyclic construction graph is encountered.
        CDefGraphCodecError: If identity-3 facets/construction would be lost at
            this legacy query boundary. Use Node.object_view for those graphs.

    Side Effects:
        None. The projection does not resolve imports, materialize objects, read
        Stores, or allocate ObjectIds.
    """

    return _object_projection_cdef(
        root, traverse_refs=traverse_refs, selector_references=True,
    )


def _object_reference_projection_cdef(
        root: ConcreteDefinition, *, traverse_refs: bool = False) -> ConcreteDefinition:
    """Return the authoritative ObjectRef-compatible non-state CDef projection."""

    return _object_projection_cdef(
        root, traverse_refs=traverse_refs, selector_references=False,
    )


def _object_projection_cdef(
        root: ConcreteDefinition, *, traverse_refs: bool,
        selector_references: bool) -> ConcreteDefinition:
    if not isinstance(root, ConcreteDefinition):
        raise TypeError(f"Expected ConcreteDefinition, got {type(root).__name__}.")
    if type(traverse_refs) is not bool:
        raise TypeError("traverse_refs must be a bool.")
    _require_v2_evidence(root)

    from .cdef_graph import EdgeKind
    from .definition import Definition
    from .factory import FactorySpec
    from .template import _BinaryExpr, _RepeatExpr

    memo: dict[int, Any] = {}
    active: set[int] = set()

    def rewrite_reference(ref: ObjectRef):
        marker = id(ref)
        if marker in memo:
            return memo[marker]
        if marker in active:
            raise ValueError("Object projection does not support cyclic reference graphs.")
        active.add(marker)
        try:
            definition = rewrite(ref.definition)
            if selector_references:
                from .selector import ObjectSelector

                reference = ObjectRef(
                    _object_reference_projection_cdef(
                        ref.definition, traverse_refs=traverse_refs,
                    ),
                    ref.objects,
                )
                result = ObjectSelector._projected(
                    definition, ref.objects, reference,
                )
            else:
                result = ObjectRef(definition, ref.objects)
            memo[marker] = result
            return result
        finally:
            active.remove(marker)

    def rewrite(current: Any) -> Any:
        if isinstance(current, StateRef):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            result = rewrite_reference(current.object)
            memo[marker] = result
            return result
        if isinstance(current, ObjectRef):
            return rewrite_reference(current)
        if isinstance(current, DefLink):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            if (
                traverse_refs
                and current.kind is EdgeKind.REF
                and isinstance(current.target, QuotedDef)
            ):
                value = rewrite(current.target.value)
                target = (
                    current.target
                    if value is current.target.value
                    else QuotedDef(value)
                )
            else:
                target = rewrite(current.target)
            result = current if target is current.target else (
                DefLink.finalized(current.kind, target)
                if current.is_finalized
                else DefLink.assertion(current.kind, target)
            )
            memo[marker] = result
            return result
        if isinstance(current, ConcreteDefinition):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            if marker in active:
                raise ValueError("Object projection does not support cyclic CDef graphs.")
            active.add(marker)
            try:
                parameters = FrozenDict(
                    (name, rewrite(value))
                    for name, value in current.parameters.items()
                )
                result = current if all(
                    parameters[name] is value
                    for name, value in current.parameters.items()
                ) else ConcreteDefinition._from_bound_record(
                    current.cls,
                    BoundArguments(parameters.items()),
                    stateful_role=current._stateful_role,
                )
                memo[marker] = result
                return result
            finally:
                active.remove(marker)
        if isinstance(current, Definition):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            args = (
                None if current.args is None
                else FrozenTuple(rewrite(value) for value in current.args)
            )
            kwargs = FrozenDict(
                (name, rewrite(value)) for name, value in current.kwargs.items()
            )
            result = current if (
                (args is None and current.args is None)
                or (args is not None and all(
                    new is old for new, old in zip(args, current.args)
                ))
            ) and all(kwargs[name] is value for name, value in current.kwargs.items()) else (
                Definition._from_symbolic_parts(current.cls, args, kwargs)
            )
            memo[marker] = result
            return result
        if isinstance(current, FactorySpec):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            args = FrozenTuple(rewrite(value) for value in current.args)
            kwargs = FrozenDict(
                (name, rewrite(value)) for name, value in current.kwargs.items()
            )
            result = current if all(
                new is old for new, old in zip(args, current.args)
            ) and all(kwargs[name] is value for name, value in current.kwargs.items()) else (
                FactorySpec._from_symbolic_parts(current.target, args, kwargs)
            )
            memo[marker] = result
            return result
        if isinstance(current, _BinaryExpr):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            left, right = rewrite(current.left), rewrite(current.right)
            result = current if left is current.left and right is current.right else (
                _BinaryExpr(current.operation, left, right)
            )
            memo[marker] = result
            return result
        if isinstance(current, _RepeatExpr):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            group = tuple(rewrite(value) for value in current.group)
            count = rewrite(current.count)
            result = current if all(
                new is old for new, old in zip(group, current.group)
            ) and count is current.count else _RepeatExpr(
                list(group) if isinstance(current.group, FrozenList) else group,
                count,
                current.shared,
            )
            memo[marker] = result
            return result
        if isinstance(current, (FrozenDict, dict)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            items = [(key, rewrite(value)) for key, value in current.items()]
            result = current if all(
                value is current[key] for key, value in items
            ) else (FrozenDict(items) if isinstance(current, FrozenDict) else dict(items))
            memo[marker] = result
            return result
        if isinstance(current, (FrozenList, list)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            values = tuple(rewrite(value) for value in current)
            result = current if all(
                new is old for new, old in zip(values, current)
            ) else FrozenList(values)
            memo[marker] = result
            return result
        if isinstance(current, (FrozenTuple, tuple)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            values = tuple(rewrite(value) for value in current)
            result = current if all(
                new is old for new, old in zip(values, current)
            ) else FrozenTuple(values)
            memo[marker] = result
            return result
        if isinstance(current, (FrozenSet, frozenset, set)):
            marker = id(current)
            if marker in memo:
                return memo[marker]
            values = tuple(rewrite(value) for value in current)
            result = current if all(
                new is old for new, old in zip(values, current)
            ) else FrozenSet(values)
            memo[marker] = result
            return result
        return current

    return rewrite(root)


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
    _require_v2_evidence(root)
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


def _require_v2_evidence(root):
    """Refuse lossy old-format exports before hashing or serializing new facets."""
    from collections.abc import Mapping
    from .definition import Definition
    from .node import Node
    from .factory import FactorySpec
    from .config import ConfigRef
    from .template import _BinaryExpr, _RepeatExpr

    seen = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, Node) and (value.object_id is not None or value.state is not None or value._construction is not None):
            raise CDefGraphCodecError("Node facets and retained construction require the identity-3 Node codec.")
        if isinstance(value, ConcreteDefinition):
            visit(value.parameters)
        elif isinstance(value, Definition):
            visit(value.args)
            visit(value.kwargs)
        elif isinstance(value, DefLink):
            visit(value.target)
        elif isinstance(value, QuotedDef):
            visit(value.value)
        elif isinstance(value, SelectorSpec):
            visit(value.selector.root)
        elif isinstance(value, (ObjectRef, StateRef)):
            visit(value.definition)
        elif isinstance(value, FactorySpec):
            visit(value.args)
            visit(value.kwargs)
        elif isinstance(value, ConfigRef) and value.has_default:
            visit(value.default)
        elif isinstance(value, _BinaryExpr):
            visit(value.left)
            visit(value.right)
        elif isinstance(value, _RepeatExpr):
            visit(value.group)
            visit(value.count)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for child in value:
                visit(child)
    visit(root)


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


class _QuotationBudget:
    """Bound cumulative quoted-expression bytes retained by one CDef record."""

    def __init__(self) -> None:
        self.bytes = 0

    def consume(self, payload: Any) -> None:
        try:
            size = len(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            )
        except (TypeError, ValueError, UnicodeError) as error:
            raise CDefGraphCodecError(
                "CDef quoted Definition payload is invalid."
            ) from error
        self.bytes += size
        if self.bytes > _MAX_QUOTED_DEFINITION_BYTES:
            raise CDefGraphCodecError(
                "CDef aggregate quoted Definition payload limit exceeded."
            )


def _encode_value(
    value: Any,
    labels: dict[object, str],
    quotation_budget: _QuotationBudget,
) -> dict[str, Any]:
    from .selector import ObjectSelector

    if isinstance(value, ObjectSelector):
        raise CDefGraphCodecError(
            "ObjectSelector values are query-only and cannot enter CDef authority."
        )
    if isinstance(value, ObjectRef):
        return {"kind": "object_ref", "value": value.to_data()}
    if isinstance(value, StateRef):
        return {"kind": "state_ref", "value": value.to_data()}
    if isinstance(value, ConcreteDefinition):
        return {"kind": "cdef", "label": labels[cdef_node_key(value)]}
    if isinstance(value, QuotedDef):
        from .definition_expression_codec import to_data

        payload = to_data(value.value)
        quotation_budget.consume(payload)
        return {"kind": "quoted-definition", "value": payload}
    if isinstance(value, DefLink):
        if not value.is_finalized:
            raise CDefGraphCodecError("Unresolved DefLink assertions cannot be encoded.")
        if isinstance(value.target, QuotedDef):
            from .definition import Definition

            if not isinstance(value.target.value, Definition):
                raise CDefGraphCodecError(
                    "CDef quoted Definition link must contain a Definition."
                )
        return {
            "kind": "link",
            "edge_kind": value.kind.value,
            "target": _encode_value(value.target, labels, quotation_budget),
        }
    if isinstance(value, FrozenDict):
        return {
            "kind": "dict",
            "items": [
                [
                    edge.segment.key,
                    _encode_value(edge.value, labels, quotation_budget),
                ]
                for edge in iter_value_edges(value)
            ],
        }
    if isinstance(value, FrozenList):
        return {
            "kind": "list",
            "items": [
                _encode_value(item, labels, quotation_budget) for item in value
            ],
        }
    if isinstance(value, FrozenTuple):
        return {
            "kind": "tuple",
            "items": [
                _encode_value(item, labels, quotation_budget) for item in value
            ],
        }
    if isinstance(value, FrozenSet):
        return {
            "kind": "set",
            "items": [
                _encode_value(edge.value, labels, quotation_budget)
                for edge in iter_value_edges(value)
            ],
        }
    return {"kind": "atom", "value": value}


def _decode_value(
    data: Any,
    build,
    payloads: dict[str, dict[str, Any]],
    quotation_budget: _QuotationBudget,
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
    if kind == "quoted-definition":
        _require_exact_keys(data, {"kind", "value"}, "QuotedDef")
        from .definition_expression_codec import from_data

        try:
            quotation_budget.consume(data["value"])
            return QuotedDef(from_data(data["value"]))
        except CDefGraphCodecError:
            raise
        except Exception as error:
            raise CDefGraphCodecError("CDef quoted Definition payload is invalid.") from error
    if kind == "link":
        _require_exact_keys(data, {"kind", "edge_kind", "target"}, "CDef link")
        from .cdef_graph import EdgeKind

        try:
            edge_kind = EdgeKind(data["edge_kind"])
        except (TypeError, ValueError) as error:
            raise CDefGraphCodecError(
                f"Invalid CDef link kind {data['edge_kind']!r}."
            ) from error
        target = _decode_value(
            data["target"], build, payloads, quotation_budget
        )
        if not isinstance(target, (
            ConcreteDefinition, ObjectRef, StateRef, QuotedDef, SelectorSpec,
        )):
            raise CDefGraphCodecError(
                "CDef link target must be a CDef, ObjectRef, StateRef, or "
                "exact constructor-data quotation."
            )
        if isinstance(target, (QuotedDef, SelectorSpec)) and edge_kind is not EdgeKind.REF:
            raise CDefGraphCodecError(
                "CDef constructor-data quotation links must use a Ref edge."
            )
        if isinstance(target, QuotedDef):
            from .definition import Definition

            if not isinstance(target.value, Definition):
                raise CDefGraphCodecError(
                    "CDef quoted Definition link must contain a Definition."
                )
        return DefLink.finalized(edge_kind, target)
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
            items.append(
                (
                    item[0],
                    _decode_value(
                        item[1], build, payloads, quotation_budget
                    ),
                )
            )
        try:
            return FrozenDict(dict(items))
        except Exception as error:
            raise CDefGraphCodecError("Invalid CDef dict payload.") from error
    if kind in {"list", "tuple", "set"}:
        _require_exact_keys(data, {"kind", "items"}, f"CDef {kind}")
        if not isinstance(data["items"], list):
            raise CDefGraphCodecError(f"CDef {kind} items must be a list.")
        items = [
            _decode_value(item, build, payloads, quotation_budget)
            for item in data["items"]
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
