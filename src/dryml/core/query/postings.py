"""Certified, primitive query graphs and exact ordered-pattern interpretation.

These derived records are built only from validated immutable Store metadata.
They retain collision-independent identity evidence, alias partitions, and typed
relationship multiplicity. Readers must authenticate their source bytes and the
current Store/index cut; this module provides neither membership nor freshness.
Unsupported patterns use the ordinary Node residual instead of approximating it.
"""

from dataclasses import dataclass
import base64
import json
import zlib


_GRAPH_VERSION = 1
_MAX_GRAPH_BYTES = 64 * 1024 * 1024


class _ResidualRequired(Exception):
    """The compact interpreter cannot establish this predicate exactly."""


def _dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _nested_tokens(value, path):
    """Yield necessary nested call/shape tokens, without following Node edges.

Paths are relative to one named constructor parameter. Unsupported values and
matchers add no constraint. Mapping cardinality and absence are deliberately
left to exact interpretation; supplied mapping patterns can be partial.
"""
    from collections.abc import Mapping
    from ..canonical import container_family
    from ..factory import FactorySpec
    from ..node import Node
    from ..links import DefLink
    from ..params import Match
    from ..node_codec import _value

    if isinstance(value, Match):
        return
    if isinstance(value, Node):
        return
    if isinstance(value, DefLink):
        if value.kind.value == "materialize":
            yield from _nested_tokens(value.target, path)
        return
    if type(value) in (str, int, bool, type(None)):
        yield ["value", path, type(value).__name__, value]
    elif isinstance(value, FactorySpec):
        yield ["factory", path, _value(value.target, lambda _: None)]
        yield from _nested_tokens(value.args, path + [["factory", "args"]])
        yield from _nested_tokens(value.kwargs, path + [["factory", "kwargs"]])
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) in (str, int):
                yield from _nested_tokens(child, path + [["key", type(key).__name__, key]])
    elif container_family(value) in ("list", "tuple"):
        yield ["sequence", path, container_family(value), len(value)]
        for index, child in enumerate(value):
            yield from _nested_tokens(child, path + [["index", index]])


def _build_graph(preparation):
    """Build bounded, compressed semantic evidence from a shared decoded owner.

Returns a JSON-compatible graph envelope. A graph exceeding the evidence budget
advertises an explicit residual-only disposition. All ordinary authority and
state validation remains mandatory before the projection can be published.
"""
    from ..definition import ConcreteDefinition
    from ..node import require_saved, _all_nodes
    from ..node_codec import _describe_node, _value
    from ..node_prepare import NodePreparation
    from ..node_views import object_projection, creation_cdef, OBJECT_PROJECTION_VIEW, CREATION_VIEW
    from .identity import _identity_kind
    from .node_pattern import _parameters
    from .relationships import _definition_edges, _node_key

    for node in preparation.decoded.nodes:
        state = getattr(node, "state", None)
        if state is not None:
            try:
                state.validate_for(node, codec=state.codec)
            except (TypeError, ValueError):
                return {"version": _GRAPH_VERSION, "supported": False}
        if not isinstance(node, ConcreteDefinition) and node.object_id is not None:
            return {"version": _GRAPH_VERSION, "supported": False}

    nodes, slots, rows = [], {}, []
    parameters, edges, guards, objects, creations = [], [], [], [], []
    guard_ids = {}
    object_nodes = {id(node) for root in preparation.objects.values() for node in _all_nodes(root)}

    def visit(node):
        if node._view in slots:
            return slots[node._view]
        slot = slots[node._view] = len(nodes)
        nodes.append(node)
        rows.append(None)
        parameters.append(None)
        edges.append(None)
        guards.append(None)
        objects.append(None)
        creations.append(None)
        rows[slot] = _describe_node(node, visit)
        try:
            parameters[slot] = _value(_parameters(node), visit)
        except TypeError:
            pass
        return slot

    for node in preparation.owner.nodes:
        visit(node)
    position = 0
    while position < len(nodes):
        node = nodes[position]
        projected = node if id(node) in object_nodes else preparation.objects.get(id(node))
        if projected is None:
            projected = object_projection(node)
            object_nodes.update(id(child) for child in _all_nodes(projected))
        objects[position] = visit(projected)
        if isinstance(node, ConcreteDefinition):
            created = preparation.creation.get(id(node))
            creations[position] = visit(creation_cdef(node) if created is None else created)
        natural = _identity_kind(node)
        links = [(edge.kind.value, visit(edge.target)) for edge in _definition_edges(node)]
        if natural == "state_ref":
            links.append(("state_object", objects[position]))
        elif node.object_id is not None:
            links.append(("object_definition", creations[position] if creations[position] is not None else position))
        edges[position] = links
        # Concrete views share an active-path guard with their retained occurrence,
        # even when a saved/object/creation association changes the semantic view.
        guard = _node_key(node)
        guards[position] = guard_ids.setdefault(guard, len(guard_ids))
        position += 1
    owner = NodePreparation(nodes, previous=preparation.owner)
    records, evidence_bytes = [], 0
    for slot, node in enumerate(nodes):
        exact = owner._flat(owner._slot(node)).decode("ascii")
        evidence_bytes += len(exact)
        if evidence_bytes > _MAX_GRAPH_BYTES // 2:
            return {"version": _GRAPH_VERSION, "supported": False}
        saved = False
        try:
            require_saved(node)
            saved = True
        except (TypeError, ValueError):
            pass
        records.append({"shape": rows[slot], "parameters": parameters[slot],
                        "edges": edges[slot], "guard": guards[slot],
                        "object": objects[slot], "creation": creations[slot],
                        "kind": _identity_kind(node), "digest": owner.digest(node), "exact": exact,
                        "keys": {policy: owner.semantic_key(node, policy=policy).decode("ascii")
                                 for policy in ("exact", OBJECT_PROJECTION_VIEW.name, CREATION_VIEW.name)},
                        "saved": saved})
    locators = {table + ":" + str(slot): visit(node)
                for (table, slot), node in preparation.decoded._transport_nodes.items()}
    facts = {key: slots[view] for view, key in preparation.fact_keys.items()}
    data = _dump({"nodes": records, "locators": locators, "facts": facts}).encode("ascii")
    if len(data) > _MAX_GRAPH_BYTES:
        return {"version": _GRAPH_VERSION, "supported": False}
    return {"version": _GRAPH_VERSION, "supported": True, "size": len(data),
            "data": base64.b64encode(zlib.compress(data)).decode("ascii")}


def _read_graph(envelope):
    """Decode bounded primitive evidence; never reconstruct a Node or resolve code."""
    from .model import QueryIndexDirty
    try:
        if (type(envelope) is not dict or type(envelope["version"]) is not int
                or envelope["version"] != _GRAPH_VERSION or type(envelope["supported"]) is not bool):
            raise ValueError()
        if envelope["supported"] is False:
            return None
        size = envelope["size"]
        if type(size) is not int or not 0 <= size <= _MAX_GRAPH_BYTES:
            raise ValueError()
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(envelope["data"], validate=True), size + 1)
        if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError()
        graph = json.loads(raw)
        if (type(graph) is not dict or set(graph) != {"nodes", "locators", "facts"}
                or type(graph["nodes"]) is not list or type(graph["locators"]) is not dict
                or type(graph["facts"]) is not dict):
            raise ValueError()
        size = len(graph["nodes"])
        if any(type(slot) is not int or not 0 <= slot < size
               for table in (graph["locators"], graph["facts"]) for slot in table.values()):
            raise ValueError()
        return graph
    except (ValueError, TypeError, KeyError, zlib.error) as error:
        raise QueryIndexDirty("Malformed certified query graph; rebuild derived coverage.") from error


def _atom(record):
    if record.get("tag") == "leaf":
        value = record["value"]
        if value["type"] == "null":
            return None
        if value["type"] in {"bool", "str"}:
            return value["value"]
        if value["type"] == "int":
            return int(value["value"])
    from ..node_codec import _restore
    return _restore(record, lambda _: (_ for _ in ()).throw(_ResidualRequired()))


@dataclass
class _PostingPredicate:
    """A closed exact program, or an explicit reason to retain Node residuals."""

    prepared: object
    rows: list
    programs: object
    residual: str | None = None

    def matches(self, graph, target):
        """Evaluate pure ordered constraints against certified primitive rows."""
        from .query import _query_match_class, _query_match_leaf
        candidate = graph["nodes"]

        def mapping(value):
            return {_dump(key): (key, item) for key, item in value["items"]}

        def contains_match(value):
            if isinstance(value, dict):
                return value.get("tag") == "match" or any(contains_match(v) for v in value.values())
            return isinstance(value, list) and any(contains_match(v) for v in value)

        def compare(left, right, forward, reverse, exact=0, present=True):
            tag = left.get("tag")
            if tag == "match" and not exact:
                match = left["matcher"]
                if match["kind"] == "missing":
                    return not present
                if not present:
                    return False
                if match["kind"] in {"any", "present"}:
                    return True
                if match["kind"] == "range":
                    value = _atom(right)
                    return isinstance(value, int) and match["lo"] <= value <= match["hi"]
                if match["kind"] == "choice":
                    return any(_atom(right) == _atom(v) for v in match["values"])
                raise _ResidualRequired()
            if not present:
                return False
            while exact != 2 and left.get("tag") == "link" and left["role"] == "materialize":
                left = left["target"]
            while exact != 2 and right.get("tag") == "link" and right["role"] == "materialize":
                right = right["target"]
            tag = left.get("tag")
            if tag == "node":
                if right.get("tag") != "node":
                    return False
                lslot, rslot = left["slot"], right["slot"]
                if lslot in forward:
                    return forward[lslot] == rslot
                if rslot in reverse:
                    return False
                a, b = self.rows[lslot], candidate[rslot]
                if a["shape"]["kind"] == "concrete" and b["shape"]["kind"] != "concrete":
                    return False
                anchor = bool(exact or a["shape"]["kind"] == "concrete")
                left_cls, right_cls = a["shape"]["cls"], b["shape"]["cls"]
                if (left_cls.get("tag") == right_cls.get("tag") == "leaf"
                        and left_cls["value"]["type"] == right_cls["value"]["type"] == "import_ref"):
                    if left_cls != right_cls:
                        return False
                else:
                    cls = _atom(left_cls)
                    if (cls is not None or anchor) and not _query_match_class(
                            cls, _atom(right_cls), strict=self.prepared.strict or anchor,
                            class_match="exact" if anchor else self.prepared.class_match):
                        return False
                identity = _atom(a["shape"]["identity"])
                if identity is not None and identity != _atom(b["shape"]["identity"]):
                    return False
                if a["parameters"] is None or b["parameters"] is None:
                    raise _ResidualRequired()
                forward[lslot], reverse[rslot] = rslot, lslot
                return compare(a["parameters"], b["parameters"], forward, reverse, int(anchor))
            if tag == "link":
                return (right.get("tag") == tag and left["role"] == right["role"]
                        and compare(left["target"], right["target"], forward, reverse, exact))
            if tag == "factory":
                if right.get("tag") != tag:
                    return False
                strength = exact or (0 if contains_match(left) else 2)
                return (compare(left["target"], right["target"], forward, reverse, 2)
                        and compare(left["args"], right["args"], forward, reverse, strength)
                        and compare(left["kwargs"], right["kwargs"], forward, reverse, strength))
            if tag == "map":
                if right.get("tag") != tag:
                    return False
                a, b = mapping(left), mapping(right)
                if exact and a.keys() != b.keys():
                    return False
                return all(compare(value, b[key][1] if key in b else {}, forward, reverse, exact,
                                   present=key in b) for key, (_, value) in a.items())
            if tag in {"list", "tuple"}:
                return (right.get("tag") == tag and len(left["items"]) == len(right["items"])
                        and all(compare(a, b, forward, reverse, exact) for a, b in zip(left["items"], right["items"])))
            if right.get("tag") in {"node", "link", "quote", "selector", "factory"}:
                return False
            if tag == "match":
                return left == right
            if (tag == right.get("tag") == "leaf"
                    and left["value"]["type"] in {"null", "bool", "str", "int"}
                    and right["value"]["type"] in {"null", "bool", "str", "int"}):
                return left == right
            return _query_match_leaf(_atom(left), _atom(right), strict=self.prepared.strict or bool(exact),
                                     class_match=self.prepared.class_match)

        def evaluate(program):
            kind, value = program
            if kind == "node":
                return compare({"tag": "node", "slot": value}, {"tag": "node", "slot": target}, {}, {})
            results = [evaluate(child) for child in value]
            return all(results) if kind == "and" else any(results) if kind == "or" else not results[0]

        if self.residual is not None:
            raise _ResidualRequired(self.residual)
        return evaluate(self.programs)


def _compile(prepared):
    """Compile exact class/ordered-container/factory predicates without IO.

Unordered assignments, quotations, state pins, runtime code and support witnesses
remain explicit Node residuals. Unsupported candidate parameter bindings may
also request residual execution while interpreting an otherwise closed program.
"""
    from ..definition import ConcreteDefinition
    from ..node_codec import _describe_node, _value
    from .node_pattern import _parameters

    rows, slots = [], {}

    def visit(node):
        if node._view in slots:
            return slots[node._view]
        if node.state is not None or (node.cls is not None and not isinstance(node, ConcreteDefinition)
                                     and not (prepared.strict or prepared.class_match == "exact")):
            raise _ResidualRequired("state or subclass matching")
        slot = slots[node._view] = len(rows)
        rows.append(None)
        shape = _describe_node(node, visit)
        parameters = _value(_parameters(node), visit)
        rows[slot] = {"shape": shape, "parameters": parameters}
        return slot

    def program(predicate):
        if predicate.kind == "node":
            return "node", visit(predicate.value)
        if predicate.kind not in {"and", "or", "not"}:
            raise _ResidualRequired("support or path predicate")
        return predicate.kind, [program(child) for child in predicate.operands]

    def inspect(value):
        if isinstance(value, dict):
            tag = value.get("tag")
            if tag == "factory":
                def has_node(item):
                    if isinstance(item, dict):
                        return item.get("tag") == "node" or any(has_node(v) for v in item.values())
                    return isinstance(item, list) and any(has_node(v) for v in item)
                if has_node(value):
                    raise _ResidualRequired("exact factory graph assignment")
            if tag in {"set", "frozenset", "quote", "selector", "local-state", "config", "par", "binary", "repeat"}:
                raise _ResidualRequired("non-ordered or symbolic value")
            if tag == "match" and value["matcher"]["kind"] not in {"any", "present", "missing", "range", "choice"}:
                raise _ResidualRequired("non-posting matcher")
            if tag == "match" and value["matcher"]["kind"] == "choice" and any(
                    item.get("tag") not in {"leaf", "float", "complex"} for item in value["matcher"]["values"]):
                raise _ResidualRequired("non-scalar Choice")
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    try:
        if prepared is None or prepared.runtime_matchers:
            raise _ResidualRequired("runtime predicate")
        programs = program(prepared.predicate)
        inspect(rows)
        return _PostingPredicate(prepared, rows, programs)
    except (TypeError, _ResidualRequired) as error:
        return _PostingPredicate(prepared, rows, None, str(error))


def _count_paths(graph, root, matches, kinds):
    """Count compact typed paths, retaining the ordinary active-guard semantics."""
    nodes = graph["nodes"]
    memo = {}

    def walk(slot, active):
        if slot in memo:
            total, dependencies = memo[slot]
            if not dependencies.intersection(active - {nodes[slot]["guard"]}):
                return total, dependencies
        total, dependencies = 0, set()
        for kind, child in nodes[slot]["edges"]:
            if kind not in kinds:
                continue
            total += bool(matches(child))
            guard = nodes[child]["guard"]
            dependencies.add(guard)
            if guard not in active:
                subtotal, child_dependencies = walk(child, active | frozenset((guard,)))
                total += subtotal
                dependencies.update(child_dependencies)
        # A suffix that cut off an earlier ancestor is contextual; do not reuse
        # its result on another path with a smaller active guard set.
        if not dependencies.intersection(active - {nodes[slot]["guard"]}):
            memo[slot] = total, dependencies
        return total, dependencies

    return walk(root, frozenset((nodes[root]["guard"],)))[0]
