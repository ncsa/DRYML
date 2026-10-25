"""Production Node key proofs; the flat oracle does not use preparation facts."""

from itertools import permutations
import json
from copy import deepcopy

import numpy as np
import pytest

from dryml.core import Definition, Object, ObjectId


class KeyNode(Object):
    def __init__(self, value=None):
        self.value = value


def _positive(value):
    return value > 0


def _negative(value):
    return value < 0


def test_graph_digest_includes_node_local_facets():
    root = Definition(KeyNode, 1).concretize()
    bound = root.with_identity(ObjectId(("keyproof",)))
    assert not root.graph_equal(bound)
    assert root.exact_digest() != bound.exact_digest()


def test_one_contract_two_schedulers_and_warm_requests():
    from dryml.core.node_prepare import NodePreparation

    leaf = Definition(KeyNode, "same")
    root = Definition(KeyNode, [leaf, leaf, Definition(KeyNode, "same")])
    eager = NodePreparation((root,), scheduler="eager")
    deferred = NodePreparation((root,), scheduler="deferred")
    for node in reversed(eager.nodes):
        assert eager.exact_key(node) == deferred.exact_key(node)
    before = dict(deferred.counts)
    deferred.exact_key(root)
    assert deferred.counts == before
    assert eager.counts["discovery"] == 3
    assert eager.counts["edges"] == 3


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _oracle(root):
    """Minimize a flat equality-partition table over ALL root-fixed bijections.

    No hybrid choice, fold, digest, refinement, production descriptor or scheduler
    is used. Leaf codecs are shared only for established atomic semantic values.
    """
    from collections.abc import Mapping
    from dryml.core.node import Node
    from dryml.core.definition import ConcreteDefinition
    from dryml.core.canonical import container_family
    from dryml.core.links import DefLink
    from dryml.core.quoted import QuotedDef, SelectorSpec
    from dryml.core.reference_values import ObjectRef, StateRef, StateSelectorRef
    from dryml.core.selector import ObjectSelector
    from dryml.core.factory import FactorySpec
    from dryml.core.config import ConfigRef
    from dryml.core.state_values import LocalState, StateKey
    from dryml.core.template import Par, _BinaryExpr, _RepeatExpr
    from dryml.core.params import Match, SatisfiesMatcher
    from dryml.core.symbol import symbol_ref
    from dryml.core.reference_json import _value_to_json

    nodes = []
    slots = {}

    def walk(value, labels=None):
        if isinstance(value, (Node, ConcreteDefinition)):
            if labels is not None:
                return ["edge", labels[id(value)]]
            if id(value) not in slots:
                slots[id(value)] = len(nodes)
                nodes.append(value)
                row(value, None)
            return None
        if isinstance(value, DefLink):
            return ["role", value.kind.value, walk(value.target, labels)]
        if isinstance(value, QuotedDef):
            return ["quoted", walk(value.value, labels)]
        if isinstance(value, SelectorSpec):
            return ["selector", walk(value.selector.root, labels), value.selector.strict, value.selector.cls_policy]
        if isinstance(value, StateSelectorRef):
            return ["state-selector", walk(value.object, labels), value.alias]
        if isinstance(value, StateRef):
            return ["state-ref", walk(value.object, labels), sorted([[walk(value.object.objects[path], labels), item] for path, item in value.states.items()], key=_json)]
        if isinstance(value, ObjectRef):
            return ["object-ref", walk(value.definition, labels), sorted([[None if target is None else walk(target, labels), walk(identity, labels)] for target, identity in value._node_bindings], key=_json)]
        if isinstance(value, ObjectSelector):
            return ["object-selector", walk(value.definition, labels), walk(value.reference, labels)]
        if isinstance(value, FactorySpec):
            return ["factory", walk(value.target, labels), walk(value.args, labels), walk(value.kwargs, labels)]
        if isinstance(value, ConfigRef):
            return ["config", value.key, value.has_default, walk(value.default, labels) if value.has_default else None]
        if isinstance(value, LocalState):
            return ["local", walk(value.key, labels), walk(value.creation_cdef, labels)]
        if isinstance(value, StateKey):
            return ["statekey", value.cdef_digest, value.codec_hash]
        if isinstance(value, Par):
            return ["par", value.name, value.path.to_data()]
        if isinstance(value, _BinaryExpr):
            return ["binary", value.operation, walk(value.left, labels), walk(value.right, labels)]
        if isinstance(value, _RepeatExpr):
            return ["repeat", walk(value.group, labels), walk(value.count, labels), value.shared]
        if isinstance(value, Match):
            if isinstance(value.matcher, SatisfiesMatcher):
                return ["satisfies", value.name, value.matcher.name, walk(symbol_ref(value.matcher.predicate), labels)]
            return ["match", walk(value.stable_key(), labels)]
        if isinstance(value, Mapping):
            return ["mapping", sorted([[walk(key, labels), walk(item, labels)] for key, item in value.items()], key=_json)]
        family = container_family(value)
        if family is not None:
            items = [walk(item, labels) for item in value]
            if family in {"set", "frozenset"}:
                items.sort(key=_json)
            return [family, items]
        if isinstance(value, np.ndarray):
            if value.dtype.fields is not None:
                return ["structured", str(value.dtype.descr), list(value.shape), [[name, walk(value[name], labels)] for name in value.dtype.names]]
            return ["array", str(value.dtype.descr), value.dtype.str, list(value.shape), value.tobytes().hex()]
        if type(value) in (float, complex):
            return [type(value).__name__, value.hex() if type(value) is float else [value.real.hex(), value.imag.hex()]]
        if isinstance(value, type):
            value = symbol_ref(value)
        return _value_to_json(value, grammar=3)

    def row(node, labels):
        concrete = isinstance(node, ConcreteDefinition)
        return [concrete, walk(node.cls, labels),
                None if concrete or node.args is None else walk(node.args, labels),
                walk(node.parameters if concrete else node.kwargs, labels),
                walk(node.object_id if isinstance(node, Node) else None, labels),
                walk(node.state if isinstance(node, Node) else None, labels)]

    walk(root)
    return min(_json([row(node, {id(n): index for index, n in enumerate((root,) + order)}) for node in (root,) + order])
               for order in permutations(nodes[1:]))


def _shapes():
    leaf = Definition(KeyNode, 0)
    pair = lambda a, b: Definition(KeyNode, [a, b])
    diamond = pair(Definition(KeyNode, leaf), Definition(KeyNode, leaf))
    independent = pair(Definition(KeyNode, 0), Definition(KeyNode, 0))
    a, b = Definition(KeyNode, 0), Definition(KeyNode, 0)
    tied = Definition(KeyNode, [{Definition(KeyNode, a), Definition(KeyNode, b)}, a])
    tied_cross = Definition(KeyNode, [{Definition(KeyNode, a), Definition(KeyNode, a)}, b])
    return [leaf, Definition(KeyNode, Definition(KeyNode, leaf)), pair(leaf, leaf),
            independent, diamond, pair(diamond, leaf), tied, tied_cross]


def test_hybrid_equality_partitions_match_independent_flat_oracle():
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph

    roots = _shapes()
    roots += [root.copy_graph() for root in roots]
    expected = [_oracle(root) for root in roots]
    eager = NodePreparation(roots, scheduler="eager")
    lazy = NodePreparation(reversed(roots), scheduler="deferred")
    for root in reversed(roots):
        assert eager.exact_key(root) == lazy.exact_key(root)
        assert eager.exact_key(root) == NodePreparation((root,)).exact_key(root)
    for i, left in enumerate(roots):
        for j, right in enumerate(roots):
            assert (eager.exact_key(left) == eager.exact_key(right)) == (expected[i] == expected[j])
    packet = encode_node_graph(eager)
    for index, root in enumerate(roots[:8]):
        decoded = decode_node_graph(packet, view=index)
        assert _oracle(decoded) == _oracle(root)
        assert decoded.exact_key() == root.exact_key()
        assert decoded is not root
    print("U03-ORACLE shapes", len(roots), "comparisons", len(roots) ** 2)


def test_unordered_alias_ties_preserve_external_and_cross_member_links(monkeypatch):
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph

    left, right = Definition(KeyNode, 0), Definition(KeyNode, 0)
    x, y = Definition(KeyNode, left), Definition(KeyNode, right)
    root = Definition(KeyNode, [{x, y}, left, x])
    copied = root.copy_graph()
    wrong = Definition(KeyNode, [{x, y}, left, y])
    monkeypatch.setattr(Definition, "__repr__", lambda self: (_ for _ in ()).throw(AssertionError("repr is not identity")))
    a, b, c = (NodePreparation((node,)) for node in (root, copied, wrong))
    assert a.exact_key(root) == b.exact_key(copied)
    assert a.exact_key(root) != c.exact_key(wrong)
    assert a.exact_equal(root, b, copied)
    restored = decode_node_graph(encode_node_graph(a))
    assert restored.exact_key() == root.exact_key()
    members, external, selected = restored.args[0]
    assert selected in members
    assert selected.args[0] is external
    assert _oracle(root) != _oracle(wrong)


def test_digest_ties_do_not_select_shape_or_authorize_equality(monkeypatch):
    import dryml.core.node_prepare as module
    roots = _shapes()
    prepared = module.NodePreparation(roots, scheduler="eager")
    schemes = [prepared.scheme(root) for root in roots]
    monkeypatch.setattr(module, "_digest", lambda *args: "0" * 64)
    tied = module.NodePreparation(roots, scheduler="deferred")
    assert [tied.scheme(root) for root in roots] == schemes
    assert schemes == ["compose", "compose", "compose", "compose", "adjacency", "adjacency", "adjacency", "adjacency"]
    assert tied.exact_key(roots[2]) == tied.exact_key(roots[3])
    assert not tied.exact_equal(roots[2], tied, roots[3])
    copied = roots[4].copy_graph()
    assert tied.exact_equal(roots[4], module.NodePreparation((copied,)), copied)
    assert not tied.exact_equal(roots[6], tied, roots[7])
    copied = roots[6].copy_graph()
    assert tied.exact_equal(roots[6], module.NodePreparation((copied,)), copied)


def test_full_value_grammar_and_retained_construction_round_trip(monkeypatch):
    from dryml.core import Serializable, ObjectRef, StateRef
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    from dryml.core.state_values import LocalState, StateKey
    from dryml.core.symbol import ImportRef, SourceSpec
    from dryml.core.template import Par, repeat
    from dryml.core.factory import FactorySpec
    from dryml.core.config import ConfigRef
    from dryml.core.dtype import DType
    from dryml.core.cardinality import Cardinality
    from dryml.core.tensor_spec import TensorSpec, Dynamic, Layout
    from dryml.core.params import AnyValue, Missing, Present, Exact, Choice, IntRange, SubclassOf, Satisfies
    from dryml.core.links import DefLink
    from dryml.core.cdef_graph import EdgeKind
    from dryml.core.utils.graph.path import GraphPath

    creation = Definition(KeyNode, 1).concretize()
    state = LocalState(StateKey("a" * 64, "pkl-" + "b" * 64), creation)
    pinned = creation.with_state(state)
    realized = pinned.with_realization(identities={"$": ObjectId(("realized",))}, states={"$": None})
    refdef = Definition(Serializable).concretize()
    object_ref = ObjectRef(refdef, {GraphPath(): ObjectId(("embedded",))})
    state_ref = StateRef(object_ref, {GraphPath(): "pkl-" + "c" * 64})
    future = Definition(KeyNode, Par("n") * 2, omitted=repeat([Definition(KeyNode, 1)], Par("n")))
    values = [None, True, 13, -0.0, 1.25, b"bytes", "str", np.int16(3),
              np.arange(4, dtype=">i2"), np.array([(3, 4.)], dtype=[("x", "i2"), ("y", ">f8")]),
              [1], (1,), {1}, frozenset({1}), {2: "two", "one": 1},
              DType("float", 32), Cardinality.UNKNOWN, TensorSpec("float32", shape=(Dynamic, 2)), Layout.DENSE,
              ConfigRef("size", 4), FactorySpec(ImportRef("builtins", "tuple"), [1, 2]),
              SourceSpec.from_source("lambda value: value", kind="function"),
              realized, pinned.quote(), future.quote(), future,
              DefLink.assertion(EdgeKind.REF, future), DefLink.finalized(EdgeKind.MATERIALIZE, creation),
              object_ref, state_ref, AnyValue(), Missing(), Present(), Exact(3), Choice([1, 2]), IntRange(1, 3), SubclassOf(Object), Satisfies(_positive)]
    root = Definition(KeyNode, values)
    # The explicit enum leaf codec may resolve its enum type; constructor targets,
    # factory targets and symbolic references must remain inert.
    resolve = ImportRef.resolve
    def guarded(ref):
        assert ref.module == "dryml.core.tensor_spec" and ref.qualname == "Layout"
        return resolve(ref)
    monkeypatch.setattr(ImportRef, "resolve", guarded)
    import dryml.core.cdef_codec as legacy
    monkeypatch.setattr(legacy, "encode_cdef_graph", lambda *a, **k: (_ for _ in ()).throw(AssertionError("nested rooted export")))
    eager = NodePreparation((root,), scheduler="eager")
    packet = encode_node_graph(eager)
    restored = decode_node_graph(json.loads(json.dumps(packet)))
    assert restored.exact_key() == root.exact_key()
    assert eager.exact_equal(root, NodePreparation((restored,)), restored)
    restored_realized = restored.args[0][22]
    assert restored_realized.construction_view.state == state
    assert restored_realized.state is None
    assert restored_realized.object_id == realized.object_id
    assert restored_realized.construction_view is not pinned


def test_shared_table_views_and_edit_reuse_are_owned_and_bounded():
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    from dryml.core.utils.graph.path import GraphPath, Parameter, Index

    leaf = Definition(KeyNode, np.arange(4)).concretize()
    root = Definition(KeyNode, [leaf, Definition(KeyNode, 2)]).concretize()
    prepared = NodePreparation((root,), scheduler="eager")
    packet = encode_node_graph(prepared, prepared.nodes)
    assert len(packet["tables"]) == 1
    assert len(next(iter(packet["tables"].values()))["rows"]) == 3
    assert len(packet["views"]) == 3
    assert prepared.counts["export_rows"] == 3
    before = dict(prepared.counts)
    for node in prepared.nodes:
        prepared.exact_key(node)
    assert prepared.counts == before
    changed = root.lens(GraphPath((Parameter("value"), Index(1), Parameter("value")))).set(3)
    edited = NodePreparation((changed,), previous=prepared, scheduler="eager")
    assert edited.counts["reused_rows"] == 1
    assert edited.counts["folds"] == 2
    assert edited.exact_key(leaf) == prepared.exact_key(leaf)
    assert edited.exact_key(changed) == NodePreparation((changed,)).exact_key(changed)
    bounded = NodePreparation((root,), max_cached_keys=1, scheduler="eager")
    assert len(bounded._facts) == 1
    assert bounded.exact_key(root) == prepared.exact_key(root)
    bounded.clear()
    assert bounded.exact_key(root) == prepared.exact_key(root)
    packet["views"][1]["slot"] = 999
    with pytest.raises(ValueError):
        decode_node_graph(packet)


def _reauthenticate(packet):
    """Reauthenticate the value DAG so shape mutations reach structural checks."""
    import hashlib
    def digest(domain, record):
        return hashlib.sha256(domain.encode() + b"\0" + _json(record).encode()).hexdigest()
    table = next(iter(packet["tables"].values()))
    mapping, values = {}, {}
    def rewrite(value):
        if isinstance(value, dict):
            if value.get("tag") == "block":
                return {"tag": "block", "key": rekey(value["key"])}
            return {key: rewrite(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        return value
    def rekey(key):
        if key not in mapping:
            value = rewrite(packet["values"][key])
            mapping[key] = digest("dryml-node-values/v1", value)
            values[mapping[key]] = value
        return mapping[key]
    for key in packet["values"]:
        rekey(key)
    packet["values"] = values
    for row in table["rows"]:
        row["structures"] = [mapping[key] for key in row["structures"]]
    for row in table["overlays"]:
        row["value"] = mapping[row["value"]]
        row["structure"] = mapping[row["structure"]]
    key = digest("dryml-node-table/v1", table)
    packet["tables"] = {key: table}
    for view in packet["views"]:
        view["table"] = key


@pytest.mark.parametrize("mutation", ["version", "identity", "bool-version", "key-format", "view-kind", "key", "slot", "table-ref", "value-ref", "cycle", "unreachable", "extra", "construction-kind", "tamper"])
def test_shared_codec_rejects_malformed_or_incompatible_inputs_intact(mutation):
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph, NodeCodecError

    root = Definition(KeyNode, Definition(KeyNode, 1)).concretize().with_realization(identities={"$": ObjectId(("test",))})
    packet = encode_node_graph(NodePreparation((root,)))
    table = next(iter(packet["tables"].values()))
    row = packet["values"][table["overlays"][0]["value"]]
    if mutation == "version": packet["format"] = "dryml-node-graph/v99"
    elif mutation == "identity": packet["identity_version"] = 2
    elif mutation == "bool-version": packet["identity_version"] = True
    elif mutation == "key-format": packet["views"][0]["key_format"] = "dryml-node-exact/v0"
    elif mutation == "view-kind": packet["views"][0]["view"] = "creation"
    elif mutation == "key": packet["views"][0]["expected_key"] = "0" * 64
    elif mutation == "slot": packet["views"][0]["slot"] = -1
    elif mutation == "table-ref": packet["views"][0]["table"] = "f" * 64
    elif mutation == "value-ref": table["rows"][0]["structures"][0] = "f" * 64
    elif mutation == "cycle":
        kwargs = packet["values"][row["kwargs"]["key"]]
        kwargs["items"][0][1] = {"tag": "node", "slot": 0}
    elif mutation == "unreachable": table["rows"].append(deepcopy(table["rows"][1]))
    elif mutation == "extra": row["unknown"] = True
    elif mutation == "construction-kind": table["overlays"][0]["construction"]["view"] = "exact"
    elif mutation == "tamper": row["stateful_role"] = not row["stateful_role"]
    if mutation in {"cycle", "unreachable", "extra", "construction-kind"}:
        _reauthenticate(packet)
    before = deepcopy(packet)
    with pytest.raises(NodeCodecError, match="Cyclic Node" if mutation == "cycle" else None):
        decode_node_graph(packet)
    assert packet == before


def test_selected_keys_ignore_unrelated_roots_and_block_placement():
    from dryml.core.node_prepare import NodePreparation, PreparationView
    from dryml.core.node_codec import encode_node_graph, decode_node_graph, NodeCodecError

    root = _shapes()[4]
    extra = Definition(KeyNode, "unrelated")
    a, b = NodePreparation((root,)), NodePreparation((extra, root))
    assert a.exact_key(root) == b.exact_key(root)
    assert encode_node_graph(a)["views"][0]["table"] != encode_node_graph(b)["views"][0]["table"]
    assert decode_node_graph(encode_node_graph(b), view=1).exact_key() == root.exact_key()
    named = NodePreparation((root,), view=PreparationView("future-policy"))
    assert named.exact_key(root) != a.exact_key(root)
    with pytest.raises(NodeCodecError, match="kind"):
        decode_node_graph(encode_node_graph(named))
    assert decode_node_graph(encode_node_graph(named), expected_view="future-policy").exact_key() == root.exact_key()


def test_legacy_codec_does_not_silently_drop_new_node_evidence():
    from dryml.core.cdef_codec import encode_cdef_graph, CDefGraphCodecError
    from dryml.core.utils.graph.path import GraphPath, Parameter
    node = Definition(KeyNode, 1).concretize().with_identity(ObjectId(("legacyboundary",)))
    for root in (node, Definition(KeyNode, node).concretize(),
                 Definition(KeyNode, 1).concretize().lens(GraphPath((Parameter("value"),))).set(node.quote()),
                 node.with_realization(identities={"$": None})):
        with pytest.raises(CDefGraphCodecError, match="Node"):
            encode_cdef_graph(root)


def test_frozen_array_rejects_object_dtype_before_retention():
    from dryml.core.freeze import FrozenNDArray
    with pytest.raises(TypeError, match="Object-dtype"):
        FrozenNDArray.from_array(np.array([object()], dtype=object))


def test_array_padding_is_not_logical_constructor_content():
    from dryml.core.freeze import FrozenNDArray, FrozenTuple, FrozenDict
    from dryml.core.symbol import ImportRef
    dtype = np.dtype({"names": ["x"], "formats": ["u1"], "offsets": [0], "itemsize": 4})
    arrays = [np.frombuffer(raw, dtype=dtype).view(FrozenNDArray) for raw in (b"\x01\0\0\0", b"\x01abc")]
    roots = [Definition._from_symbolic_parts(ImportRef(__name__, "KeyNode"), FrozenTuple((array,)), FrozenDict({})) for array in arrays]
    assert roots[0].exact_key() == roots[1].exact_key()
    assert _oracle(roots[0]) == _oracle(roots[1])


def test_full_grammar_equality_partitions_against_flat_oracle():
    from dryml.core import SKIP_ARGS, Serializable, ObjectRef, StateRef
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    from dryml.core.template import Par, repeat, Shared
    from dryml.core.symbol import ImportRef, SourceSpec
    from dryml.core.factory import FactorySpec
    from dryml.core.config import ConfigRef
    from dryml.core.dtype import DType
    from dryml.core.tensor_spec import TensorSpec, Dynamic
    from dryml.core.cardinality import Cardinality
    from dryml.core.params import AnyValue, Missing, Present, Exact, Choice, IntRange, SubclassOf, Satisfies
    from dryml.core.state_values import LocalState, StateKey
    from dryml.core.quoted import SelectorSpec
    from dryml.core.selector import Selector
    from dryml.core.links import DefLink
    from dryml.core.cdef_graph import EdgeKind
    from dryml.core.utils.graph.path import GraphPath

    child = Definition(KeyNode, 1)
    creation = child.concretize()
    state = LocalState(StateKey("a" * 64, "pkl-" + "b" * 64), creation)
    obj = ObjectRef(Definition(Serializable).concretize(), {GraphPath(): ObjectId(("oracle",))})
    values = [None, False, 0, -0.0, 0.0, b"", "", [1], (1,), {1}, frozenset({1}),
              {"x": 1, 2: "y"}, {2: "y", "x": 1},
              np.array([1], dtype="i2"), np.array([1], dtype=">i2"), np.int16(1),
              DType("float", 32), TensorSpec("float32", shape=(Dynamic,)), Cardinality.UNKNOWN,
              ConfigRef("size", child), FactorySpec(ImportRef("builtins", "tuple"), [FactorySpec(ImportRef("builtins", "tuple"), [1, 2])]),
              SourceSpec.from_source("lambda value: value", kind="function"),
              Par("n"), Par("n") * 2, 2 / Par("n"), Par("n") // 2,
              repeat([child], Par("n")), repeat((child,), Shared(Par("n"))),
              child, child.quote(), DefLink.assertion(EdgeKind.REF, child), DefLink.assertion(EdgeKind.MATERIALIZE, child),
              SelectorSpec(Selector(Definition(SKIP_ARGS, value=AnyValue()))),
              obj, StateRef(obj, {GraphPath(): "pkl-" + "c" * 64}), obj.state("best"),
              obj.object_projection(),
              AnyValue(), Missing(), Present(), Exact(child), Choice([child, 1]), IntRange(1, 2), SubclassOf(Object),
              Satisfies(_positive, name="sign"), Satisfies(_negative, name="sign"), Satisfies(_positive)]
    roots = [Definition(KeyNode, value) for value in values]
    roots.extend([Definition(KeyNode), Definition(SKIP_ARGS, value=1), creation,
                  creation.with_identity(ObjectId(("bound",))), creation.with_state(state)])
    # Fresh portable round trips, rather than the production scheduler, supply the
    # second occurrence labeling. Each oracle closure here has at most 3 Nodes.
    copies = [decode_node_graph(encode_node_graph(NodePreparation((root,)))) for root in roots]
    expected = [_oracle(root) for root in roots + copies]
    eager = NodePreparation(roots + copies, scheduler="eager")
    deferred = NodePreparation(reversed(roots + copies))
    keys = [eager.exact_key(root) for root in roots + copies]
    for root, key in zip(roots + copies, keys):
        assert deferred.exact_key(root) == key
    for i in range(len(keys)):
        for j in range(len(keys)):
            assert (keys[i] == keys[j]) == (expected[i] == expected[j])
    packet = encode_node_graph(eager)
    assert decode_node_graph(packet, view=len(roots)-3).exact_key() == creation.exact_key()
    print("U03-ORACLE grammar", len(keys), "comparisons", len(keys) ** 2)


def _workload(kind, size):
    leaf = lambda: Definition(KeyNode, "leaf")
    pair = lambda a, b: Definition(KeyNode, [a, b])
    root = leaf()
    if kind == "tree":
        level = [leaf() for _ in range(size)]
        while len(level) > 1:
            level = [pair(level[index], level[index+1]) if index+1 < len(level) else level[index]
                     for index in range(0, len(level), 2)]
        return level[0]
    for index in range(size):
        if kind == "chain": root = Definition(KeyNode, root)
        elif kind == "alias": root = pair(root, root)
        elif kind == "diamond": root = pair(Definition(KeyNode, root), Definition(KeyNode, root))
        elif kind == "crossjoin": root = pair(pair(root, leaf()), pair(root, root))
        elif kind == "unordered": root = Definition(KeyNode, [{Definition(KeyNode, root), Definition(KeyNode, root)}, root])
        else: raise AssertionError(kind)
    return root


@pytest.mark.parametrize("kind", ["chain", "tree", "alias", "diamond", "crossjoin", "unordered"])
@pytest.mark.parametrize("size", [3, 6])
@pytest.mark.parametrize("mode", ["recipe", "facets", "quoted-symbolic"])
def test_small_workload_counts(kind, size, mode):
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph
    root = _workload(kind, size)
    if mode == "facets":
        from dryml.core.state_values import LocalState, StateKey
        state = LocalState(StateKey("d" * 64, "pkl-" + "e" * 64), Definition(KeyNode, "creation").concretize())
        root = root.with_identity(ObjectId(("workload",))).with_state(state)
    elif mode == "quoted-symbolic":
        from dryml.core.template import Par
        root = Definition(KeyNode, {"quote": root.quote(), "symbol": Par("width") * 2})
    eager = NodePreparation((root,), scheduler="eager")
    deferred = NodePreparation((root,))
    for node in reversed(eager.nodes):
        assert eager.exact_key(node) == deferred.exact_key(node)
        eager.digest(node)
    before = dict(eager.counts)
    for node in eager.nodes:
        eager.exact_key(node)
        eager.digest(node)
    assert eager.counts == before
    packet = encode_node_graph(eager, eager.nodes)
    assert len(next(iter(packet["tables"].values()))["rows"]) == len(eager.nodes)
    assert eager.counts["discovery"] == eager.counts["folds"] == len(eager.nodes)
    assert eager.counts["export_rows"] == len(eager.nodes)
    if kind in {"chain", "tree", "alias"}:
        assert eager.counts["fallback_rows"] == 0
    assert len(eager.edges) == eager.counts["edges"]
    print("U03-WORK", kind, size, mode, _json(eager.counts))


def test_quotation_keeps_aliases_to_other_branches_and_role_changes_are_exact():
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    from dryml.core.links import DefLink
    from dryml.core.cdef_graph import EdgeKind
    child = Definition(KeyNode, 1)
    shared = Definition(KeyNode, [child, child.quote()])
    split = Definition(KeyNode, [child, Definition(KeyNode, 1).quote()])
    reference = Definition(KeyNode, [DefLink.assertion(EdgeKind.REF, child), child.quote()])
    assert _oracle(shared) != _oracle(split)
    assert shared.exact_key() != split.exact_key() != reference.exact_key()
    restored = decode_node_graph(encode_node_graph(NodePreparation((shared,))))
    assert restored.args[0][0] is restored.args[0][1].value
    assert restored.graph_equal(shared)


def test_disjoint_unordered_equal_occurrences_compose_without_interning():
    from dryml.core.node_prepare import NodePreparation
    first, second = Definition(KeyNode, 1), Definition(KeyNode, 1)
    root = Definition(KeyNode, {first, second})
    copied = root.copy_graph()
    singleton = Definition(KeyNode, {first, first})
    prepared = NodePreparation((root,), scheduler="eager")
    assert prepared.scheme(root) == "compose"
    assert prepared.counts["fallback_rows"] == 0
    assert prepared.counts["unordered_ties"] == 2
    assert len(root.args[0]) == 2
    assert prepared.exact_key(root) == copied.exact_key()
    assert prepared.exact_key(root) != singleton.exact_key()
    assert _oracle(root) == _oracle(copied) != _oracle(singleton)


def test_bounded_caches_keep_concurrent_requests_and_clear_semantically_stable():
    from concurrent.futures import ThreadPoolExecutor
    from dryml.core.node_prepare import NodePreparation
    root = _workload("diamond", 3)
    prepared = NodePreparation((root,))
    expected = root.exact_key()
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(lambda _: prepared.exact_key(root), range(12))) == [expected] * 12
    assert prepared.counts["folds"] == len(prepared.nodes)
    copied = NodePreparation((root,), previous=prepared)
    copied.digest(root)
    before = dict(prepared.counts)
    prepared.digest(root)
    assert prepared.counts["hashes"] == before["hashes"] + 1
    tiny = NodePreparation((root,), max_cached_bytes=1)
    assert tiny.exact_key(root) == expected
    assert tiny._cache_bytes == 0
    assert not tiny._facts
    with ThreadPoolExecutor(max_workers=3) as pool:
        def request(index):
            if index % 2:
                prepared.clear()
                return NodePreparation((root,), previous=prepared).exact_key(root)
            return prepared.exact_key(root)
        assert list(pool.map(request, range(12))) == [expected] * 12


def test_metadata_field_names_in_symbol_imports_remain_exact_constructor_data():
    from dryml.core.symbol import SourceSpec, ImportRef
    roots = [Definition(KeyNode, SourceSpec.from_source(
        "lambda: stateful_role", kind="function",
        imports={"stateful_role": ImportRef("builtins", target)})) for target in ("str", "int")]
    assert _oracle(roots[0]) != _oracle(roots[1])
    assert roots[0].exact_key() != roots[1].exact_key()


@pytest.mark.parametrize("source_backed", [False, True])
def test_symbolic_predicate_codec_is_inert_until_matching(monkeypatch, source_backed):
    from dryml.core import Satisfies
    from dryml.core.symbol import ImportRef, SourceSpec
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    def positive(value):
        return value > 0
    predicate = positive if source_backed else _positive
    roots = [Definition(KeyNode, Satisfies(predicate, name="sign")) for predicate in (predicate, _negative)]
    assert roots[0].stable_hash() == roots[1].stable_hash()
    assert roots[0].exact_key() != roots[1].exact_key()
    with monkeypatch.context() as guarded:
        guarded.setattr(ImportRef, "resolve", lambda *a: pytest.fail("predicate resolution at metadata boundary"))
        guarded.setattr(SourceSpec, "resolve", lambda *a: pytest.fail("predicate source execution at metadata boundary"))
        restored = decode_node_graph(encode_node_graph(NodePreparation((roots[0],))))
        assert restored.exact_key() == roots[0].exact_key()
    assert restored.args[0].matches(1)
    assert not restored.args[0].matches(-1)
    assert restored.args[0].matcher.stable_key() == roots[0].args[0].matcher.stable_key()


def test_construction_handles_authenticate_occurrence_association():
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph, NodeCodecError
    root = Definition(KeyNode, Definition(KeyNode, 1)).concretize().with_realization(identities={"$": ObjectId(("association",))})
    packet = encode_node_graph(NodePreparation((root,)))
    table = next(iter(packet["tables"].values()))
    retained = table["overlays"][0]["construction"]["slot"]
    table["roots"].append(retained)
    table["overlays"][0]["construction"]["slot"] = 1
    _reauthenticate(packet)
    with pytest.raises(NodeCodecError, match="occurrence"):
        decode_node_graph(packet)


def test_soft_retained_view_survives_concretization_metadata_round_trip():
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    root = Definition(KeyNode, 1).with_realization(identities={"$": ObjectId(("softview",))}).concretize()
    restored = decode_node_graph(encode_node_graph(NodePreparation((root,))))
    assert restored.construction_view.graph_equal(root.construction_view)
    assert restored.construction_view._occurrence is restored._occurrence


def test_legacy_handles_with_one_private_node_token_remain_aliases():
    from dryml.core.definition import ConcreteDefinition
    from dryml.core.bound_args import BoundArguments
    from dryml.core.freeze import FrozenTuple
    from dryml.core.symbol import ImportRef
    from dryml.core.node_prepare import NodePreparation
    cls = ImportRef(__name__, "KeyNode")
    leaf = ConcreteDefinition._from_bound_record(cls, BoundArguments({"value": 1}), stateful_role=False)
    alias = object.__new__(ConcreteDefinition)
    alias._copy_record(leaf)
    roots = [ConcreteDefinition._from_bound_record(cls, BoundArguments({"value": FrozenTuple(children)}), stateful_role=False)
             for children in ((leaf, leaf), (leaf, alias))]
    prepared = NodePreparation(roots)
    assert prepared.exact_key(roots[0]) == prepared.exact_key(roots[1])
    assert len(prepared.nodes) == 3


def test_reference_bindings_survive_new_unordered_path_context(monkeypatch):
    from collections.abc import Mapping
    from dryml.core import Serializable, ObjectRef, StateRef
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    from dryml.core import reference_values
    from dryml.core.utils.graph import value as paths

    left, right = Definition(Serializable), Definition(Serializable)
    graph = Definition(KeyNode, {"alias": left, "members": {left, right}}).concretize()
    alias = graph.parameters["value"]["alias"]
    old_ordinal = next(segment.ordinal for segment, member in paths.iter_set_members(graph.parameters["value"]["members"]) if member is alias)
    direct, _ = reference_values._expected_paths(graph)
    id_a, id_b = ObjectId(("alias",)), ObjectId(("other",))
    objects = {path: id_a if key == ("cdef", alias._node_id) else id_b for key, path in direct.items()}
    ref = ObjectRef(graph, objects)
    state = StateRef(ref, {path: "pkl-" + ("a" if identity == id_a else "b") * 64 for path, identity in objects.items()})
    root = Definition(KeyNode, [ref, state, alias])
    packet = encode_node_graph(NodePreparation((root,)))
    flat = _oracle(root)
    expected, members = reference_values._expected_paths, paths.iter_set_members

    def changed_context(definition, **kwargs):
        payload = definition.parameters.get("value")
        if not isinstance(payload, Mapping) or set(payload) != {"alias", "members"}:
            return expected(definition, **kwargs)
        selected = payload["alias"]
        def ordered(values):
            original = members(values)
            if len(original) != 2 or not any(member is selected for _, member in original):
                return original
            values = sorted((member for _, member in original), key=lambda member: (member is selected) == (old_ordinal == 0))
            return tuple((segment, member) for (segment, _), member in zip(original, values))
        with monkeypatch.context() as local:
            local.setattr(paths, "iter_set_members", ordered)
            return expected(definition, **kwargs)

    monkeypatch.setattr(reference_values, "_expected_paths", changed_context)
    restored = decode_node_graph(packet)
    assert restored.exact_key() == root.exact_key()
    assert _oracle(restored) == flat
    restored_ref, restored_state, external = restored.args[0]
    assert restored_ref.definition.parameters["value"]["alias"] is external
    current, _ = reference_values._expected_paths(restored_ref.definition)
    assert restored_ref.objects[current[("cdef", external._node_id)]] == id_a
    assert restored_state.states[current[("cdef", external._node_id)]] == "pkl-" + "a" * 64


def test_imported_reference_pins_remain_logical_object_bindings():
    from dryml.core import Serializable, ObjectRef, StateRef
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    from dryml.core.utils.graph.path import GraphPath
    from dryml.core import reference_values
    identity = ObjectId(("imported",))
    child = ObjectRef(Definition(Serializable).concretize(), {GraphPath(): identity})
    selected = StateRef(child, {GraphPath(): "pkl-" + "a" * 64})
    parent = Definition(KeyNode, [selected, selected]).concretize()
    expected, _ = reference_values._expected_paths(parent)
    reference = ObjectRef(parent, {path: identity for path in expected.values()})
    assert reference._node_bindings == ((None, identity),)
    state = StateRef(reference, {path: "pkl-" + "a" * 64 for path in reference.objects})
    root = Definition(KeyNode, [reference, state])
    restored = decode_node_graph(encode_node_graph(NodePreparation((root,))))
    assert restored.exact_key() == root.exact_key()
    assert _oracle(restored) == _oracle(root)
    assert restored.args[0][0]._node_bindings == ((None, identity),)


@pytest.mark.parametrize("mutation", ["version", "bool-version", "extra", "missing", "definition", "reference", "bindings"])
def test_object_selector_codec_rejects_malformed_metadata_intact(mutation):
    from dryml.core import Serializable, ObjectRef, ObjectSelector
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph, NodeCodecError
    from dryml.core.utils.graph.path import GraphPath

    reference = ObjectRef(Definition(Serializable).concretize(), {GraphPath(): ObjectId(("selector",))})
    packet = encode_node_graph(NodePreparation((Definition(KeyNode, ObjectSelector(reference)),)))
    selector = next(record for record in packet["values"].values() if record.get("tag") == "object-selector")
    if mutation == "version": selector["version"] = 2
    elif mutation == "bool-version": selector["version"] = True
    elif mutation == "extra": selector["unknown"] = None
    elif mutation == "missing": del selector["reference"]
    elif mutation in {"definition", "reference"}:
        selector[mutation] = {"tag": "leaf", "value": {"type": "int", "value": "1"}}
    elif mutation == "bindings":
        bindings = next(record for record in packet["values"].values() if record.get("tag") == "object-ref")["bindings"]
        packet["values"][bindings["key"]]["items"].append(
            deepcopy(packet["values"][bindings["key"]]["items"][0]))
    _reauthenticate(packet)
    before = deepcopy(packet)
    with pytest.raises(NodeCodecError):
        decode_node_graph(packet)
    assert packet == before


def _legacy_key_root(value):
    from dryml.core.definition import ConcreteDefinition
    from dryml.core.bound_args import BoundArguments
    from dryml.core.symbol import ImportRef

    return ConcreteDefinition._from_bound_record(
        ImportRef(__name__, "KeyNode"), BoundArguments({"value": value}), stateful_role=False)


def test_legacy_cdef_lookup_compares_recipes_across_fresh_node_handles(monkeypatch):
    """Legacy digest buckets use value evidence, not Node handle equality."""
    from dryml.core.cdef_codec import encode_cdef_graph, decode_cdef_graph
    from dryml.core.cdef_identity import same_cdef
    from dryml.core.definition import ConcreteDefinition, ConcreteNode

    child = Definition(KeyNode, 7).concretize()
    root = Definition(KeyNode, (child, child)).concretize()
    decoded = decode_cdef_graph(encode_cdef_graph(root))
    independent = Definition(KeyNode, (child, child.copy_graph())).concretize()
    wrong = Definition(KeyNode, 8).concretize()

    assert root != decoded
    assert root.stable_hash() == decoded.stable_hash()
    for left, right in ((root, decoded), (decoded, root), (root, root.copy_graph())):
        assert same_cdef(left, right)
        assert left.graph_equal(right)
    assert same_cdef(root, independent)
    assert not root.graph_equal(independent)

    for cls in (ConcreteDefinition, ConcreteNode):
        monkeypatch.setattr(cls, "stable_hash", lambda self: "0" * 64)
    assert not same_cdef(root, wrong)
    assert not same_cdef(wrong, decoded)


def test_legacy_reference_scopes_compare_exact_evidence_without_weakening_node_keys(monkeypatch):
    from dryml.core import Serializable, ObjectRef, ObjectSelector
    from dryml.core.freeze import FrozenTuple
    from dryml.core.node import _as_node_view
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    from dryml.core.utils.graph.path import GraphPath
    import dryml.core.node_prepare as preparation

    child = Definition(Serializable).concretize()
    copied = decode_node_graph(encode_node_graph(NodePreparation((child,))))
    identity = ObjectId(("scoped",))
    a, b = (ObjectRef(node, {GraphPath(): identity}) for node in (child, copied))
    left = _legacy_key_root(FrozenTuple((a, a)))
    right = _legacy_key_root(FrozenTuple((a, b)))
    assert left.graph_equal(right)
    assert not NodePreparation((left,)).exact_equal(left, NodePreparation((right,)), right)
    assert not left.graph_equal(_as_node_view(right))
    assert NodePreparation((left,)).exact_key(left) != NodePreparation((right,)).exact_key(right)

    faceted = ObjectRef(child.with_identity(ObjectId(("facet",))), {GraphPath(): identity})
    bound = _legacy_key_root(FrozenTuple((a, faceted)))
    selectors = [_legacy_key_root(ObjectSelector(ref)) for ref in (a, faceted)]
    monkeypatch.setattr(preparation, "_digest", lambda *args: "0" * 64)
    assert left.graph_equal(right)
    assert not left.graph_equal(bound)
    assert not selectors[0].graph_equal(selectors[1])
    assert not _legacy_key_root(FrozenTuple((child, child))).graph_equal(
        _legacy_key_root(FrozenTuple((child, copied))))


def test_legacy_scoped_comparison_backtracks_unordered_choices_through_external_aliases():
    from dryml.core.freeze import FrozenSet, FrozenTuple

    def fixture():
        a, b = Definition(KeyNode, 0).concretize(), Definition(KeyNode, 0).concretize()
        x, y = Definition(KeyNode, a).concretize(), Definition(KeyNode, b).concretize()
        return a, b, x, y

    a, b, x, y = fixture()
    left = _legacy_key_root(FrozenTuple((FrozenSet((x, y)), x, a)))
    a, b, x, y = fixture()
    for selected, child, other in ((x, a, b), (y, b, a)):
        right = _legacy_key_root(FrozenTuple((FrozenSet((x, y)), selected, child)))
        wrong = _legacy_key_root(FrozenTuple((FrozenSet((x, y)), selected, other)))
        assert left.graph_equal(right)
        assert not left.graph_equal(wrong)


def test_legacy_scoped_comparison_keeps_codec_field_names_in_symbol_data():
    from dryml.core.symbol import ImportRef, SourceSpec

    roots = [_legacy_key_root(SourceSpec.from_source(
        "lambda: tag", kind="function", imports={"tag": ImportRef("builtins", target)}))
        for target in ("str", "str", "int")]
    assert roots[0].graph_equal(roots[1])
    assert not roots[0].graph_equal(roots[2])


@pytest.mark.parametrize("concrete", [False, True])
def test_field_edits_keep_node_facets_and_invalidate_retained_construction(concrete):
    from dryml.core import Node
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    from dryml.core.state_values import LocalState, StateKey
    from dryml.core.utils.graph.path import GraphPath, Arg, Parameter
    root = Definition(KeyNode, 1)
    if concrete:
        root = root.concretize()
    identity = ObjectId(("edited",))
    state = LocalState(StateKey("a" * 64, "pkl-" + "b" * 64), Definition(KeyNode, 2).concretize())
    root = root.with_realization(identities={"$": identity}, states={"$": state})
    path = GraphPath((Parameter("value") if concrete else Arg(0),))
    changed = root.lens(path).set(2)
    assert isinstance(changed, Node)
    assert changed.object_id == identity and changed.state == state
    assert changed.construction_view is changed
    assert root.construction_view is not root
    assert changed.exact_key() != root.exact_key()
    restored = decode_node_graph(encode_node_graph(NodePreparation((changed,))))
    assert restored.object_id == identity and restored.state == state
    assert restored.exact_key() == changed.exact_key()


def test_role_assertion_array_targets_are_snapshots_before_pure_reuse():
    from dryml.core.links import DefLink
    from dryml.core.cdef_graph import EdgeKind
    from dryml.core.node_prepare import NodePreparation
    values = np.arange(4)
    root = Definition(KeyNode, DefLink.assertion(EdgeKind.MATERIALIZE, {"array": values}))
    prepared = NodePreparation((root,), scheduler="eager")
    key = prepared.exact_key(root)
    values[0] = 99
    assert root.exact_key() == key
    assert not root.args[0].target["array"].flags.writeable
    with pytest.raises(TypeError):
        root.args[0].target["other"] = 1
    assert NodePreparation((root,), previous=prepared).exact_key(root) == key


def test_registered_nonobject_dtype_retains_interpretation_not_just_void_bytes():
    extension = pytest.importorskip("ml_dtypes")
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    values = np.array([1, 2], dtype=extension.bfloat16)
    root = Definition(KeyNode, values)
    restored = decode_node_graph(encode_node_graph(NodePreparation((root,))))
    assert restored.args[0].dtype == values.dtype
    np.testing.assert_array_equal(restored.args[0], values)
    assert restored.exact_key() == root.exact_key()


def test_structured_numpy_scalar_does_not_retain_its_mutable_array_buffer():
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    values = np.array([(1,)], dtype=[("value", "i4")])
    root = Definition(KeyNode, values[0])
    key = root.exact_key()
    values[0]["value"] = 99
    assert root.exact_key() == key
    assert not root.args[0].flags.writeable
    restored = decode_node_graph(encode_node_graph(NodePreparation((root,))))
    assert restored.args[0]["value"] == 1
    assert restored.exact_key() == key


@pytest.mark.parametrize("value", [np.bytes_(b""), np.bytes_(b"a\0"), np.str_(""), np.str_("a\0"), np.int16(3), np.float32(-0.)])
def test_numpy_scalar_dtype_and_literal_content_round_trip(value):
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    root = Definition(KeyNode, value)
    assert root.args[0].dtype == value.dtype
    assert root.args[0].tobytes() == value.tobytes()
    restored = decode_node_graph(encode_node_graph(NodePreparation((root,))))
    assert restored.args[0].dtype == value.dtype
    assert restored.args[0].tobytes() == value.tobytes()
    assert restored.exact_key() == root.exact_key()


@pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_x87_storage_padding_does_not_change_logical_array_keys(dtype, byteorder):
    from dryml.core.freeze import FrozenNDArray, FrozenTuple, FrozenDict
    from dryml.core.symbol import ImportRef
    component = np.dtype(np.longdouble)
    if np.finfo(component).nmant != 63 or component.itemsize <= 10:
        pytest.skip("This platform does not use padded x87 long doubles.")
    values = np.array([1.25], dtype=np.dtype(dtype).newbyteorder(byteorder))
    first, second = bytearray(values.tobytes()), bytearray(values.tobytes())
    for offset in range(0, values.dtype.itemsize, component.itemsize):
        start = offset + 10 if byteorder == "<" else offset
        stop = offset + component.itemsize if byteorder == "<" else offset + component.itemsize - 10
        first[start:stop] = b"\0" * (component.itemsize - 10)
        second[start:stop] = b"x" * (component.itemsize - 10)
    arrays = [np.frombuffer(bytes(raw), dtype=values.dtype).view(FrozenNDArray) for raw in (first, second)]
    np.testing.assert_array_equal(*arrays)
    roots = [Definition._from_symbolic_parts(ImportRef(__name__, "KeyNode"), FrozenTuple((array,)), FrozenDict({})) for array in arrays]
    assert roots[0].exact_key() == roots[1].exact_key()


def test_public_digest_domains_hash_the_versioned_exact_envelope():
    import hashlib
    from dryml.core.node_prepare import NodePreparation
    root = Definition(KeyNode, 1)
    prepared = NodePreparation((root,))
    key = prepared.exact_key(root)
    assert prepared.digest(root) == hashlib.sha256(b"dryml-node-graph/v1\0" + key).hexdigest()
    assert prepared.digest(root, domain="dryml-creation-cdef/v1") == hashlib.sha256(b"dryml-creation-cdef/v1\0" + key).hexdigest()
    assert prepared.digest(root) != prepared.digest(root, domain="dryml-creation-cdef/v1")
    before = dict(prepared.counts)
    prepared.digest(root)
    assert prepared.counts == before


@pytest.mark.parametrize("depth", [8, 16])
def test_shared_symbolic_value_dag_uses_shared_blocks_not_expanded_trees(depth):
    from dryml.core.template import Par
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    expression = Par("width")
    for _ in range(depth):
        expression = expression * expression
    root = Definition(KeyNode, expression)
    eager = NodePreparation((root,), scheduler="eager")
    deferred = NodePreparation((root,))
    assert eager.exact_key(root) == deferred.exact_key(root)
    packet = encode_node_graph(eager)
    assert eager.counts["export_bytes"] < 7000 + 500 * depth
    restored = decode_node_graph(packet)
    assert restored.exact_key() == root.exact_key()
    assert restored.args[0].left is restored.args[0].right
    print("U03-VALUE-DAG", depth, _json(eager.counts))


def test_pure_symbolic_sharing_is_not_node_occurrence_identity():
    from dryml.core.template import Par
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph

    def expression(depth, shared):
        if not depth:
            return Par("width")
        left = expression(depth - 1, shared)
        return left * (left if shared else expression(depth - 1, shared))

    roots = [Definition(KeyNode, expression(3, shared)) for shared in (True, False)]
    assert _oracle(roots[0]) == _oracle(roots[1])
    prepared = NodePreparation(roots, scheduler="eager")
    assert prepared.exact_key(roots[0]) == prepared.exact_key(roots[1])
    assert prepared.exact_equal(roots[0], prepared, roots[1])
    restored = decode_node_graph(encode_node_graph(prepared), view=1)
    assert restored.graph_equal(roots[0])


@pytest.mark.parametrize("size", [10, 12])
@pytest.mark.parametrize("concrete", [False, True])
def test_canonical_ordered_transport_avoids_artificial_symmetry(monkeypatch, size, concrete):
    """Ordered equal siblings need no search, without collapsing occurrences."""
    import dryml.core.node_prepare as preparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph

    candidates = 0
    labelings = preparation._labelings

    def bounded_labelings(cells, index=0, prefix=()):
        nonlocal candidates
        for labeling in labelings(cells, index, prefix):
            if index == 0:
                candidates += 1
                assert candidates <= 4, "Ordered transport is enumerating artificial symmetry."
            yield labeling

    monkeypatch.setattr(preparation, "_labelings", bounded_labelings)
    children = [Definition(KeyNode, "equal") for _ in range(size)]
    root = Definition(KeyNode, [*children, children[0]])
    wrong = Definition(KeyNode, [*children, children[-1]])
    if concrete:
        root, wrong = root.concretize(), wrong.concretize()
    prepared = preparation.NodePreparation((root,))
    packet = encode_node_graph(prepared, canonical=True)
    assert candidates == 0
    assert prepared.counts["partition_rows"] == prepared.counts["unordered_ties"] == 0
    assert prepared.counts["discovery"] == prepared.counts["export_rows"] == size + 1
    table = next(iter(packet["tables"].values()))
    assert len(table["rows"]) == len(table["overlays"]) == size + 1

    restored = decode_node_graph(packet)
    values = restored.parameters["value"] if concrete else restored.args[0]
    assert len(set(values[:size])) == size
    assert values[-1] is values[0]
    assert all(left is not right for left, right in zip(values[:size], children))
    assert restored.graph_equal(root)
    assert encode_node_graph(preparation.NodePreparation((restored,)), canonical=True) == packet
    assert encode_node_graph(preparation.NodePreparation((root.copy_graph(),)), canonical=True) == packet
    assert encode_node_graph(preparation.NodePreparation((wrong,)), canonical=True) != packet
    assert wrong.exact_key() != root.exact_key()
    assert candidates == 0


@pytest.mark.parametrize("case", ["unordered", "multiview", "construction"])
def test_canonical_transport_keeps_exact_fallback_for_real_view_choices(case):
    """Unordered aliases and distinct occurrence views retain full-table search."""
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, decode_node_graph
    from dryml.core.state_values import LocalState, StateKey

    original = Definition(KeyNode, 1).concretize()
    identity = ObjectId(("canonicaltransport",))
    state = LocalState(StateKey("a" * 64, "pkl-" + "b" * 64), original)
    if case == "unordered":
        root = Definition(KeyNode, [{original, original.copy_graph()}, original])
    elif case == "multiview":
        selected = original.with_identity(identity)
        root = Definition(KeyNode, [selected, selected, original.quote()])
    else:
        pinned = original.with_state(state)
        realized = pinned.with_realization(identities={"$": identity}, states={"$": None})
        root = Definition(KeyNode, [realized, realized, pinned.quote()])
    prepared = NodePreparation((root,))
    packet = encode_node_graph(prepared, canonical=True)
    assert prepared.counts["partition_rows"] > 0
    restored = decode_node_graph(packet)
    assert restored.graph_equal(root)
    assert encode_node_graph(NodePreparation((restored,)), canonical=True) == packet
    if case == "unordered":
        members, selected = restored.args[0]
        assert len(members) == 2 and selected in members
    else:
        selected, alias, quoted = restored.args[0]
        retained = quoted.value
        assert alias is selected
        assert selected is not retained
        assert selected._occurrence is retained._occurrence
        assert selected.object_id == identity and retained.object_id is None
        if case == "construction":
            assert selected.state is None and retained.state == state
            assert selected.construction_view is retained
