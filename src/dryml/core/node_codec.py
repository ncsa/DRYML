"""Closed Node graph v1 blocks and compact authenticated root views.

This is the identity-3 family, separate from the still-live CDef V2 Store codec.
Values use explicit codecs; neither pickle nor repr is semantic authority.
Detached projection owners can retain all validated views from one shared decode.
"""

import base64
import struct

from .node_prepare import EXACT_KEY_VERSION, NodePreparation, _bytes, _digest, _map, _canonical_values
from .cdef_identity import NODE_IDENTITY_VERSION, validate_node_identity_version
from .symbol import ImportRef, SourceSpec

NODE_GRAPH_VERSION = "dryml-node-graph/v1"


class NodeCodecError(ValueError):
    """Malformed, incompatible or unauthenticated Node graph metadata."""


class _PredicateImport(ImportRef):
    """A symbolic predicate resolved only at the existing matcher call boundary."""

    def __call__(self, value):
        return self.resolve()(value)


class _PredicateSource(SourceSpec):
    """Source-backed equivalent of _PredicateImport; never executed on decode."""

    def __call__(self, value):
        return self.resolve()(value)


def _describe_node(node, visit, state=None):
    from .definition import ConcreteDefinition
    from .node import Node
    concrete = isinstance(node, ConcreteDefinition)
    if state is None:
        state = {"memo": {}, "active": set()}
    encode = lambda value: _value(value, visit, state)
    return {
        "kind": "concrete" if concrete else "soft",
        "cls": encode(node.cls),
        "args": None if concrete or node.args is None else encode(node.args),
        "kwargs": encode(node.parameters if concrete else node.kwargs),
        "stateful_role": node._stateful_role if concrete else None,
        "identity": encode(node.object_id if isinstance(node, Node) else None),
        "state": encode(node.state if isinstance(node, Node) else None),
    }


def _value(value, visit, state=None):
    from .definition import ConcreteDefinition
    from .node import Node
    if isinstance(value, (Node, ConcreteDefinition)):
        return {"tag": "node", "slot": visit(value)}
    if state is None:
        state = {"memo": {}, "active": set()}
    marker = id(value)
    if marker in state["active"]:
        raise TypeError("Cyclic constructor value.")
    if marker in state["memo"]:
        return state["memo"][marker][1]
    state["active"].add(marker)
    try:
        result = _value_record(value, visit, state)
        state["memo"][marker] = value, result
        return result
    finally:
        state["active"].remove(marker)


def _value_record(value, visit, state):
    from collections.abc import Mapping
    import numpy as np
    from .canonical import container_family
    from .definition import ConcreteDefinition
    from .node import Node
    from .factory import FactorySpec
    from .config import ConfigRef
    from .links import DefLink
    from .quoted import QuotedDef, SelectorSpec
    from .reference_values import ObjectRef, StateRef, StateSelectorRef
    from .selector import ObjectSelector
    from .state_values import LocalState, StateKey
    from .template import Expr, Par, _BinaryExpr, _RepeatExpr, _validate_number
    from .params import Match
    from .symbol import symbol_ref
    from .reference_json import _value_to_json

    encode = lambda item: _value(item, visit, state)
    if isinstance(value, (Node, ConcreteDefinition)):
        return {"tag": "node", "slot": visit(value)}
    if isinstance(value, DefLink):
        return {"tag": "link", "role": value.kind.value, "finalized": value.is_finalized, "target": encode(value.target)}
    if isinstance(value, QuotedDef):
        return {"tag": "quote", "value": encode(value.value)}
    if isinstance(value, SelectorSpec):
        return {"tag": "selector", "root": encode(value.selector.root),
                "strict": value.selector.strict, "cls_policy": value.selector.cls_policy}
    if isinstance(value, LocalState):
        return {"tag": "local-state", "version": 1, "key": encode(value.key), "creation": encode(value.creation_cdef)}
    if isinstance(value, StateKey):
        return {"tag": "state-key", "version": 1, "cdef": value.cdef_digest, "codec": value.codec_hash}
    if isinstance(value, FactorySpec):
        return {"tag": "factory", "target": encode(value.target), "args": encode(value.args), "kwargs": encode(value.kwargs)}
    if isinstance(value, ConfigRef):
        return {"tag": "config", "key": value.key, "has_default": value.has_default,
                "default": encode(value.default) if value.has_default else None}
    if isinstance(value, Par):
        return {"tag": "par", "name": value.name, "path": value.path.to_data()}
    if isinstance(value, _BinaryExpr):
        if value.operation not in {"mul", "truediv", "floordiv"} or not all(
                isinstance(operand, Expr) or _validate_number(operand) for operand in (value.left, value.right)):
            raise TypeError("Unsupported symbolic binary expression.")
        return {"tag": "binary", "op": value.operation, "left": encode(value.left), "right": encode(value.right)}
    if isinstance(value, _RepeatExpr):
        if type(value.shared) is not bool or container_family(value.group) not in {"list", "tuple"}:
            raise TypeError("Unsupported symbolic repetition.")
        return {"tag": "repeat", "group": encode(value.group), "count": encode(value.count), "shared": value.shared}
    if isinstance(value, Match):
        return _match(value, encode)
    if isinstance(value, StateRef):
        return {"tag": "state-ref", "version": 1, "object": encode(value.object),
                "states": {"tag": "set", "items": [
                    {"identity": encode(value.object.objects[path]), "state": state}
                    for path, state in value.states.items()]}}
    if isinstance(value, ObjectRef):
        bindings = _reference_bindings(value)
        return {"tag": "object-ref", "version": 1, "definition": encode(value.definition),
                "bindings": {"tag": "set", "items": [
                    {"target": None if target is None else encode(target), "identity": encode(identity)}
                    for target, identity in bindings]}}
    if isinstance(value, StateSelectorRef):
        return {"tag": "state-selector", "version": 1, "object": encode(value.object), "alias": value.alias}
    if isinstance(value, ObjectSelector):
        return {"tag": "object-selector", "version": 1,
                "definition": encode(value.definition), "reference": encode(value.reference)}
    if isinstance(value, Mapping):
        from .utils.graph.path import canonical_key_bytes
        if any(type(key) not in (str, int) for key in value):
            raise TypeError("Node mapping keys must be exact strings or integers.")
        return {"tag": "map", "items": [[encode(key), encode(item)] for key, item in sorted(value.items(), key=lambda pair: canonical_key_bytes(pair[0]))]}
    family = container_family(value)
    if family is not None:
        return {"tag": family, "items": [encode(item) for item in value]}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object-dtype arrays are not durable Node values.")
        return {"tag": "array", "dtype": _dtype(value.dtype), "shape": list(value.shape),
                "bytes": base64.b64encode(_logical_bytes(value)).decode("ascii")}
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise TypeError("Object-dtype scalars are not durable Node values.")
        raw = value.tobytes()[:value.dtype.itemsize]
        scalar_array = np.ndarray((), dtype=value.dtype, buffer=raw)
        return {"tag": "numpy-scalar", "dtype": _dtype(value.dtype),
                "bytes": base64.b64encode(_logical_bytes(scalar_array)).decode("ascii")}
    if type(value) is float:
        return {"tag": "float", "bits": struct.pack("!d", value).hex()}
    if type(value) is complex:
        return {"tag": "complex", "bits": struct.pack("!dd", value.real, value.imag).hex()}
    if isinstance(value, type):
        value = symbol_ref(value)
    # The shared explicit semantic leaf codecs cover DType, TensorSpec,
    # Cardinality and ObjectId. Unsupported/live values fail closed there.
    return {"tag": "leaf", "value": _value_to_json(value, grammar=3)}


def _reference_bindings(reference):
    bindings = getattr(reference, "_node_bindings", None)
    if bindings is None:
        # Old process-local pickle shells may predate this derived pure fact.
        # Interpret their existing path authority once; never export rooted blobs.
        from .reference_values import _expected_paths
        nodes = {}
        expected, _ = _expected_paths(reference.definition, node_values=nodes)
        bindings = tuple((nodes.get(key), reference.objects[path]) for key, path in expected.items())
        object.__setattr__(reference, "_node_bindings", bindings)
    return bindings


def _match(value, encode):
    from .params import (AnyMatcher, ChoiceMatcher, ExactMatcher, IntRangeMatcher,
                         MissingMatcher, PresentMatcher, SubclassMatcher, SatisfiesMatcher)
    matcher = value.matcher
    if value.name is not None and type(value.name) is not str:
        raise TypeError("Match names must be strings or absent.")
    kinds = {AnyMatcher: "any", MissingMatcher: "missing", PresentMatcher: "present"}
    if type(matcher) in kinds:
        record = {"kind": kinds[type(matcher)]}
    elif type(matcher) is ExactMatcher:
        record = {"kind": "exact", "value": encode(matcher.value)}
    elif type(matcher) is ChoiceMatcher:
        record = {"kind": "choice", "values": [encode(item) for item in matcher.values]}
    elif type(matcher) is IntRangeMatcher:
        if type(matcher.lo) is not int or type(matcher.hi) is not int or matcher.lo > matcher.hi:
            raise TypeError("Invalid Match integer range.")
        record = {"kind": "range", "lo": matcher.lo, "hi": matcher.hi}
    elif type(matcher) is SubclassMatcher:
        record = {"kind": "subclass", "cls": encode(matcher.cls)}
    elif type(matcher) is SatisfiesMatcher:
        from .symbol import symbol_ref
        if not callable(matcher.predicate) or matcher.name is not None and type(matcher.name) is not str:
            raise TypeError("Satisfies needs a symbol-addressable predicate and optional string name.")
        record = {"kind": "satisfies", "name": matcher.name, "predicate": encode(symbol_ref(matcher.predicate))}
    else:
        raise TypeError("Unsupported Node Match leaf.")
    return {"tag": "match", "name": value.name, "matcher": record}


def _dtype(dtype):
    # Explicit offsets/itemsize preserve padding, overlapping fields and alignment;
    # dtype.str alone loses those properties for structured/subarray dtypes.
    if dtype.hasobject or dtype.kind not in "buifcmMSUV":
        raise TypeError("Unsupported managed or object NumPy dtype.")
    if dtype.fields is not None:
        fields = []
        for name in dtype.names:
            field = dtype.fields[name]
            fields.append([name, _dtype(field[0]), field[1], field[2] if len(field) == 3 else None])
        body = {"fields": fields, "itemsize": dtype.itemsize, "aligned": dtype.isalignedstruct}
    elif dtype.subdtype is not None:
        body = {"base": _dtype(dtype.subdtype[0]), "shape": list(dtype.subdtype[1])}
    else:
        import numpy as np
        plain = np.dtype(dtype.str)
        if dtype == plain and type(dtype) is type(plain):
            body = {"str": dtype.str}
        else:
            from .reference_json import _encoded_type
            restored = np.dtype(dtype.type).newbyteorder(dtype.byteorder)
            if restored != dtype or restored.str != dtype.str or type(restored) is not type(dtype):
                raise TypeError("NumPy extension dtype needs an explicit interpretation codec.")
            body = {"extension": _encoded_type(dtype.type, "numpy_dtype"),
                    "byteorder": dtype.byteorder, "name": dtype.name,
                    "str": dtype.str, "itemsize": dtype.itemsize}
        if ((dtype.type is np.longdouble and dtype.itemsize > 8)
                or (dtype.type is np.clongdouble and dtype.itemsize > 16)):
            # f16/c32 have different numeric interpretations on different ABIs.
            info = np.finfo(np.longdouble)
            body["floating_layout"] = [int(info.nmant), int(info.nexp), int(info.minexp), int(info.maxexp)]
    metadata = None if dtype.metadata is None else _value(dict(dtype.metadata), lambda _: (_ for _ in ()).throw(TypeError("dtype metadata cannot contain Nodes")))
    return {"body": body, "metadata": metadata}


def _logical_bytes(array):
    import numpy as np
    raw = array.tobytes(order="C")
    def padded_longdouble(dtype):
        return (dtype.type is np.longdouble and dtype.itemsize > 10
                and np.finfo(dtype).nmant == 63 and np.finfo(dtype).nexp == 15)

    def padded_complex(dtype):
        return dtype.type is np.clongdouble and padded_longdouble(np.dtype(np.longdouble))

    if not raw or (array.dtype.fields is None and not padded_longdouble(array.dtype) and not padded_complex(array.dtype)):
        return raw
    occupied = np.zeros(array.dtype.itemsize, dtype=bool)

    def mark(dtype, offset):
        if dtype.fields is not None:
            for name in dtype.names:
                field, position = dtype.fields[name][:2]
                mark(field, offset + position)
        elif dtype.subdtype is not None:
            base, shape = dtype.subdtype
            from math import prod
            for index in range(prod(shape)):
                mark(base, offset + index * base.itemsize)
        elif padded_complex(dtype):
            base = np.dtype(np.longdouble).newbyteorder(dtype.byteorder)
            mark(base, offset)
            mark(base, offset + base.itemsize)
        elif padded_longdouble(dtype):
            # x87's 80 value bits occupy 10 bytes in a 12/16-byte ABI slot.
            # Unused storage can contain allocation-dependent bytes.
            start = offset if dtype.str[0] == "<" else offset + dtype.itemsize - 10
            occupied[start:start + 10] = True
        else:
            occupied[offset:offset + dtype.itemsize] = True
    mark(array.dtype, 0)
    canonical = bytearray(raw)
    np.frombuffer(canonical, dtype=np.uint8).reshape(-1, array.dtype.itemsize)[:, ~occupied] = 0
    return bytes(canonical)


def _restore_dtype(record):
    import numpy as np
    body = record["body"]
    if "fields" in body:
        fields = body["fields"]
        dtype = np.dtype({"names": [f[0] for f in fields], "formats": [_restore_dtype(f[1]) for f in fields],
                          "offsets": [f[2] for f in fields], "titles": [f[3] for f in fields],
                          "itemsize": body["itemsize"]}, align=body["aligned"])
    elif "base" in body:
        dtype = np.dtype((_restore_dtype(body["base"]), tuple(body["shape"])))
    elif "extension" in body:
        from .reference_json import _decode_imported_type
        target = _decode_imported_type(body["extension"], "NumPy dtype", {"type", "module", "qualname"})
        dtype = np.dtype(target).newbyteorder(body["byteorder"])
    else:
        dtype = np.dtype(body["str"])
    if record["metadata"] is not None:
        dtype = np.dtype(dtype, metadata=dict(_restore(record["metadata"], None)))
    if dtype.hasobject or _dtype(dtype) != record:
        raise NodeCodecError("Invalid or noncanonical dtype.")
    return dtype


def _restore(data, build, memo=None):
    if memo is None:
        memo = {}
    # Expressions are pure values. Sharing their immutable syntax must not cause
    # exponential reconstruction; ordinary array/container occurrences stay fresh.
    shared = isinstance(data, dict) and data.get("tag") in ("binary", "repeat", "par")
    if shared and id(data) in memo:
        return memo[id(data)]
    result = _restore_record(data, build, memo)
    if shared:
        memo[id(data)] = result
    return result


def _restore_record(data, build, memo):
    from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple, FrozenNDArray
    from .reference_json import _value_from_json
    from .reference_values import ObjectRef, StateRef, StateSelectorRef
    from .selector import ObjectSelector
    from .state_values import StateKey, LocalState
    from .utils.graph.path import GraphPath
    from .quoted import QuotedDef, SelectorSpec
    from .links import DefLink
    from .cdef_graph import EdgeKind
    from .factory import FactorySpec
    from .config import ConfigRef
    from .template import Par, _BinaryExpr, _RepeatExpr
    from .params import (Match, AnyMatcher, MissingMatcher, PresentMatcher,
                         ExactMatcher, ChoiceMatcher, IntRangeMatcher, SubclassMatcher, SatisfiesMatcher)

    if type(data) is not dict or type(data.get("tag")) is not str:
        raise NodeCodecError("Node values must be tagged records.")
    tag = data["tag"]
    if tag in {"object-ref", "state-ref", "state-selector", "object-selector", "state-key", "local-state"}:
        if type(data.get("version")) is not int or data["version"] != 1:
            raise NodeCodecError("Incompatible nested Node reference/state version.")
    value = lambda item: _restore(item, build, memo)
    if tag == "node":
        if set(data) != {"tag", "slot"} or type(data["slot"]) is not int:
            raise NodeCodecError("Invalid node slot.")
        return build(data["slot"])
    if tag == "leaf":
        return _value_from_json(data["value"], grammar=3)
    if tag == "float":
        return struct.unpack("!d", bytes.fromhex(data["bits"]))[0]
    if tag == "complex":
        return complex(*struct.unpack("!dd", bytes.fromhex(data["bits"])))
    if tag in {"array", "numpy-scalar"}:
        import numpy as np
        from math import prod
        dtype = _restore_dtype(data["dtype"])
        raw = base64.b64decode(data["bytes"], validate=True)
        if tag == "numpy-scalar":
            if len(raw) != dtype.itemsize:
                raise NodeCodecError("Invalid NumPy scalar size.")
            if dtype.kind == "S":
                return np.bytes_(raw)
            if dtype.kind == "U":
                import sys
                little = dtype.str[0] == "<" or dtype.str[0] == "=" and sys.byteorder == "little"
                return np.str_(raw.decode("utf-32-le" if little else "utf-32-be", errors="surrogatepass"))
            return np.ndarray((), dtype=dtype, buffer=raw)[()]
        if any(type(dim) is not int or dim < 0 for dim in data["shape"]):
            raise NodeCodecError("Invalid array shape.")
        if len(raw) != prod(data["shape"]) * dtype.itemsize:
            raise NodeCodecError("Invalid array byte count.")
        return FrozenNDArray.from_array(np.ndarray(tuple(data["shape"]), dtype=dtype, buffer=raw))
    if tag == "map":
        pairs = [(value(key), value(item)) for key, item in data["items"]]
        if any(type(key) not in (str, int) for key, _ in pairs) or len(dict(pairs)) != len(pairs):
            raise NodeCodecError("Invalid or duplicate mapping key.")
        return FrozenDict(pairs)
    if tag in {"list", "tuple", "set", "frozenset"}:
        items = [value(item) for item in data["items"]]
        result = {"list": FrozenList, "tuple": FrozenTuple, "set": FrozenSet, "frozenset": frozenset}[tag](items)
        if len(result) != len(items):
            raise NodeCodecError("Duplicate unordered member.")
        return result
    if tag == "link":
        if type(data["finalized"]) is not bool:
            raise NodeCodecError("Invalid link finalization flag.")
        factory = DefLink.finalized if data["finalized"] else DefLink.assertion
        return factory(EdgeKind(data["role"]), value(data["target"]))
    if tag == "quote":
        return QuotedDef(value(data["value"]))
    if tag == "selector":
        from .selector import Selector
        return SelectorSpec(Selector(value(data["root"]), strict=data["strict"], cls_policy=data["cls_policy"]))
    if tag == "state-key":
        return StateKey(data["cdef"], data["codec"])
    if tag == "local-state":
        return LocalState(value(data["key"]), value(data["creation"]))
    if tag == "factory":
        return FactorySpec._from_symbolic_parts(value(data["target"]), value(data["args"]), value(data["kwargs"]))
    if tag == "config":
        if type(data["has_default"]) is not bool:
            raise NodeCodecError("Invalid config default flag.")
        return ConfigRef(data["key"], value(data["default"])) if data["has_default"] else ConfigRef(data["key"])
    if tag == "par":
        return Par(data["name"], path=GraphPath.from_data(data["path"]))
    if tag == "binary":
        return _BinaryExpr(data["op"], value(data["left"]), value(data["right"]))
    if tag == "repeat":
        group = value(data["group"])
        return _RepeatExpr(list(group) if isinstance(group, FrozenList) else group, value(data["count"]), data["shared"])
    if tag == "object-ref":
        from .reference_values import _expected_paths
        definition = value(data["definition"])
        expected, _ = _expected_paths(definition)
        objects = {}
        if data["bindings"]["tag"] != "set":
            raise NodeCodecError("Reference bindings must be unordered incidence.")
        for item in data["bindings"]["items"]:
            identity = value(item["identity"])
            target = None if item["target"] is None else value(item["target"])
            key = ("object-id", identity) if target is None else ("cdef", target._node_id)
            path = expected[key]
            if path in objects:
                raise NodeCodecError("Duplicate reference binding target.")
            objects[path] = identity
        return ObjectRef(definition, objects)
    if tag == "state-ref":
        reference = value(data["object"])
        if data["states"]["tag"] != "set":
            raise NodeCodecError("Reference states must be unordered bindings.")
        states = {}
        for item in data["states"]["items"]:
            identity = value(item["identity"])
            if identity in states:
                raise NodeCodecError("Duplicate state binding identity.")
            states[identity] = item["state"]
        if set(states) != set(reference.objects.values()):
            raise NodeCodecError("State bindings must cover exactly the referenced identities.")
        return StateRef(reference, {path: states[identity] for path, identity in reference.objects.items()})
    if tag == "state-selector":
        return StateSelectorRef(value(data["object"]), data["alias"])
    if tag == "object-selector":
        if set(data) != {"tag", "version", "definition", "reference"}:
            raise NodeCodecError("Invalid ObjectSelector fields.")
        definition, reference = value(data["definition"]), value(data["reference"])
        if not isinstance(reference, ObjectRef):
            raise NodeCodecError("ObjectSelector association must be an ObjectRef.")
        return ObjectSelector._projected(definition, reference.objects, reference)
    if tag == "match":
        matcher = data["matcher"]
        kind = matcher["kind"]
        factories = {"any": AnyMatcher, "missing": MissingMatcher, "present": PresentMatcher}
        if kind in factories:
            match = factories[kind]()
        elif kind == "exact":
            match = ExactMatcher(value(matcher["value"]))
        elif kind == "choice":
            match = ChoiceMatcher(tuple(value(item) for item in matcher["values"]))
        elif kind == "range":
            match = IntRangeMatcher(matcher["lo"], matcher["hi"])
        elif kind == "subclass":
            match = SubclassMatcher(value(matcher["cls"]))
        elif kind == "satisfies":
            predicate = value(matcher["predicate"])
            if isinstance(predicate, ImportRef):
                predicate = _PredicateImport(predicate.module, predicate.qualname)
            elif isinstance(predicate, SourceSpec):
                predicate = _PredicateSource(predicate.kind, predicate.source, predicate.name, predicate.imports)
            else:
                raise NodeCodecError("Satisfies predicate must be symbolic callable evidence.")
            match = SatisfiesMatcher(predicate, matcher["name"])
        else:
            raise NodeCodecError("Invalid Match kind.")
        return Match(match, name=data["name"])
    raise NodeCodecError("Unknown Node value tag.")


def encode_node_graph(prepared, roots=None):
    """Export one authenticated unique table/value block and compact root views.

    Args:
        prepared (NodePreparation): Owning immutable graph preparation.
        roots (Iterable[Node] | None): Required views, defaulting to declared roots.
    Returns:
        dict: Closed JSON-compatible identity-3 ``dryml-node-graph/v1`` packet.
        Each view names a table hash, slot, policy and expected exact key. The
        table contains value-block references and retained construction handles.
    Raises:
        TypeError: If prepared is not a NodePreparation.
        ValueError: If a selected handle is outside the prepared table.
    Side Effects:
        Updates owned export counts and requested pure key facts only; no IO.
    """
    if not isinstance(prepared, NodePreparation):
        raise TypeError("Node graph export requires a NodePreparation.")
    values, rows, overlays = {}, [], []
    occurrences = {}
    occurrence_slots = []
    from .node import Node
    for node in prepared.nodes:
        token = node._occurrence if isinstance(node, Node) else node._node_id
        if token not in occurrences:
            occurrences[token] = len(occurrences)
        occurrence_slots.append(occurrences[token])

    def intern(value):
        key = _digest("dryml-node-values/v1", value)
        prepared.counts["block_hashes"] += 1
        if key in values and values[key] != value:
            raise NodeCodecError("Value block digest collision.")
        values[key] = value
        return key

    exported = {}
    def externalize(value):
        if isinstance(value, (dict, list, tuple)) and id(value) in exported:
            return exported[id(value)][1]
        if isinstance(value, dict):
            result = {key: externalize(child) for key, child in value.items()}
            if value.get("tag") is not None and value.get("tag") != "node":
                result = {"tag": "block", "key": intern(result)}
            exported[id(value)] = value, result
            return result
        if isinstance(value, (tuple, list)):
            result = [externalize(child) for child in value]
            exported[id(value)] = value, result
            return result
        return value

    structural_memo = {}
    for slot, row in enumerate(prepared._rows):
        # externalize owns every emitted container and keeps shared syntax compact.
        key = intern(externalize(row))
        structural = {name: _map(value, lambda child: {"tag": "node", "slot": occurrence_slots[child]}, structural_memo)
                      for name, value in row.items() if name not in {"identity", "state"}}
        structural_key = intern(externalize(structural))
        occurrence = occurrence_slots[slot]
        if occurrence == len(rows):
            rows.append({"structures": [structural_key]})
        elif structural_key not in rows[occurrence]["structures"]:
            rows[occurrence]["structures"].append(structural_key)
        construction = prepared._construction.get(slot)
        overlays.append({"occurrence": occurrence, "value": key, "structure": structural_key,
                         "construction": None if construction is None else
                         {"slot": construction, "view": "construction"}})
    selected = prepared.roots if roots is None else tuple(roots)
    declared = list(dict.fromkeys(prepared._slot(root) for root in prepared.roots))
    table = {"rows": rows, "overlays": overlays, "roots": declared}
    table_key = _digest("dryml-node-table/v1", table)
    prepared.counts["block_hashes"] += 1
    views = [{"table": table_key, "slot": prepared._slot(root), "view": prepared.view.name,
              "key_format": EXACT_KEY_VERSION, "expected_key": prepared.exact_key(root).decode("ascii")}
             for root in selected]
    packet = {"format": NODE_GRAPH_VERSION, "identity_version": NODE_IDENTITY_VERSION,
              "tables": {table_key: table}, "values": values, "views": views}
    prepared.counts["export_rows"] += len(rows)
    prepared.counts["export_bytes"] += len(_bytes(packet))
    return packet


def decode_node_graph(packet, *, view=0, expected_view="exact"):
    """Validate shared blocks once and realize a selected root with fresh tokens.

    Args:
        packet (dict): Closed v1 packet produced by ``encode_node_graph``.
        view (int): Selected compact root-view index.
        expected_view (str): Required policy name, checked before realization.
    Returns:
        Node: Selected immutable root; aliases, exact facets and retained
        construction evidence are restored over the same shared table.
    Raises:
        NodeCodecError: For incompatible identity/key formats, bad authentication,
            slots, cycles, unreachable rows, view kinds or semantic-key mismatch.
    Side Effects:
        Allocates metadata only. Source input is never rewritten. Ordinary symbol
        references are not resolved; explicit enum and extension-dtype value
        types use their owning lazy type codecs.
    """
    try:
        return _decode(packet, view, expected_view)
    except NodeCodecError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, OverflowError, RecursionError, struct.error) as error:
        raise NodeCodecError("Malformed Node graph metadata.") from error


def _decode_node_graph_views(packet):
    """Return all advertised roots using one validated shared-block decode."""
    try:
        return _decode(packet, 0, packet["views"][0]["view"], _all_views=True)
    except NodeCodecError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, OverflowError,
            RecursionError, struct.error) as error:
        raise NodeCodecError("Malformed Node graph metadata.") from error


def _decode(packet, selected, expected_view, *, _all_views=False):
    from .bound_args import BoundArguments
    from .definition import ConcreteNode, Definition
    from .freeze import FrozenDict
    from .node import _Facets
    from .reference_values import ObjectId
    from .state_values import LocalState

    if type(packet) is not dict or set(packet) != {"format", "identity_version", "tables", "values", "views"}:
        raise NodeCodecError("Invalid Node graph envelope.")
    try:
        validate_node_identity_version(packet["identity_version"])
    except ValueError as error:
        raise NodeCodecError(str(error)) from error
    if packet["format"] != NODE_GRAPH_VERSION:
        raise NodeCodecError("Incompatible Node graph or CDef identity version.")
    if type(selected) is not int or selected < 0 or type(packet["views"]) is not list:
        raise NodeCodecError("Invalid selected root view.")
    values, tables = packet["values"], packet["tables"]
    if type(values) is not dict or type(tables) is not dict or not tables:
        raise NodeCodecError("Invalid shared blocks.")
    for key, record in values.items():
        if key != _digest("dryml-node-values/v1", record):
            raise NodeCodecError("Value block authentication failed.")
    for key, table in tables.items():
        if key != _digest("dryml-node-table/v1", table):
            raise NodeCodecError("Table block authentication failed.")
        if set(table) != {"rows", "overlays", "roots"} or any(type(table[field]) is not list for field in ("rows", "overlays", "roots")):
            raise NodeCodecError("Invalid table structure.")
    for root_view in packet["views"]:
        if type(root_view) is not dict or set(root_view) != {"table", "slot", "view", "key_format", "expected_key"}:
            raise NodeCodecError("Invalid root view fields.")
        if root_view["key_format"] != EXACT_KEY_VERSION or root_view["table"] not in tables:
            raise NodeCodecError("Incompatible key format or invalid table reference.")
        if type(root_view["view"]) is not str or not root_view["view"] or type(root_view["expected_key"]) is not str:
            raise NodeCodecError("Invalid root view kind/key.")
    chosen = packet["views"][selected]
    if chosen["view"] != expected_view:
        raise NodeCodecError("Selected semantic view kind mismatch.")
    built_tables, used_values = {}, set()
    expanded, expanding = {}, set()

    def block(key):
        if key in expanding:
            raise NodeCodecError("Cyclic value block reference.")
        used_values.add(key)
        if key not in expanded:
            expanding.add(key)
            expanded[key] = expand(values[key])
            expanding.remove(key)
        return expanded[key]

    def expand(value):
        if type(value) is dict:
            if value.get("tag") == "block":
                if set(value) != {"tag", "key"}:
                    raise NodeCodecError("Invalid value block reference.")
                return block(value["key"])
            return {key: expand(child) for key, child in value.items()}
        if type(value) is list:
            return [expand(child) for child in value]
        return value
    for table_key, table in tables.items():
        rows, overlays, built, active = table["rows"], table["overlays"], {}, set()
        reverse = {}
        occurrence_tokens = [object() for _ in rows]
        used_occurrences = set()
        used_structures = {slot: set() for slot in range(len(rows))}
        restored_values = {}

        def build(slot):
            if type(slot) is not int or slot < 0 or slot >= len(overlays):
                raise NodeCodecError("Undeclared Node slot.")
            if slot in active:
                raise NodeCodecError("Cyclic Node block reference.")
            if slot in built:
                return built[slot]
            active.add(slot)
            record = overlays[slot]
            if type(record) is not dict or set(record) != {"occurrence", "value", "structure", "construction"}:
                raise NodeCodecError("Invalid realization overlay.")
            occurrence = record["occurrence"]
            if type(occurrence) is not int or occurrence < 0 or occurrence >= len(rows):
                raise NodeCodecError("Invalid occurrence slot.")
            used_occurrences.add(occurrence)
            base = rows[occurrence]
            if (type(base) is not dict or set(base) != {"structures"} or
                    type(base["structures"]) is not list or not base["structures"] or
                    len(set(base["structures"])) != len(base["structures"]) or
                    record["structure"] not in base["structures"]):
                raise NodeCodecError("Invalid occurrence row.")
            used_structures[occurrence].add(record["structure"])
            row = block(record["value"])
            if type(row) is not dict or set(row) != {"kind", "cls", "args", "kwargs", "stateful_role", "identity", "state"}:
                raise NodeCodecError("Invalid occurrence value fields.")
            value = lambda item: _restore(item, build, restored_values)
            cls, kwargs = value(row["cls"]), value(row["kwargs"])
            if not isinstance(kwargs, FrozenDict):
                raise NodeCodecError("Node constructor fields must be a mapping.")
            if row["kind"] == "concrete":
                if row["args"] is not None or type(row["stateful_role"]) is not bool:
                    raise NodeCodecError("Invalid concrete Node record.")
                result = ConcreteNode._from_bound_record(cls, BoundArguments(kwargs.items()), stateful_role=row["stateful_role"])
                object.__setattr__(result, "_node_id", result._occurrence)
            elif row["kind"] == "soft":
                if row["stateful_role"] is not None:
                    raise NodeCodecError("Invalid soft Node record.")
                result = Definition._from_symbolic_parts(cls, None if row["args"] is None else value(row["args"]), kwargs)
            else:
                raise NodeCodecError("Unknown Node kind.")
            object.__setattr__(result, "_occurrence", occurrence_tokens[occurrence])
            if isinstance(result, ConcreteNode):
                object.__setattr__(result, "_node_id", occurrence_tokens[occurrence])
            identity, state = value(row["identity"]), value(row["state"])
            if identity is not None and not isinstance(identity, ObjectId) or state is not None and not isinstance(state, LocalState):
                raise NodeCodecError("Invalid exact Node facets.")
            object.__setattr__(result, "_facets", _Facets(identity, state))
            construction = record["construction"]
            if construction is not None:
                if type(construction) is not dict or set(construction) != {"slot", "view"} or construction["view"] != "construction":
                    raise NodeCodecError("Invalid retained construction view kind.")
                target = build(construction["slot"])
                if target._occurrence is not occurrence_tokens[occurrence]:
                    raise NodeCodecError("Retained construction handle selects a different occurrence.")
                object.__setattr__(result, "_construction", target)
            # Re-description is the closed grammar validator, not digest equality.
            described = _describe_node(result, lambda node: reverse[id(node)])
            if _canonical_values([described], lambda label: label, semantic=False) != _canonical_values([row], lambda label: label, semantic=False):
                raise NodeCodecError("Noncanonical or unsupported Node value fields.")
            structural = {name: value for name, value in row.items() if name not in {"identity", "state"}}
            structural = _map(structural, lambda child: {"tag": "node", "slot": overlays[child]["occurrence"]})
            if _canonical_values([structural], lambda label: label, semantic=False) != _canonical_values([block(record["structure"])], lambda label: label, semantic=False):
                raise NodeCodecError("Overlay disagrees with authenticated occurrence structure.")
            built[slot] = result
            reverse[id(result)] = slot
            active.remove(slot)
            return result

        for root in table["roots"]:
            build(root)
        if len(built) != len(overlays) or len(used_occurrences) != len(rows):
            raise NodeCodecError("Unreachable Node rows or overlays.")
        if any(used_structures[slot] != set(row["structures"]) for slot, row in enumerate(rows)):
            raise NodeCodecError("Unreachable construction structures.")
        built_tables[table_key] = built
    if used_values != set(values):
        raise NodeCodecError("Unreachable value blocks.")
    # Validate every advertised key; unrelated storage roots do not enter any
    # selected semantic preimage. No policy is inferred from an untrusted name.
    from .node_prepare import PreparationView
    groups = {}
    for root_view in packet["views"]:
        slot = root_view["slot"]
        if type(slot) is not int or slot < 0:
            raise NodeCodecError("Invalid root slot.")
        root = built_tables[root_view["table"]][slot]
        groups.setdefault(root_view["view"], []).append((root_view, root))
    for policy, selections in groups.items():
        prepared = NodePreparation((root for _, root in selections), view=PreparationView(policy))
        for root_view, root in selections:
            if prepared.exact_key(root).decode("ascii") != root_view["expected_key"]:
                raise NodeCodecError("Root view semantic key mismatch.")
    if _all_views:
        return tuple(built_tables[item["table"]][item["slot"]] for item in packet["views"])
    return built_tables[chosen["table"]][chosen["slot"]]
