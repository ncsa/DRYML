"""Bounded data-only managed result v1, separate from arguments and transport.

Exact builtins, acyclic containers, nonobject NumPy and explicitly owned immutable
metadata only. No import resolution, materialization, pickle or custom fallback.
"""

import base64
import json
import math
import struct

from dryml.formats import canonical_json_bytes, canonical_json_load_bytes
from .errors import ManagedResultError

RESULT_VERSION = 1
MAX_RESULT_BYTES = 8 * 1024 * 1024
_MAX_NODES = 65536
_MAX_DEPTH = 80


def _fail():
    raise ManagedResultError("invalid_result", "unsupported or malformed durable result")


def _budget(max_bytes):
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_RESULT_BYTES:
        _fail()


def _json(value):
    return canonical_json_bytes(value, max_depth=96, max_nodes=1048576,
        max_entries=262144, max_string=MAX_RESULT_BYTES, max_int_bits=64)


def encode_result(value, *, max_bytes=MAX_RESULT_BYTES):
    """Encode a supported value into bounded v1 bytes preserving DAG aliases.

    value follows the module's closed grammar; max_bytes is a positive int no
    greater than 8 MiB. Returns bytes. Unsupported types/subclasses/cycles and
    exceeded byte/depth/node bounds raise ManagedResultError before publication.
    Metadata codecs are inert; this function performs no IO or user callbacks.
    """
    _budget(max_bytes)
    import numpy as np
    from dryml.core.definition import Definition, ConcreteDefinition, ConcreteNode
    from dryml.core.state_values import StateKey, LocalState, IdentitySpec, StateSpec
    from dryml.core.reference_values import ObjectId
    from dryml.core.node_codec import encode_node_graph, _dtype, _logical_bytes
    from dryml.core.node_prepare import NodePreparation
    memo, active, nodes = {}, set(), []
    charged = 0

    def visit(item, depth=0):
        nonlocal charged
        if depth > _MAX_DEPTH or len(nodes) >= _MAX_NODES or id(item) in active:
            _fail()
        if id(item) in memo:
            return memo[id(item)][1]
        active.add(id(item))
        kind = type(item)
        child = lambda v: visit(v, depth + 1)
        if item is None: row = ["none"]
        elif kind is bool: row = ["bool", item]
        elif kind is int:
            if item.bit_length() > 262144: _fail()
            row = ["int", ("-" if item < 0 else "") + format(abs(item), "x")]
        elif kind is float: row = ["float", struct.pack("!d", item).hex()]
        elif kind is complex: row = ["complex", struct.pack("!dd", item.real, item.imag).hex()]
        elif kind is str:
            if len(item) > max_bytes: _fail()
            row = ["str", item]
        elif kind is bytes:
            if len(item) > max_bytes: _fail()
            row = ["bytes", base64.b64encode(item).decode("ascii")]
        elif kind in (list, tuple, set, frozenset):
            if len(item) > min(_MAX_NODES, max_bytes): _fail()
            row = [kind.__name__, [child(v) for v in item]]
        elif kind is dict:
            if len(item) > min(_MAX_NODES, max_bytes): _fail()
            row = ["dict", [[child(k), child(v)] for k, v in item.items()]]
        elif kind in (IdentitySpec, StateSpec):
            row = [kind.__name__, child(item.root)]
        elif kind in (Definition, ConcreteDefinition, ConcreteNode, StateKey, LocalState, ObjectId):
            # A closed wrapper delegates immutable metadata to its owning codec.
            wrapper = Definition._from_prepared_parameters(None, {"value": item})
            row = ["metadata", encode_node_graph(NodePreparation((wrapper,))) ]
        elif kind is np.ndarray or isinstance(item, np.generic) and kind is item.dtype.type:
            if item.dtype.hasobject: _fail()
            array = item if kind is np.ndarray else np.ndarray((), dtype=item.dtype, buffer=item.tobytes())
            if array.nbytes > max_bytes: _fail()
            row = ["array" if kind is np.ndarray else "numpy", _dtype(array.dtype), list(array.shape),
                   base64.b64encode(_logical_bytes(array)).decode("ascii")]
        else: _fail()
        active.remove(id(item))
        charged += len(_json(row))
        if charged > max_bytes: _fail()
        index = len(nodes)
        nodes.append(row)
        memo[id(item)] = item, index
        return index
    try:
        root = visit(value)
        payload = _json({"schema": "dryml-managed-result", "version": RESULT_VERSION, "nodes": nodes, "root": root})
        if len(payload) > max_bytes: _fail()
        return payload
    except ManagedResultError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise ManagedResultError("invalid_result", "durable result encoding refused") from error


def decode_result(payload, *, max_bytes=MAX_RESULT_BYTES):
    """Decode bounded closed v1 bytes, preserving tags, aliases and bit patterns.

    payload must be exact bytes within max_bytes (positive int, at most 8 MiB).
    Returns a fresh supported value; invalid versions/tags/forward references,
    cycles, unreachable nodes, sizes or metadata raise ManagedResultError. No
    source IO, symbol resolution, user hook or receiver materialization occurs.
    """
    _budget(max_bytes)
    if type(payload) is not bytes or len(payload) > max_bytes: _fail()
    import numpy as np
    from dryml.core.node_codec import decode_node_graph, _dtype, _restore_dtype
    from dryml.core.state_values import IdentitySpec, StateSpec
    try:
        canonical_json_load_bytes(payload, max_depth=96, max_nodes=1048576,
            max_entries=262144, max_string=MAX_RESULT_BYTES, max_int_bits=64)
        packet = json.loads(payload)
        if (type(packet) is not dict or set(packet) != {"schema", "version", "nodes", "root"}
                or packet["schema"] != "dryml-managed-result" or type(packet["version"]) is not int
                or packet["version"] != RESULT_VERSION or type(packet["nodes"]) is not list
                or not 1 <= len(packet["nodes"]) <= _MAX_NODES): _fail()
        values, depths, edges = [], [], []
        for index, row in enumerate(packet["nodes"]):
            if type(row) is not list or not row or type(row[0]) is not str: _fail()
            tag, refs = row[0], []
            def ref(slot):
                if type(slot) is not int or not 0 <= slot < index: _fail()
                refs.append(slot)
                return values[slot]
            if tag == "none" and len(row) == 1: value = None
            elif tag == "bool" and len(row) == 2 and type(row[1]) is bool: value = row[1]
            elif tag == "int" and len(row) == 2 and type(row[1]) is str:
                import re
                if not re.fullmatch(r"-?(?:0|[1-9a-f][0-9a-f]*)", row[1]) or len(row[1]) > 65537: _fail()
                value = int(row[1], 16)
            elif tag in ("float", "complex") and len(row) == 2 and type(row[1]) is str:
                raw = bytes.fromhex(row[1])
                value = struct.unpack("!d", raw)[0] if tag == "float" else complex(*struct.unpack("!dd", raw))
            elif tag == "str" and len(row) == 2 and type(row[1]) is str: value = row[1]
            elif tag == "bytes" and len(row) == 2 and type(row[1]) is str: value = base64.b64decode(row[1], validate=True)
            elif tag in ("list", "tuple", "set", "frozenset") and len(row) == 2 and type(row[1]) is list:
                items = [ref(v) for v in row[1]]
                value = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[tag](items)
                if len(value) != len(items): _fail()
            elif tag == "dict" and len(row) == 2 and type(row[1]) is list:
                value = {}
                for pair in row[1]:
                    if type(pair) is not list or len(pair) != 2: _fail()
                    key, val = ref(pair[0]), ref(pair[1])
                    if key in value: _fail()
                    value[key] = val
            elif tag in ("IdentitySpec", "StateSpec") and len(row) == 2:
                value = {"IdentitySpec": IdentitySpec, "StateSpec": StateSpec}[tag](ref(row[1]))
            elif tag == "metadata" and len(row) == 2:
                wrapper = decode_node_graph(row[1])
                if (wrapper.cls is not None or set(wrapper.parameters) != {"value"}
                        or wrapper.object_id is not None or wrapper.state is not None): _fail()
                value = wrapper.parameters["value"]
                # Refuse broader constructor codecs at the result boundary.
                from dryml.core.definition import Definition, ConcreteDefinition, ConcreteNode
                from dryml.core.state_values import StateKey, LocalState
                from dryml.core.reference_values import ObjectId
                if type(value) not in (Definition, ConcreteDefinition, ConcreteNode, StateKey, LocalState, ObjectId): _fail()
            elif tag in ("array", "numpy") and len(row) == 4:
                dtype = _restore_dtype(row[1])
                if dtype.hasobject or _dtype(dtype) != row[1] or dtype.itemsize > max_bytes: _fail()
                shape = row[2]
                if type(shape) is not list or len(shape) > 32 or any(type(v) is not int or not 0 <= v <= max_bytes for v in shape): _fail()
                if tag == "numpy" and shape != []: _fail()
                size = math.prod(shape) * dtype.itemsize
                raw = base64.b64decode(row[3], validate=True)
                if size > max_bytes or size != len(raw): _fail()
                value = np.ndarray(tuple(shape), dtype=dtype, buffer=raw).copy()
                if tag == "numpy": value = value[()]
            else: _fail()
            depth = 1 + max((depths[v] for v in refs), default=0)
            if depth > _MAX_DEPTH + 1: _fail()
            values.append(value); depths.append(depth); edges.append(refs)
        root = packet["root"]
        if type(root) is not int or not 0 <= root < len(values): _fail()
        reachable, pending = set(), [root]
        while pending:
            slot = pending.pop()
            if slot not in reachable:
                reachable.add(slot); pending.extend(edges[slot])
        if len(reachable) != len(values): _fail()
        return values[root]
    except ManagedResultError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError, struct.error) as error:
        raise ManagedResultError("invalid_result", "durable result decoding refused") from error
