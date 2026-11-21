"""Bounded record-local sharing of exact managed Node value-block syntax.

The optional wire envelope pools repeated large blocks, not Nodes or semantic
identities. Graph tables, views, bindings and block keys remain unchanged. Nested
wire envelopes retain independent reference scopes. No Store or payload IO occurs.
"""

import json
import re

from dryml.formats import canonical_json_bytes, canonical_json_load_bytes
from dryml.formats.canonical import json_ready


SCHEMA = "dryml-managed-control-blocks"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_BOUNDS = dict(max_depth=96, max_nodes=65536, max_entries=16384, max_string=1024 * 1024, max_int_bits=64)


class ControlSizeError(ValueError):
    """Stored wire or expanded logical control metadata exceeds its byte budget."""


def _measure(data, maximum, *, exact=False):
    """Bound expanded JSON bytes/nodes/depth without serializing repeated blocks."""
    memo = {}

    def string_size(value):
        # Six bytes per code point covers every UTF-8 character and JSON escape.
        # Most controls fit even this upper bound, avoiding repeated UTF-8 copies.
        return len(json.encoder.encode_basestring(value).encode("utf8")) if exact else 2 + 6 * len(value)

    def visit(value):
        marker = id(value)
        if marker in memo:
            return memo[marker]
        if type(value) in (dict, list):
            if type(value) is dict:
                children = value.values()
                size = 2 + max(0, len(value) - 1) + sum(
                    string_size(key) + 1 for key in value)
            else:
                children = value
                size = 2 + max(0, len(value) - 1)
            nodes, depth = 1, 0
            for child in children:
                child_size, child_nodes, child_depth = visit(child)
                size += child_size
                nodes += child_nodes
                depth = max(depth, child_depth + 1)
                if size > maximum:
                    raise ControlSizeError("Expanded control metadata exceeds its byte budget.")
                if nodes > _BOUNDS["max_nodes"] or depth > _BOUNDS["max_depth"]:
                    raise ValueError("Expanded control metadata exceeds its grammar budget.")
        else:
            if type(value) is str:
                size = string_size(value)
            elif value is None or type(value) is bool:
                size = 5 if value is False else 4
            elif type(value) is int:
                size = len(str(value))
            else:
                size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf8"))
            nodes, depth = 1, 0
        if size > maximum:
            raise ControlSizeError("Expanded control metadata exceeds its byte budget.")
        memo[marker] = size, nodes, depth
        return memo[marker]
    try:
        return visit(data)[0]
    except ControlSizeError:
        if exact:
            raise
        return _measure(data, maximum, exact=True)


def check_expanded(data, *, max_bytes):
    """Validate a logical record with the existing grammar and an 8x wire budget.

    Args:
        data: Closed JSON-compatible metadata; no native payload values.
        max_bytes: Stored-byte budget selected by the managed control owner.

    Returns:
        Detached mutable JSON data, bounded to eight times max_bytes and the
        existing depth/node/entry/string/integer grammar.

    Raises:
        TypeError, ValueError: For malformed metadata or grammar bounds.
        ControlSizeError: For excessive expanded bytes. No IO is performed.
    """
    result = json_ready(data, **_BOUNDS)
    _measure(result, max_bytes * 8)
    return result


def _walk(data, group):
    if type(data) is list:
        return [_walk(value, group) for value in data]
    if type(data) is not dict:
        return data
    if data.get("format") == "dryml-node-graph/v1" and type(data.get("values")) is dict:
        return {**data, "values": group(data["values"])}
    if data.get("schema") == SCHEMA:
        if set(data) != {"schema", "version", "record", "blocks"} or type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("Unsupported nested managed control block envelope.")
        # The inner record's references belong to its own pool, not this one.
        return {**data, "blocks": group(data["blocks"])}
    return {key: _walk(value, group) for key, value in data.items()}


def encode_control(data, *, max_bytes, shared=True, validated=False):
    """Encode bounded canonical JSON, optionally pooling repeated graph blocks.

    Args:
        data: Logical control or preparation wire mapping.
        max_bytes: Stored byte limit. Expanded metadata is separately bounded.
        shared: False keeps legacy plain canonical JSON, for exact packet evidence.
        validated: Internal proof that data is already detached and checked by
            check_expanded with this budget; avoids validating a snapshot again
            after its enclosing logical preparation has been checked.

    Returns:
        bytes. Records at most 256 KiB retain plain encoding. Larger records use
        a closed version-1 pool only when it saves bytes. Exact syntax, not a
        digest alone, determines whether same-key blocks can share storage.

    Raises:
        ValueError, TypeError: For malformed data or conflicting block content.
        ControlSizeError: If no admitted encoding fits. No IO or mutation occurs.
    """
    if not validated:
        data = check_expanded(data, max_bytes=max_bytes)
    plain = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf8")
    if shared and len(plain) > 256 * 1024:
        counts, blocks, encoded = {}, {}, {}
        def gather(values):
            if type(values) is not dict:
                raise ValueError("Invalid control value-block group.")
            for key, value in values.items():
                if type(value) is not dict:
                    raise ValueError("Unscoped control value-block reference.")
                if type(key) is not str or not _HEX.fullmatch(key):
                    raise ValueError("Invalid control value-block key.")
                raw = canonical_json_bytes(value, **_BOUNDS)
                if key in encoded and encoded[key] != raw:
                    raise ValueError("Conflicting same-key graph value blocks.")
                encoded[key], blocks[key] = raw, value
                counts[key] = counts.get(key, 0) + 1
            return values
        _walk(data, gather)
        selected = {key: blocks[key] for key, count in counts.items() if count > 1 and len(encoded[key]) >= 256}
        if selected:
            record = _walk(data, lambda values: {key: key if key in selected and type(value) is dict else value
                                                for key, value in values.items()})
            wire = {"schema": SCHEMA, "version": 1, "record": record, "blocks": selected}
            compact = canonical_json_bytes(wire, **_BOUNDS)
            if len(compact) < len(plain):
                plain = compact
    if len(plain) > max_bytes:
        raise ControlSizeError("Stored control metadata exceeds its byte budget.")
    return plain


def decode_control(payload, *, max_bytes, validated=False):
    """Return fresh logical JSON from plain or shared control wire bytes.

    Args:
        payload: Exact bytes, at most max_bytes.
        max_bytes: Owner's stored-byte bound.
        validated: Internal exact-byte memo proof permits skipping repeated JSON
            grammar/expanded-budget validation. Reference scopes still decode anew.

    Returns:
        A new mutable tree; references share only detached value-block syntax.

    Raises:
        ValueError, TypeError: For duplicate keys, malformed pools/references,
            unsupported versions or unused blocks. ControlSizeError bounds stored
            and expanded bytes before any large expanded serialization. No IO,
            graph materialization, digest-only merging or authority checks occur.
    """
    if type(payload) is not bytes or len(payload) > max_bytes:
        raise ControlSizeError("Stored control metadata exceeds its byte budget.")
    if not validated:
        canonical_json_load_bytes(payload, **_BOUNDS)
    data = json.loads(payload)
    if type(data) is dict and data.get("schema") == SCHEMA:
        if (set(data) != {"schema", "version", "record", "blocks"}
                or type(data["version"]) is not int or data["version"] != 1
                or type(data["blocks"]) is not dict or not data["blocks"]):
            raise ValueError("Unsupported managed control block envelope.")
        blocks, used = data["blocks"], set()
        if any(type(key) is not str or not _HEX.fullmatch(key) or type(value) is not dict for key, value in blocks.items()):
            raise ValueError("Invalid managed control value blocks.")
        def restore(values):
            if type(values) is not dict:
                raise ValueError("Invalid control value-block group.")
            result = {}
            for key, value in values.items():
                if type(value) is str:
                    if value != key or value not in blocks:
                        raise ValueError("Invalid managed control block reference.")
                    used.add(value)
                    value = blocks[value]
                result[key] = value
            return result
        data = _walk(data["record"], restore)
        if used != blocks.keys():
            raise ValueError("Unused managed control value blocks.")
        if not validated:
            _measure(data, max_bytes * 8)
    return data
