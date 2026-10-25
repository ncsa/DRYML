"""Bounded portable encoding for inert symbolic Node expression data.

The expression payload reuses the closed graph machinery introduced for the
retired Template carrier, but names its own schema so persisted Definition
quotation is not a runtime Template value.
Node-only values explicitly select ``dryml-node-template`` v1 with an authenticated
``dryml-node-graph/v1`` packet. Compatible old expressions retain their own v1
schema; the legacy reader is not extended to reinterpret identity-3 fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ParameterizationError

SCHEMA = "dryml-definition-expression"
VERSION = 1
_KIND = "definition-expression"


def to_data(value: Any) -> dict[str, object]:
    """Encode inert Node expression data without resolving ordinary symbols.

    Args:
        value: A frozen expression graph with supported aliases, exact Node facets,
            retained construction views, roles, and immutable semantic values.

    Returns:
        A bounded canonical v1 graph record, in the Node family when required.

    Raises:
        ParameterizationError: If the value is unsupported, cyclic, or exceeds
            the shared expression limits.
    """

    from .template_codec import _encode, _encode_node_value, _needs_node_codec

    if _needs_node_codec(value):
        return _encode_node_value(_KIND, value)
    data = _encode(_KIND, value)
    data["schema"] = SCHEMA
    return data


def from_data(data: Mapping[str, object]) -> Any:
    """Decode a quoted expression graph without recipe resolution or construction.

    Args:
        data: A record produced by :func:`to_data`.

    Returns:
        The frozen symbolic value with graph aliases, facets and construction
        evidence restored. Handles are fresh; occurrence equality with the source
        is not claimed. Ordinary recipe symbols remain inert; explicit enum or
        extension-dtype values follow their owning lazy type codecs.

    Raises:
        ParameterizationError: If the payload is malformed, noncanonical, or
            exceeds the expression limits.
    """

    from .template_codec import TEMPLATE_CODEC_SCHEMA, NODE_TEMPLATE_SCHEMA, _decode, _decode_node_value

    if isinstance(data, Mapping) and data.get("schema") == NODE_TEMPLATE_SCHEMA:
        return _decode_node_value(data, _KIND)
    if not isinstance(data, Mapping):
        raise ParameterizationError("Definition expression payload must be a mapping.")
    if data.get("schema") != SCHEMA or data.get("version") != VERSION or data.get("kind") != _KIND:
        raise ParameterizationError("Definition expression payload schema or version is invalid.")
    legacy = dict(data)
    legacy["schema"] = TEMPLATE_CODEC_SCHEMA
    kind, value = _decode(legacy)
    if kind != _KIND:
        raise ParameterizationError("Definition expression payload kind is invalid.")
    return value
