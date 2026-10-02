"""Bounded portable encoding for quoted symbolic Definition expression data.

The expression payload reuses the closed graph machinery introduced for the
retired Template carrier, but names its own schema so persisted Definition
quotation is not a runtime Template value.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ParameterizationError

SCHEMA = "dryml-definition-expression"
VERSION = 1
_KIND = "definition-expression"


def to_data(value: Any) -> dict[str, object]:
    """Encode quoted Definition expression data without resolving symbols.

    Args:
        value: A frozen Definition expression graph with supported aliases.

    Returns:
        A bounded canonical versioned data-only graph record.

    Raises:
        ParameterizationError: If the value is unsupported, cyclic, or exceeds
            the shared expression limits.
    """

    from .template_codec import _encode

    data = _encode(_KIND, value)
    data["schema"] = SCHEMA
    return data


def from_data(data: Mapping[str, object]) -> Any:
    """Decode a quoted expression graph without imports or construction.

    Args:
        data: A record produced by :func:`to_data`.

    Returns:
        The frozen symbolic value with graph aliases restored.

    Raises:
        ParameterizationError: If the payload is malformed, noncanonical, or
            exceeds the expression limits.
    """

    from .template_codec import TEMPLATE_CODEC_SCHEMA, _decode

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
