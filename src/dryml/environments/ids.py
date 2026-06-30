"""Content-addressed ID utilities for environment metadata."""

from __future__ import annotations

import hashlib
from typing import Any

from .serialization import canonical_json_bytes


def stable_hash(data: Any) -> str:
    """Return a SHA-256 hex digest over canonical JSON data."""

    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def content_id(prefix: str, schema_version: int, data: Any) -> str:
    """Return a namespaced content ID including schema namespace and version."""

    payload = {
        "id_prefix": prefix,
        "schema_version": schema_version,
        "data": data,
    }
    return f"{prefix}-v{schema_version}-{stable_hash(payload)}"


__all__ = ["stable_hash", "content_id"]
