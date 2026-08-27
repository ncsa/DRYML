"""Domain-separated content IDs for closed v1.1 metadata families."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .canonical import canonical_json_bytes
from .errors import ContentIDError

CONTRACT_VERSION = "1.1"
_ID = re.compile(r"^(?P<prefix>[a-z][a-z0-9_]*)-v1\.1-(?P<digest>[0-9a-f]{64})$")


def semantic_id(prefix: str, schema: str, kind: str, identifying_payload: Any, **bounds: Any) -> str:
    """Return the v1.1 ID for one family identifying projection."""

    if not isinstance(prefix, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", prefix):
        raise ContentIDError("invalid v1.1 ID prefix", context={"prefix": prefix})
    preimage = {"prefix": prefix, "contract_version": CONTRACT_VERSION, "schema": schema, "kind": kind, "payload": identifying_payload}
    return f"{prefix}-v1.1-{hashlib.sha256(canonical_json_bytes(preimage, **bounds)).hexdigest()}"


def verify_semantic_id(value: str, *, prefix: str, schema: str, kind: str, identifying_payload: Any, **bounds: Any) -> None:
    """Validate ID grammar, family prefix, and digest against its projection."""

    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ContentIDError("invalid v1.1 semantic ID", context={"prefix": prefix})
    if not value.startswith(f"{prefix}-v1.1-"):
        raise ContentIDError("semantic ID belongs to another family", context={"expected_prefix": prefix})
    expected = semantic_id(prefix, schema, kind, identifying_payload, **bounds)
    if value != expected:
        raise ContentIDError("semantic ID does not match payload", context={"expected": expected, "observed": value})


__all__ = ["CONTRACT_VERSION", "semantic_id", "verify_semantic_id"]
