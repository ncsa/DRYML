"""Closed v1.1 environment family identity constants.

Generic source-v1 ID builders are intentionally absent. Typed environment
values compute IDs through :mod:`dryml.formats` after domain normalization.
"""

ENVIRONMENT_RECORD_ID_PREFIX = "envrec"
ENVIRONMENT_REQUIREMENT_ID_PREFIX = "envreq"
ENVIRONMENT_SPEC_ID_PREFIX = "envspec"
ENVIRONMENT_LOCK_ID_PREFIX = "envlock"

__all__ = [
    "ENVIRONMENT_LOCK_ID_PREFIX",
    "ENVIRONMENT_RECORD_ID_PREFIX",
    "ENVIRONMENT_REQUIREMENT_ID_PREFIX",
    "ENVIRONMENT_SPEC_ID_PREFIX",
]
