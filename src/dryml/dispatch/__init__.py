"""Metadata-only dispatch request and execution recipe APIs."""

from .errors import DispatchSpecError
from .fake import fake_execution_record
from .links import EXECUTION_KINDS, EXECUTION_STATUSES, normalize_backend_identity, normalize_execution_kind, normalize_execution_status
from .recipes import (
    EXECUTION_RECIPE_KIND,
    EXECUTION_RECIPE_SCHEMA,
    EXECUTION_RECIPE_SCHEMA_VERSION,
    EXECUTION_RECIPE_SPEC_FAMILY,
    attach_recipe_id,
    compute_recipe_id,
    make_execution_recipe,
    recipe_payload_for_id,
    validate_execution_recipe,
)
from .specs import (
    DISPATCH_KIND,
    DISPATCH_SCHEMA,
    DISPATCH_SCHEMA_VERSION,
    DISPATCH_SPEC_FAMILY,
    attach_dispatch_id,
    compute_dispatch_id,
    dispatch_payload_for_id,
    make_dispatch_spec,
    validate_dispatch_spec,
)


__all__ = [
    "DISPATCH_KIND",
    "DISPATCH_SCHEMA",
    "DISPATCH_SCHEMA_VERSION",
    "DISPATCH_SPEC_FAMILY",
    "EXECUTION_KINDS",
    "EXECUTION_RECIPE_KIND",
    "EXECUTION_RECIPE_SCHEMA",
    "EXECUTION_RECIPE_SCHEMA_VERSION",
    "EXECUTION_RECIPE_SPEC_FAMILY",
    "EXECUTION_STATUSES",
    "DispatchSpecError",
    "attach_dispatch_id",
    "attach_recipe_id",
    "compute_dispatch_id",
    "compute_recipe_id",
    "dispatch_payload_for_id",
    "fake_execution_record",
    "make_dispatch_spec",
    "make_execution_recipe",
    "normalize_backend_identity",
    "normalize_execution_kind",
    "normalize_execution_status",
    "recipe_payload_for_id",
    "validate_dispatch_spec",
    "validate_execution_recipe",
]
