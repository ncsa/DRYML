"""Dependency-light operation spec APIs."""

from .errors import OperationError, OperationResolutionError, OperationSpecError
from .resolver import CDefRefArg, MaterializeCDefArg, ResolvedOperationCall, plan_call_resolution, resolve_call_arguments
from .specs import (
    OPERATION_KINDS,
    OPERATION_SCHEMA,
    OPERATION_SCHEMA_VERSION,
    OPERATION_SPEC_FAMILY,
    attach_operation_id,
    compute_operation_id,
    make_function_call_spec,
    make_method_call_spec,
    operation_payload_for_id,
    validate_operation_spec,
)

__all__ = [
    "CDefRefArg",
    "MaterializeCDefArg",
    "OPERATION_KINDS",
    "OPERATION_SCHEMA",
    "OPERATION_SCHEMA_VERSION",
    "OPERATION_SPEC_FAMILY",
    "OperationError",
    "OperationResolutionError",
    "OperationSpecError",
    "ResolvedOperationCall",
    "attach_operation_id",
    "compute_operation_id",
    "make_function_call_spec",
    "make_method_call_spec",
    "operation_payload_for_id",
    "plan_call_resolution",
    "resolve_call_arguments",
    "validate_operation_spec",
]
