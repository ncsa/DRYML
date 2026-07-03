"""Dependency-light record/spec sidecar APIs for DRYML stores."""

from .errors import (
    RecordError,
    RecordIOError,
    RecordNotFoundError,
    RecordValidationError,
    SpecNotFoundError,
    SpecValidationError,
    StorageRefError,
)
from .records import (
    RECORD_ID_PREFIX,
    RECORD_SCHEMA,
    RECORD_SCHEMA_VERSION,
    attach_record_id,
    compute_record_id,
    make_record,
    record_payload_for_id,
    validate_record,
)
from .refs import LocatedRecordRef, LocatedSpecRef, RecordRef, SpecRef
from .specs import (
    SPEC_FAMILIES,
    attach_spec_id,
    compute_spec_id,
    make_spec,
    spec_dir_name,
    spec_family_for_id,
    spec_id_prefix,
    spec_payload_for_id,
    validate_spec,
)
from .storage import StorageRef
from .store import RecordStoreIO


__all__ = [
    "RECORD_ID_PREFIX",
    "RECORD_SCHEMA",
    "RECORD_SCHEMA_VERSION",
    "SPEC_FAMILIES",
    "LocatedRecordRef",
    "LocatedSpecRef",
    "RecordError",
    "RecordIOError",
    "RecordNotFoundError",
    "RecordRef",
    "RecordStoreIO",
    "RecordValidationError",
    "SpecNotFoundError",
    "SpecRef",
    "SpecValidationError",
    "StorageRef",
    "StorageRefError",
    "attach_record_id",
    "attach_spec_id",
    "compute_record_id",
    "compute_spec_id",
    "make_record",
    "make_spec",
    "record_payload_for_id",
    "spec_dir_name",
    "spec_family_for_id",
    "spec_id_prefix",
    "spec_payload_for_id",
    "validate_record",
    "validate_spec",
]
