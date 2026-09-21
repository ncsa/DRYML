"""Dependency-light generic envelopes for DRYML sidecar records.

This package intentionally validates only the common record wrapper.  Record
owners validate their kinds and domain payloads without creating dependencies
from this foundation back to core or environment packages.
"""

from .records import GENERIC_RECORD_SCHEMA, GenericRecord, decode_record, encode_record

__all__ = ["GENERIC_RECORD_SCHEMA", "GenericRecord", "decode_record", "encode_record"]
