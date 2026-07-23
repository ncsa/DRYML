"""Probe-report record helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.records import LocatedRecordRef, RecordStoreIO, RecordValidationError, make_record, validate_record


from .errors import ProviderReportError
from .reports import ProbeReport


def make_probe_report_record(report: ProbeReport, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a generic ``kind='probe_report'`` record envelope."""

    return make_record(kind="probe_report", payload=report.to_data(), metadata=metadata)


def validate_probe_report_record(record: Mapping[str, Any]) -> ProbeReport:
    """Validate a probe-report record and return its payload model."""

    try:
        validate_record(record, kind="probe_report")
    except RecordValidationError as exc:
        raise ProviderReportError("invalid probe report record", context=exc.context) from exc
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise ProviderReportError("probe report record payload must be a mapping")
    report = ProbeReport.from_data(payload)
    if isinstance(record.get("id"), str):
        return ProbeReport.from_data({**report.to_data(), "report_id": record["id"]})
    return report


def probe_report_from_record(record: Mapping[str, Any]) -> ProbeReport:
    """Return a ``ProbeReport`` from a validated record envelope."""

    return validate_probe_report_record(record)


def write_probe_report(store_io: RecordStoreIO, report: ProbeReport, *, overwrite: bool = False) -> LocatedRecordRef:
    """Write a probe report through ``RecordStoreIO``."""

    return store_io.write_record(make_probe_report_record(report), overwrite=overwrite)


__all__ = ["make_probe_report_record", "probe_report_from_record", "validate_probe_report_record", "write_probe_report"]
