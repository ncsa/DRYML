"""Explicit probe cache keys and lightweight cache helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dryml.formats import json_ready
from dryml.formats.ids import stable_hash

from .records import probe_report_from_record
from .reports import ProbeReport


@dataclass(frozen=True, slots=True)
class ProbeCacheKey:
    """Exact-match cache key for provider probe reports."""

    request_kind: str
    operation_id: str | None
    environment_spec_id: str | None
    provider_id: str
    runtime_id: str | None
    probe_policy_hash: str
    provider_options_hash: str

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready key data."""

        return {
            "request_kind": self.request_kind,
            "operation_id": self.operation_id,
            "environment_spec_id": self.environment_spec_id,
            "provider_id": self.provider_id,
            "runtime_id": self.runtime_id,
            "probe_policy_hash": self.probe_policy_hash,
            "provider_options_hash": self.provider_options_hash,
        }


class ProbeCache:
    """In-memory explicit probe cache."""

    def __init__(self) -> None:
        self._reports: dict[ProbeCacheKey, ProbeReport] = {}

    def get(self, key: ProbeCacheKey) -> ProbeReport | None:
        """Return a cached report for an exact key, if present."""

        return self._reports.get(key)

    def put(self, key: ProbeCacheKey, report: ProbeReport) -> None:
        """Store a report under an exact key."""

        self._reports[key] = report


def hash_json_payload(payload: Mapping[str, Any] | None) -> str:
    """Return a stable hash for a JSON-ready mapping payload."""

    return stable_hash(json_ready(payload or {}))


def key_for_report(report: ProbeReport, *, provider_id: str, provider_options: Mapping[str, Any] | None = None) -> ProbeCacheKey:
    """Build a cache key for one provider identity in a probe report."""

    request = dict(json_ready(report.request or {}))
    return ProbeCacheKey(
        request_kind=str(request.get("request_kind")),
        operation_id=report.operation_id,
        environment_spec_id=report.environment_spec_id,
        provider_id=provider_id,
        runtime_id=report.runtime_id,
        probe_policy_hash=hash_json_payload(report.probe_policy),
        provider_options_hash=hash_json_payload(provider_options or request.get("provider_options") or {}),
    )


def lookup_store_probe_report(store_io, key: ProbeCacheKey) -> ProbeReport | None:
    """Scan store records for a probe report matching an exact cache key."""

    for record in store_io.iter_records():
        if record.get("kind") != "probe_report":
            continue
        report = probe_report_from_record(record)
        for provider_report in report.reports:
            if key_for_report(report, provider_id=provider_report.provider_identity.id) == key:
                return report
    return None


__all__ = ["ProbeCache", "ProbeCacheKey", "hash_json_payload", "key_for_report", "lookup_store_probe_report"]
