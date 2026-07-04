"""Scan-based state record resolution helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dryml.formats.errors import ReferenceParseError
from dryml.formats.refs import parse_cdef_id
from dryml import reporting

from .errors import RecordValidationError, SpecNotFoundError
from .refs import LocatedRecordRef
from .representations import RepresentationRequirement, RepresentationSpec, representation_satisfies
from .storage import StorageRef
from .typed import AdapterRecord, DataRecord, ProgramRecord, StoredStateRecord, TypedRecord, typed_record_from_envelope


ResolutionStatus = Literal["ok", "requires_adapter", "not_found", "ambiguous", "unsupported", "failed"]


@dataclass(frozen=True, slots=True)
class RecordResolutionIssue:
    """Structured issue emitted by state/adapter resolution."""

    code: str
    message: str
    record_id: str | None = None
    representation_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return JSON-ready issue data."""

        return {
            "code": self.code,
            "message": self.message,
            "record_id": self.record_id,
            "representation_id": self.representation_id,
        }


@dataclass(frozen=True, slots=True)
class RecordResolutionReport:
    """Structured resolution status and counters."""

    status: ResolutionStatus
    issues: tuple[RecordResolutionIssue, ...] = ()
    candidates_considered: int = 0
    adapter_steps_considered: int = 0

    def to_json(self) -> dict[str, Any]:
        """Return JSON-ready report data."""

        return {
            "status": self.status,
            "issues": [issue.to_json() for issue in self.issues],
            "candidates_considered": self.candidates_considered,
            "adapter_steps_considered": self.adapter_steps_considered,
        }


@dataclass(frozen=True, slots=True)
class LocatedTypedRecord:
    """A typed record plus the located reference for the sidecar copy."""

    ref: LocatedRecordRef
    record: TypedRecord


@dataclass(frozen=True, slots=True)
class RepresentationCandidate:
    """A typed record paired with its representation spec."""

    located: LocatedTypedRecord
    representation: RepresentationSpec
    store_index: int = 0


@dataclass(frozen=True, slots=True)
class StateResolutionRequest:
    """Input for stored-state resolution."""

    cdef_id: str
    requirement: RepresentationRequirement = RepresentationRequirement()

    def __post_init__(self) -> None:
        try:
            parse_cdef_id(self.cdef_id)
        except ReferenceParseError as exc:
            raise RecordValidationError("invalid state resolution CDef ID", context=exc.context) from exc
        if not isinstance(self.requirement, RepresentationRequirement):
            object.__setattr__(self, "requirement", RepresentationRequirement.from_json(self.requirement))


@dataclass(frozen=True, slots=True)
class StateResolutionResult:
    """Result for selecting or adapting a stored-state record."""

    status: ResolutionStatus
    selected: LocatedTypedRecord | None = None
    selected_representation: RepresentationSpec | None = None
    adapter_source: LocatedTypedRecord | None = None
    adapter_source_representation: RepresentationSpec | None = None
    adapter_plan: Any = None
    report: RecordResolutionReport = RecordResolutionReport("not_found")


def find_stored_state_records(repo: Any, cdef_id: str, representation: RepresentationRequirement | Mapping[str, Any] | None = None) -> tuple[LocatedTypedRecord, ...]:
    """Find stored-state records for a CDef by scanning authoritative JSON."""

    try:
        parse_cdef_id(cdef_id)
    except ReferenceParseError as exc:
        raise RecordValidationError("invalid CDef ID", context=exc.context) from exc
    req = representation if isinstance(representation, RepresentationRequirement) else RepresentationRequirement.from_json(representation)
    reporting.step("dryml.records.state.find", "Resolving stored state records", data={"cdef_id": cdef_id})
    matches: list[LocatedTypedRecord] = []
    for store_index, store in enumerate(_stores(repo)):
        for ref in store.records.find_records(kind="stored_state", subject_cdef_id=cdef_id):
            record = StoredStateRecord.from_envelope(store.records.read_record(ref.record_id))
            if req.representation_id is not None and record.representation_id != req.representation_id:
                continue
            matches.append(LocatedTypedRecord(ref, record))
    reporting.detail("dryml.records.state.find", "Stored state candidates found", data={"candidates": len(matches)})
    return tuple(matches)


def find_compatible_state_record(repo: Any, cdef_id: str, requirement: RepresentationRequirement | Mapping[str, Any] | None = None) -> StateResolutionResult:
    """Select the best existing compatible stored-state record without adapters."""

    request = StateResolutionRequest(cdef_id, RepresentationRequirement.from_json(requirement) if not isinstance(requirement, RepresentationRequirement) else requirement)
    issues: list[RecordResolutionIssue] = []
    candidates = _state_candidates(repo, request.cdef_id, issues=issues)
    compatible = []
    for candidate in candidates:
        report = representation_satisfies(candidate.representation, request.requirement)
        reporting.detail(
            "dryml.records.representation.check",
            "Checked representation compatibility",
            record_id=candidate.located.ref.record_id,
            data=report.to_json(),
        )
        if report.compatible:
            compatible.append(candidate)
    if not candidates:
        if issues:
            report = RecordResolutionReport("failed", tuple(issues), 0)
            return StateResolutionResult("failed", report=report)
        report = RecordResolutionReport("not_found", (RecordResolutionIssue("not_found", "no stored_state records for CDef"),), 0)
        return StateResolutionResult("not_found", report=report)
    if not compatible:
        report = RecordResolutionReport("unsupported", (RecordResolutionIssue("unsupported", "no stored_state record satisfies representation requirement"), *issues), len(candidates))
        return StateResolutionResult("unsupported", report=report)
    ordered = sorted(compatible, key=lambda candidate: _candidate_sort_key(candidate, request.requirement))
    selected = ordered[0]
    report = RecordResolutionReport("ok", candidates_considered=len(candidates))
    reporting.detail(
        "dryml.records.state.find",
        "Selected stored state record",
        record_id=selected.located.ref.record_id,
        data={"representation_id": selected.representation.id},
    )
    return StateResolutionResult("ok", selected=selected.located, selected_representation=selected.representation, report=report)


def resolve_state_record(
    repo: Any,
    cdef_id: str,
    requirement: RepresentationRequirement | Mapping[str, Any] | None = None,
    *,
    adapters: Any = None,
) -> StateResolutionResult:
    """Resolve an existing compatible record or plan a fake/local adapter path."""

    req = requirement if isinstance(requirement, RepresentationRequirement) else RepresentationRequirement.from_json(requirement)
    existing = find_compatible_state_record(repo, cdef_id, req)
    if existing.status == "ok":
        return existing
    if adapters is None:
        return existing
    from .adapters import find_adapter_path

    issues: list[RecordResolutionIssue] = []
    candidates = _state_candidates(repo, cdef_id, issues=issues)
    plan = find_adapter_path(tuple(candidates), req, registry=adapters)
    if plan.status == "ok":
        if not plan.steps:
            report = RecordResolutionReport("ok", tuple(issues), candidates_considered=len(candidates))
            return StateResolutionResult("ok", selected=plan.source_record, selected_representation=plan.source_representation, adapter_plan=plan, report=report)
        report = RecordResolutionReport("requires_adapter", tuple(issues), candidates_considered=len(candidates), adapter_steps_considered=len(plan.steps))
        return StateResolutionResult(
            "requires_adapter",
            adapter_source=plan.source_record,
            adapter_source_representation=plan.source_representation,
            adapter_plan=plan,
            report=report,
        )
    report = RecordResolutionReport(plan.status, (*plan.issues, *issues), candidates_considered=len(candidates), adapter_steps_considered=len(plan.steps))
    return StateResolutionResult(plan.status, report=report)


def _state_candidates(repo: Any, cdef_id: str, *, issues: list[RecordResolutionIssue] | None = None) -> tuple[RepresentationCandidate, ...]:
    candidates: list[RepresentationCandidate] = []
    for store_index, store in enumerate(_stores(repo)):
        for ref in store.records.find_records(kind="stored_state", subject_cdef_id=cdef_id):
            located = LocatedTypedRecord(ref, StoredStateRecord.from_envelope(store.records.read_record(ref.record_id)))
            try:
                spec = _read_representation(repo, located.record.representation_id)
            except SpecNotFoundError:
                if issues is not None:
                    issues.append(RecordResolutionIssue(
                        "missing_representation_spec",
                        "stored_state record references a missing representation spec",
                        record_id=ref.record_id,
                        representation_id=located.record.representation_id,
                    ))
                continue
            try:
                candidates.append(RepresentationCandidate(located, RepresentationSpec(spec), store_index))
            except Exception as exc:
                if issues is not None:
                    issues.append(RecordResolutionIssue(
                        "invalid_representation_spec",
                        str(exc),
                        record_id=ref.record_id,
                        representation_id=located.record.representation_id,
                    ))
    return tuple(candidates)


def _read_representation(repo: Any, representation_id: str) -> dict[str, Any]:
    for store in _stores(repo):
        if store.records.has_spec(representation_id, family="representation"):
            return store.records.read_spec(representation_id, family="representation")
    raise SpecNotFoundError("representation spec not found", context={"representation_id": representation_id})


def _candidate_sort_key(candidate: RepresentationCandidate, requirement: RepresentationRequirement) -> tuple[Any, ...]:
    exact = requirement.representation_id is not None and candidate.representation.id == requirement.representation_id
    same_kind = requirement.kind is not None and candidate.representation.kind == requirement.kind
    object_storage = any(ref.kind == "object-dir" for ref in candidate.located.record.storage) if isinstance(candidate.located.record, StoredStateRecord) else False
    return (0 if exact else 1, 0 if same_kind else 1, candidate.store_index, 0 if object_storage else 1, candidate.located.ref.record_id)


def _stores(repo: Any) -> tuple[Any, ...]:
    stores = getattr(repo, "stores", None)
    if stores is None:
        return (repo,)
    return tuple(stores)


__all__ = [
    "LocatedTypedRecord",
    "RecordResolutionIssue",
    "RecordResolutionReport",
    "RepresentationCandidate",
    "StateResolutionRequest",
    "StateResolutionResult",
    "find_compatible_state_record",
    "find_stored_state_records",
    "resolve_state_record",
]
