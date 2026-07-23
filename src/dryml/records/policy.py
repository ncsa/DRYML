"""Policy primitives and save-side metadata helpers for record sidecars."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dryml.formats.refs import format_cdef_id

from .errors import RecordPolicyError
from .records import make_record
from .refs import LocatedRecordRef, LocatedSpecRef
from .specs import attach_spec_id, make_spec
from .storage import StorageRef


RecordPolicy = Literal["none", "descriptive", "closure", "provenance", "all"]

RECORD_POLICY_NONE: RecordPolicy = "none"
RECORD_POLICY_DESCRIPTIVE: RecordPolicy = "descriptive"
RECORD_POLICY_CLOSURE: RecordPolicy = "closure"
RECORD_POLICY_PROVENANCE: RecordPolicy = "provenance"
RECORD_POLICY_ALL: RecordPolicy = "all"
DEFAULT_RECORD_POLICY: RecordPolicy = RECORD_POLICY_DESCRIPTIVE
RECORD_POLICIES: tuple[RecordPolicy, ...] = (
    RECORD_POLICY_NONE,
    RECORD_POLICY_DESCRIPTIVE,
    RECORD_POLICY_CLOSURE,
    RECORD_POLICY_PROVENANCE,
    RECORD_POLICY_ALL,
)

_DEFAULT_REPRESENTATION_KIND = "dryml.object_state"
_DEFAULT_REPRESENTATION_PAYLOAD = {
    "format": "dryml.object_state",
    "storage_kind": "object-dir",
    "role": "default-state",
    "description": "Default DRYML object state layout written under objects/.",
}
_POLICIES_WITH_DIRECT_SAVE_RECORDS = {
    RECORD_POLICY_DESCRIPTIVE,
    RECORD_POLICY_CLOSURE,
    RECORD_POLICY_PROVENANCE,
    RECORD_POLICY_ALL,
}


@dataclass(frozen=True, slots=True)
class RecordPolicyOptions:
    """Options controlling explicit record-policy side effects.

    ``include_products=None`` means use the selected policy default: currently
    only ``all`` includes existing product directories by default. Indexes are
    derived data and are omitted unless callers request rebuilding after writes.
    ``destination_collision='adopt-identical'`` makes retries idempotent while
    still rejecting a content-addressed destination with different bytes.
    """

    include_products: bool | None = None
    rebuild_index: bool = False
    overwrite_sidecars: bool = False
    destination_collision: Literal["error", "adopt-identical"] = "adopt-identical"
    representation_kind: str = _DEFAULT_REPRESENTATION_KIND
    representation_payload: Mapping[str, Any] | None = None
    record_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.destination_collision not in {"error", "adopt-identical"}:
            raise RecordPolicyError(
                "invalid destination collision policy",
                context={"value": self.destination_collision},
            )


@dataclass(frozen=True, slots=True)
class RecordPolicyReport:
    """Deterministic summary of records/specs/products written by a policy."""

    policy: RecordPolicy
    store_ref: str
    records_written: tuple[LocatedRecordRef, ...] = ()
    specs_written: tuple[LocatedSpecRef, ...] = ()
    products_copied: tuple[str, ...] = ()
    indexes_rebuilt: bool = False
    warnings: tuple[str, ...] = ()


def normalize_record_policy(value: str | None) -> RecordPolicy:
    """Return a canonical record policy or raise ``RecordPolicyError``."""

    if value is None:
        return DEFAULT_RECORD_POLICY
    if value in RECORD_POLICIES:
        return value  # type: ignore[return-value]
    raise RecordPolicyError(
        "invalid record policy",
        context={"value": value, "allowed": RECORD_POLICIES},
    )


def policy_includes_products(policy: RecordPolicy, options: RecordPolicyOptions | None = None) -> bool:
    """Return whether products should be copied for *policy* and *options*."""

    resolved = RecordPolicyOptions() if options is None else options
    if resolved.include_products is not None:
        return resolved.include_products
    return policy == RECORD_POLICY_ALL


def default_object_state_representation_spec(
    options: RecordPolicyOptions | None = None,
) -> dict[str, Any]:
    """Build the stable default representation spec for DRYML object state."""

    resolved = RecordPolicyOptions() if options is None else options
    payload = dict(_DEFAULT_REPRESENTATION_PAYLOAD)
    if resolved.representation_payload is not None:
        payload.update(resolved.representation_payload)
    return attach_spec_id(
        make_spec(
            family="representation",
            kind=resolved.representation_kind,
            payload=payload,
            metadata={"writer": "dryml.records.policy"},
        ),
        family="representation",
    )


def stored_state_record_for_save_action(
    action: Any,
    representation_id: str,
    *,
    policy: RecordPolicy = RECORD_POLICY_DESCRIPTIVE,
    options: RecordPolicyOptions | None = None,
) -> dict[str, Any]:
    """Build a direct ``stored_state`` record for a successful save action."""

    resolved = RecordPolicyOptions() if options is None else options
    subject_cdef_id = format_cdef_id(action.definition.stable_hash())
    save_payload: dict[str, Any] = {
        "reason": action.reason,
        "minimum_root_depth": action.minimum_root_depth,
    }
    if action.revision is not None:
        save_payload["revision"] = action.revision
    metadata = {"writer": "dryml.records.policy", "record_policy": policy}
    if resolved.record_metadata is not None:
        metadata.update(resolved.record_metadata)
    return make_record(
        kind="stored_state",
        payload={
            "subject_cdef_id": subject_cdef_id,
            "representation_id": representation_id,
            "storage": [
                StorageRef.object_dir(
                    subject_cdef_id,
                    path=".",
                    role="default-state",
                ).to_json()
            ],
            "save": save_payload,
        },
        metadata=metadata,
    )


def apply_save_record_policy(repo: Any, plan: Any, save_options: Any) -> RecordPolicyReport:
    """Apply explicit record side effects after a save plan has written objects.

    The function is deliberately duck-typed so importing this module remains
    independent of ``dryml.core``. ``none`` returns a no-op report and does not
    touch store record directories or indexes.
    """

    policy = normalize_record_policy(getattr(save_options, "record_policy", None))
    options = getattr(save_options, "record_options", None) or RecordPolicyOptions()
    if policy == RECORD_POLICY_NONE:
        return RecordPolicyReport(policy=policy, store_ref=_report_store_ref(plan))
    if policy not in _POLICIES_WITH_DIRECT_SAVE_RECORDS:
        raise RecordPolicyError("unsupported save record policy", context={"policy": policy})

    representation_spec = default_object_state_representation_spec(options)
    records: list[LocatedRecordRef] = []
    specs: list[LocatedSpecRef] = []
    seen_specs: set[tuple[str, str]] = set()
    seen_records: set[tuple[str, str]] = set()
    touched_ios = []

    for action in plan.actions:
        record_io = action.store.records
        store_ref = record_io._store_ref()
        spec_ref = record_io.write_spec(
            representation_spec,
            family="representation",
            overwrite=options.overwrite_sidecars,
        )
        spec_key = (spec_ref.store_ref, spec_ref.spec_id)
        if spec_key not in seen_specs:
            seen_specs.add(spec_key)
            specs.append(spec_ref)
        record = stored_state_record_for_save_action(
            action,
            representation_spec["id"],
            policy=policy,
            options=options,
        )
        record_ref = record_io.write_record(record, overwrite=options.overwrite_sidecars)
        record_key = (store_ref, record_ref.record_id)
        if record_key not in seen_records:
            seen_records.add(record_key)
            records.append(record_ref)
        if record_io not in touched_ios:
            touched_ios.append(record_io)

    indexes_rebuilt = False
    if options.rebuild_index:
        for record_io in touched_ios:
            record_io.rebuild_ref_index()
        indexes_rebuilt = bool(touched_ios)

    records.sort(key=lambda ref: (ref.store_ref, ref.record_id))
    specs.sort(key=lambda ref: (ref.store_ref, ref.kind or "", ref.spec_id))
    return RecordPolicyReport(
        policy=policy,
        store_ref=_report_store_ref(plan),
        records_written=tuple(records),
        specs_written=tuple(specs),
        indexes_rebuilt=indexes_rebuilt,
    )


def _report_store_ref(plan: Any) -> str:
    refs = sorted({action.store.records._store_ref() for action in getattr(plan, "actions", ())})
    if not refs:
        return ""
    if len(refs) == 1:
        return refs[0]
    return "*"


__all__ = [
    "DEFAULT_RECORD_POLICY",
    "RECORD_POLICIES",
    "RECORD_POLICY_ALL",
    "RECORD_POLICY_CLOSURE",
    "RECORD_POLICY_DESCRIPTIVE",
    "RECORD_POLICY_NONE",
    "RECORD_POLICY_PROVENANCE",
    "RecordPolicy",
    "RecordPolicyOptions",
    "RecordPolicyReport",
    "apply_save_record_policy",
    "default_object_state_representation_spec",
    "normalize_record_policy",
    "policy_includes_products",
    "stored_state_record_for_save_action",
]
