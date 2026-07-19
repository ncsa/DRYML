"""Closure planning and copy/export helpers for record sidecars."""

from __future__ import annotations

import shutil
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dryml.formats.ids import parse_content_id

from .errors import RecordClosureError, RecordExportError
from .policy import (
    RECORD_POLICY_ALL,
    RECORD_POLICY_CLOSURE,
    RECORD_POLICY_DESCRIPTIVE,
    RECORD_POLICY_NONE,
    RECORD_POLICY_PROVENANCE,
    RecordPolicy,
    RecordPolicyOptions,
    RecordPolicyReport,
    normalize_record_policy,
    policy_includes_products,
)
from .products import (
    _fsync_directory,
    _fsync_tree,
    _trees_match,
    validate_product_availability,
)
from .refs import LocatedRecordRef, LocatedSpecRef, RecordRef, SpecRef
from .scanner import ReferenceMention, scan_record_refs, scan_spec_refs
from .specs import spec_dir_name, spec_family_for_id


_PROVENANCE_RECORD_KINDS = {
    "execution",
    "adapter",
    "probe_report",
    "compatibility_report",
    "lowering_report",
}
_PROVENANCE_RECORD_ROLES = {"derived_from", "record", "source_record", "target_record", "consumed_record", "produced_record"}


@dataclass(frozen=True, slots=True)
class RecordClosurePlan:
    """Deterministic record/spec/product closure rooted in one source store."""

    policy: RecordPolicy
    source_store_ref: str
    destination_store_ref: str | None
    records: tuple[str, ...]
    specs: tuple[tuple[str, str], ...]
    products: tuple[str, ...]
    omitted_records: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def plan_record_closure(
    source_store: Any,
    *,
    destination_store: Any | None = None,
    seed_records: Iterable[str | RecordRef | LocatedRecordRef] = (),
    seed_specs: Iterable[tuple[str, str] | str | SpecRef | LocatedSpecRef] = (),
    policy: RecordPolicy | str | None = RECORD_POLICY_CLOSURE,
    options: RecordPolicyOptions | None = None,
) -> RecordClosurePlan:
    """Build a record/spec/product closure from authoritative JSON sidecars."""

    resolved_policy = normalize_record_policy(policy)
    resolved_options = RecordPolicyOptions() if options is None else options
    source_io = source_store.records
    destination_ref = None if destination_store is None else destination_store.records._store_ref()
    include_products = policy_includes_products(resolved_policy, resolved_options)
    records: set[str] = set()
    specs: set[tuple[str, str]] = set()
    products: set[str] = set()
    provenance_targets: set[str] = set()
    warnings: list[str] = []

    record_queue = deque(_normalize_seed_records(seed_records))
    spec_queue = deque(_normalize_seed_specs(seed_specs))
    provenance_targets.update(record_queue)
    provenance_targets.update(spec_id for _, spec_id in spec_queue)

    if resolved_policy == RECORD_POLICY_NONE:
        return RecordClosurePlan(
            policy=resolved_policy,
            source_store_ref=source_io._store_ref(),
            destination_store_ref=destination_ref,
            records=(),
            specs=(),
            products=(),
        )

    if resolved_policy == RECORD_POLICY_DESCRIPTIVE:
        records.update(record_queue)
        specs.update(spec_queue)
        if include_products:
            products.update(_existing_product_record_ids(source_io, records))
        return RecordClosurePlan(
            policy=resolved_policy,
            source_store_ref=source_io._store_ref(),
            destination_store_ref=destination_ref,
            records=tuple(sorted(records)),
            specs=tuple(sorted(specs)),
            products=tuple(sorted(products)),
        )

    if resolved_policy == RECORD_POLICY_ALL:
        records.update(source_io.iter_record_ids())
        for spec in source_io.iter_specs():
            specs.add((spec_family_for_id(spec["id"]), spec["id"]))
        if include_products:
            products.update(_existing_product_record_ids(source_io, records))
        return RecordClosurePlan(
            policy=resolved_policy,
            source_store_ref=source_io._store_ref(),
            destination_store_ref=destination_ref,
            records=tuple(sorted(records)),
            specs=tuple(sorted(specs)),
            products=tuple(sorted(products)),
        )

    while record_queue or spec_queue:
        while record_queue:
            record_id = record_queue.popleft()
            if record_id in records:
                continue
            record = source_io.read_record(record_id)
            records.add(record_id)
            if include_products and source_io.product_root(record_id).exists():
                products.add(record_id)
            for mention in scan_record_refs(record):
                if resolved_policy == RECORD_POLICY_PROVENANCE:
                    provenance_targets.add(mention.target_id)
                _expand_mention(
                    mention,
                    specs=specs,
                    spec_queue=spec_queue,
                    record_queue=record_queue,
                    follow_records=resolved_policy == RECORD_POLICY_PROVENANCE,
                )
        while spec_queue:
            family, spec_id = spec_queue.popleft()
            spec_key = (family, spec_id)
            if spec_key in specs:
                continue
            spec = source_io.read_spec(spec_id, family=family)
            specs.add(spec_key)
            for mention in scan_spec_refs(spec, family=family):
                if resolved_policy == RECORD_POLICY_PROVENANCE:
                    provenance_targets.add(mention.target_id)
                _expand_mention(
                    mention,
                    specs=specs,
                    spec_queue=spec_queue,
                    record_queue=record_queue,
                    follow_records=resolved_policy == RECORD_POLICY_PROVENANCE,
                )

        if resolved_policy == RECORD_POLICY_PROVENANCE:
            provenance_targets.update(records)
            provenance_targets.update(spec_id for _, spec_id in specs)
            added = _queue_reverse_provenance_records(source_io, records, record_queue)
            if added:
                continue

    return RecordClosurePlan(
        policy=resolved_policy,
        source_store_ref=source_io._store_ref(),
        destination_store_ref=destination_ref,
        records=tuple(sorted(records)),
        specs=tuple(sorted(specs)),
        products=tuple(sorted(products)),
        warnings=tuple(warnings),
    )



def copy_record_closure(
    source_store: Any,
    destination_store: Any,
    *,
    seed_records: Iterable[str | RecordRef | LocatedRecordRef] = (),
    seed_specs: Iterable[tuple[str, str] | str | SpecRef | LocatedSpecRef] = (),
    policy: RecordPolicy | str | None = RECORD_POLICY_CLOSURE,
    options: RecordPolicyOptions | None = None,
) -> RecordPolicyReport:
    """Copy a planned record/spec/product closure between stores."""

    resolved_policy = normalize_record_policy(policy)
    resolved_options = RecordPolicyOptions() if options is None else options
    source_io = source_store.records
    dest_io = destination_store.records
    plan = plan_record_closure(
        source_store,
        destination_store=destination_store,
        seed_records=seed_records,
        seed_specs=seed_specs,
        policy=resolved_policy,
        options=resolved_options,
    )

    products_copied = []
    warnings = list(plan.warnings)
    if policy_includes_products(resolved_policy, resolved_options):
        for record_id in plan.records:
            issues = validate_product_availability(source_io, source_io.read_record(record_id))
            if issues:
                raise RecordExportError("source record has unavailable products", context={"record_id": record_id, "issues": [issue.to_json() for issue in issues]})
        for record_id in plan.products:
            if _copy_product_root(
                source_io,
                dest_io,
                record_id,
                collision=resolved_options.destination_collision,
            ):
                products_copied.append(record_id)
            else:
                warnings.append(f"missing product root for {record_id}")

    specs_written = []
    records_written = []
    for family, spec_id in plan.specs:
        specs_written.append(
            dest_io.write_spec(
                source_io.read_spec(spec_id, family=family),
                family=family,
                overwrite=resolved_options.overwrite_sidecars,
            )
        )
    for record_id in plan.records:
        records_written.append(
            dest_io.write_record(
                source_io.read_record(record_id),
                overwrite=resolved_options.overwrite_sidecars,
            )
        )

    indexes_rebuilt = False
    if resolved_options.rebuild_index:
        dest_io.rebuild_ref_index()
        indexes_rebuilt = True

    return RecordPolicyReport(
        policy=resolved_policy,
        store_ref=dest_io._store_ref(),
        records_written=tuple(sorted(records_written, key=lambda ref: (ref.store_ref, ref.record_id))),
        specs_written=tuple(sorted(specs_written, key=lambda ref: (ref.store_ref, ref.kind or "", ref.spec_id))),
        products_copied=tuple(sorted(products_copied)),
        indexes_rebuilt=indexes_rebuilt,
        warnings=tuple(sorted(set(warnings))),
    )


def record_export_include_paths(plan: RecordClosurePlan) -> set[str]:
    """Return ZipExportStore relative include paths for a closure plan."""

    paths = {f"records/items/{record_id}.json" for record_id in plan.records}
    paths.update(
        f"records/specs/{spec_dir_name(family)}/{spec_id}.json"
        for family, spec_id in plan.specs
    )
    paths.update(f"products/{record_id}/" for record_id in plan.products)
    return paths


def _normalize_seed_records(seeds: Iterable[str | RecordRef | LocatedRecordRef]) -> tuple[str, ...]:
    result = []
    for seed in seeds:
        if isinstance(seed, str):
            result.append(seed)
        elif isinstance(seed, (RecordRef, LocatedRecordRef)):
            result.append(seed.record_id)
        else:
            raise RecordClosureError("invalid seed record", context={"seed": repr(seed)})
        _require_prefix(result[-1], "record")
    return tuple(sorted(set(result)))


def _normalize_seed_specs(seeds: Iterable[tuple[str, str] | str | SpecRef | LocatedSpecRef]) -> tuple[tuple[str, str], ...]:
    result = []
    for seed in seeds:
        if isinstance(seed, str):
            family = spec_family_for_id(seed)
            if family is None:
                raise RecordClosureError("spec seed requires a known family", context={"spec_id": seed})
            result.append((family, seed))
        elif isinstance(seed, (SpecRef, LocatedSpecRef)):
            family = seed.kind or spec_family_for_id(seed.spec_id)
            if family is None:
                raise RecordClosureError("spec seed requires a family", context={"spec_id": seed.spec_id})
            result.append((family, seed.spec_id))
        elif isinstance(seed, tuple) and len(seed) == 2:
            family, spec_id = seed
            if not isinstance(family, str) or not isinstance(spec_id, str):
                raise RecordClosureError("spec seed tuple must be (family, spec_id)")
            result.append((family, spec_id))
        else:
            raise RecordClosureError("invalid seed spec", context={"seed": repr(seed)})
    return tuple(sorted(set(result)))


def _expand_mention(
    mention: ReferenceMention,
    *,
    specs: set[tuple[str, str]],
    spec_queue: deque[tuple[str, str]],
    record_queue: deque[str],
    follow_records: bool,
) -> None:
    if mention.target_kind != "content_id":
        return
    parts = parse_content_id(mention.target_id)
    if parts.prefix == "record":
        if follow_records and mention.typed_role in _PROVENANCE_RECORD_ROLES:
            record_queue.append(mention.target_id)
        return
    family = spec_family_for_id(mention.target_id)
    if family is None:
        return
    spec_key = (family, mention.target_id)
    if spec_key not in specs:
        spec_queue.append(spec_key)


def _queue_reverse_provenance_records(source_io: Any, records: set[str], queue: deque[str]) -> bool:
    if not records:
        return False
    queued = False
    mentions = source_io.find_mentions(refresh="auto")
    for mention in mentions:
        if mention.source_kind != "record" or mention.target_id not in records:
            continue
        if mention.typed_role not in _PROVENANCE_RECORD_ROLES:
            continue
        if mention.source_id in records or mention.source_id in queue:
            continue
        record = source_io.read_record(mention.source_id)
        if record.get("kind") not in _PROVENANCE_RECORD_KINDS:
            continue
        _report_provenance_export(record, mention.target_id)
        queue.append(mention.source_id)
        queued = True
    return queued


def _existing_product_record_ids(source_io: Any, record_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(record_id for record_id in sorted(record_ids) if source_io.product_root(record_id).exists())


def _copy_product_root(
    source_io: Any,
    dest_io: Any,
    record_id: str,
    *,
    collision: str,
) -> bool:
    source = source_io.product_root(record_id)
    if not source.exists():
        return False
    dest = dest_io.product_root(record_id)
    if dest.exists():
        identical = _trees_match(source, dest)
        if collision == "adopt-identical" and identical:
            return True
        message = (
            "destination product root already exists with different bytes"
            if not identical
            else "destination product root already exists"
        )
        raise RecordExportError(message, context={"record_id": record_id})
    dest.parent.mkdir(parents=True, exist_ok=True)
    import uuid

    tmp = dest.parent / f".copying-{record_id}-{uuid.uuid4().hex}"
    try:
        if source.is_dir():
            shutil.copytree(source, tmp)
        else:
            tmp.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, tmp / Path(source).name)
        _fsync_tree(tmp)
        try:
            tmp.replace(dest)
        except OSError:
            if dest.exists() and collision == "adopt-identical" and _trees_match(source, dest):
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                raise RecordExportError(
                    "destination product root appeared during copy",
                    context={"record_id": record_id},
                )
        _fsync_directory(dest.parent)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return True


def _require_prefix(value: str, prefix: str) -> None:
    parts = parse_content_id(value)
    if parts.prefix != prefix:
        raise RecordClosureError("content ID prefix mismatch", context={"value": value, "expected": prefix, "observed": parts.prefix})


def _report_provenance_export(record: Mapping[str, Any], target_id: str) -> None:
    try:
        from dryml import reporting

        payload = record.get("payload") or {}
        reporting.detail(
            "dryml.records.execution.export",
            "Including execution provenance in export closure",
            operation_id=payload.get("operation_id"),
            record_id=record.get("id"),
            data={"target_id": target_id, "kind": record.get("kind"), "status": payload.get("status"), "execution_kind": payload.get("execution_kind")},
        )
    except Exception:
        pass


__all__ = [
    "RecordClosurePlan",
    "copy_record_closure",
    "plan_record_closure",
    "record_export_include_paths",
]
