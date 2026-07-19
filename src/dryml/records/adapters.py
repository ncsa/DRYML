"""Local/fake adapter descriptors, planning, and lineage writing."""

from __future__ import annotations

import inspect
import heapq
import itertools
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dryml import reporting
from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready

from .errors import RecordValidationError, SpecValidationError
from .products import ProductManifest, ProductWriteSession, require_product_integrity
from .refs import LocatedRecordRef
from .representations import RepresentationRequirement, RepresentationSpec, make_representation_spec, representation_satisfies
from .resolution import LocatedTypedRecord, RecordResolutionIssue, RepresentationCandidate
from .storage import StorageRef
from .typed import AdapterRecord, DataRecord, ProgramRecord, StoredStateRecord, TypedRecord


PlanStatus = Literal["ok", "not_found", "unsupported", "failed"]
AdapterRunner = Callable[..., Mapping[str, Any] | None]


@dataclass(frozen=True, slots=True)
class AdapterDescriptor:
    """Serializable description of one representation conversion edge."""

    name: str
    source: RepresentationRequirement
    target: RepresentationRequirement
    version: str | None = None
    provider_name: str | None = None
    provider_version: str | None = None
    operation: Mapping[str, Any] | None = None
    cost: float = 1.0
    streaming: bool = False
    materializes_source: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise RecordValidationError("adapter descriptor name must be a non-empty string")
        for field_name in ("version", "provider_name", "provider_version"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise RecordValidationError(f"adapter descriptor {field_name} must be a non-empty string")
        if not isinstance(self.source, RepresentationRequirement):
            object.__setattr__(self, "source", RepresentationRequirement.from_json(self.source))
        if not isinstance(self.target, RepresentationRequirement):
            object.__setattr__(self, "target", RepresentationRequirement.from_json(self.target))
        if not _is_nonnegative_finite_number(self.cost):
            raise RecordValidationError("adapter descriptor cost must be non-negative")
        if not isinstance(self.streaming, bool) or not isinstance(self.materializes_source, bool):
            raise RecordValidationError("adapter streaming flags must be boolean")
        object.__setattr__(self, "operation", None if self.operation is None else _freeze_mapping(self.operation, "operation"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @property
    def key(self) -> tuple[str, str | None, str | None]:
        """Return the registry identity key."""

        return (self.name, self.version, self.provider_name)

    def to_json(self) -> dict[str, Any]:
        """Return JSON-ready descriptor data."""

        return {
            "name": self.name,
            "version": self.version,
            "source": self.source.to_json(),
            "target": self.target.to_json(),
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "operation": None if self.operation is None else json_ready(self.operation),
            "cost": self.cost,
            "streaming": self.streaming,
            "materializes_source": self.materializes_source,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "AdapterDescriptor":
        """Build a descriptor from JSON-ready data."""

        if not isinstance(data, Mapping):
            raise RecordValidationError("adapter descriptor must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"name", "version", "source", "target", "provider_name", "provider_version", "operation", "cost", "streaming", "materializes_source", "metadata"}
        if unknown:
            raise RecordValidationError("adapter descriptor has unknown fields", context={"fields": sorted(unknown)})
        return cls(
            name=data.get("name"),
            version=data.get("version"),
            source=RepresentationRequirement.from_json(data.get("source") or {}),
            target=RepresentationRequirement.from_json(data.get("target") or {}),
            provider_name=data.get("provider_name"),
            provider_version=data.get("provider_version"),
            operation=data.get("operation"),
            cost=float(data.get("cost", 1.0)),
            streaming=data.get("streaming", False),
            materializes_source=data.get("materializes_source", True),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True, slots=True)
class AdapterStep:
    """One selected adapter edge in a plan."""

    descriptor: AdapterDescriptor
    source: RepresentationRequirement
    target: RepresentationRequirement


@dataclass(frozen=True, slots=True)
class AdapterPlan:
    """Selected adapter path from a source record to a target requirement."""

    source_record: LocatedTypedRecord | None
    source_representation: RepresentationSpec | None
    target: RepresentationRequirement
    steps: tuple[AdapterStep, ...] = ()
    status: PlanStatus = "ok"
    issues: tuple[RecordResolutionIssue, ...] = ()
    total_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class AdapterSearchLimits:
    """Hard bounds for deterministic adapter path resolution."""

    max_steps: int = 4
    max_expansions: int = 128
    max_total_cost: float = 1_000_000.0
    max_materialize_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_expansions", "max_materialize_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise RecordValidationError(f"adapter search {name} must be a positive integer")
        if not _is_nonnegative_finite_number(self.max_total_cost):
            raise RecordValidationError("adapter search max_total_cost must be non-negative")


class AdapterUnsupportedError(RuntimeError):
    """Structured unsupported outcome raised by an executable adapter."""

    def __init__(self, message: str, *, code: str = "unsupported"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class AdapterExecutionContext:
    """Context passed to local fake adapter runners."""

    repo: Any
    store: Any
    step: AdapterStep
    source_record: LocatedTypedRecord
    source_representation: RepresentationSpec
    target_requirement: RepresentationRequirement
    session: ProductWriteSession


@dataclass(frozen=True, slots=True)
class AdapterExecutionResult:
    """Result of local fake adapter execution."""

    status: PlanStatus
    target_records: tuple[LocatedRecordRef, ...] = ()
    adapter_records: tuple[LocatedRecordRef, ...] = ()
    issues: tuple[RecordResolutionIssue, ...] = ()


class AdapterRegistry:
    """In-memory registry for local/fake adapter descriptors and runners."""

    def __init__(self) -> None:
        self._descriptors: dict[tuple[str, str | None, str | None], AdapterDescriptor] = {}
        self._runners: dict[tuple[str, str | None, str | None], AdapterRunner] = {}

    def register(self, descriptor: AdapterDescriptor | Mapping[str, Any], *, runner: AdapterRunner | None = None) -> AdapterDescriptor:
        """Register a descriptor and optional local test runner."""

        desc = descriptor if isinstance(descriptor, AdapterDescriptor) else AdapterDescriptor.from_json(descriptor)
        if desc.key in self._descriptors:
            raise RecordValidationError("adapter descriptor is already registered", context={"name": desc.name, "version": desc.version})
        self._descriptors[desc.key] = desc
        if runner is not None:
            self._runners[desc.key] = runner
        return desc

    def descriptors(self) -> tuple[AdapterDescriptor, ...]:
        """Return registered descriptors in deterministic order."""

        return tuple(sorted(self._descriptors.values(), key=_descriptor_sort_key))

    def runner_for(self, descriptor: AdapterDescriptor) -> AdapterRunner | None:
        """Return the local runner for a descriptor, if any."""

        return self._runners.get(descriptor.key)

    def matching(self, *, source: RepresentationSpec | RepresentationRequirement | None = None, target: RepresentationRequirement | None = None) -> tuple[AdapterDescriptor, ...]:
        """Return descriptors matching optional source/target requirements."""

        result = []
        for descriptor in self.descriptors():
            if source is not None and not _descriptor_source_matches(descriptor, source):
                continue
            if target is not None and not _requirement_satisfies(descriptor.target, target):
                continue
            result.append(descriptor)
        return tuple(result)


def find_adapter_path(
    sources: Iterable[RepresentationCandidate | LocatedTypedRecord],
    target: RepresentationRequirement | Mapping[str, Any],
    *,
    registry: AdapterRegistry | None = None,
    descriptors: Iterable[AdapterDescriptor | Mapping[str, Any]] | None = None,
    limits: AdapterSearchLimits | None = None,
) -> AdapterPlan:
    """Find the bounded deterministic best-cost adapter path."""

    req = target if isinstance(target, RepresentationRequirement) else RepresentationRequirement.from_json(target)
    bounds = limits or AdapterSearchLimits()
    descs = _descriptors_from(registry, descriptors)
    candidates = tuple(_coerce_candidate(source) for source in sources)
    reporting.step("dryml.records.adapter.plan", "Planning adapter path", data={"sources": len(candidates), "descriptors": len(descs)})
    if not candidates:
        return AdapterPlan(None, None, req, status="not_found", issues=(RecordResolutionIssue("not_found", "no source records available"),))
    zero = [candidate for candidate in candidates if representation_satisfies(candidate.representation, req).compatible]
    if zero:
        selected = sorted(zero, key=lambda candidate: (candidate.store_index, candidate.located.ref.record_id))[0]
        return AdapterPlan(selected.located, selected.representation, req, steps=(), total_cost=0.0)

    serial = itertools.count()
    queue: list[tuple[Any, ...]] = []
    best: dict[tuple[str, str], float] = {}
    materializing_rejected = False
    for candidate in candidates:
        current = _requirement_from_spec(candidate.representation)
        heapq.heappush(queue, (0.0, 0, (), candidate.store_index, candidate.located.ref.record_id, next(serial), candidate, current, ()))
    expansions = 0
    search_bound_reached = False
    selected = None
    while queue:
        cost, _length, _sort, _store_index, _record_id, _serial, candidate, current, steps = heapq.heappop(queue)
        state_key = (candidate.located.ref.record_id, _requirement_key(current))
        if cost > best.get(state_key, float("inf")):
            continue
        best[state_key] = cost
        if steps and _requirement_satisfies(current, req):
            selected = AdapterPlan(candidate.located, candidate.representation, req, steps, total_cost=cost)
            break
        if len(steps) >= bounds.max_steps:
            search_bound_reached = True
            continue
        if expansions >= bounds.max_expansions:
            search_bound_reached = True
            break
        expansions += 1
        source_bytes = _record_product_bytes(candidate.located.record)
        for descriptor in descs:
            if not _requirement_satisfies(current, descriptor.source):
                continue
            if descriptor.materializes_source and source_bytes > bounds.max_materialize_bytes:
                materializing_rejected = True
                continue
            next_cost = cost + descriptor.cost
            if next_cost > bounds.max_total_cost:
                continue
            step = AdapterStep(descriptor, current, descriptor.target)
            next_steps = steps + (step,)
            next_key = (candidate.located.ref.record_id, _requirement_key(descriptor.target))
            if next_cost >= best.get(next_key, float("inf")):
                continue
            best[next_key] = next_cost
            sort_key = tuple(_descriptor_sort_key(item.descriptor) for item in next_steps)
            heapq.heappush(queue, (next_cost, len(next_steps), sort_key, candidate.store_index, candidate.located.ref.record_id, next(serial), candidate, descriptor.target, next_steps))
    if selected is None:
        if materializing_rejected:
            issue = RecordResolutionIssue("materializing_adapter_rejected", "adapter would materialize a product larger than the configured bound")
        elif search_bound_reached:
            issue = RecordResolutionIssue(
                "search_bound_exceeded",
                "adapter search exceeded a configured path or expansion bound",
            )
        else:
            issue = RecordResolutionIssue("unsupported", "no adapter path satisfies target requirement")
        return AdapterPlan(None, None, req, status="unsupported", issues=(issue,))
    reporting.detail("dryml.records.adapter.plan", "Selected adapter path", data={"steps": [step.descriptor.name for step in selected.steps], "total_cost": selected.total_cost})
    return selected


def run_adapter_plan(plan: AdapterPlan, *, repo: Any, store: Any, registry: AdapterRegistry) -> AdapterExecutionResult:
    """Run a local/fake adapter plan and write target plus AdapterRecord sidecars.

    This is intentionally not dispatch v2. It only calls registered in-process
    runners for unit tests and fake adapters.
    """

    if plan.status != "ok" or plan.source_record is None or plan.source_representation is None:
        return AdapterExecutionResult("unsupported", issues=plan.issues or (RecordResolutionIssue("unsupported", "adapter plan is not executable"),))
    if not plan.steps:
        return AdapterExecutionResult("ok", target_records=(plan.source_record.ref,), adapter_records=())
    current_record = plan.source_record
    current_repr = plan.source_representation
    targets: list[LocatedRecordRef] = []
    adapters: list[LocatedRecordRef] = []
    for step in plan.steps:
        runner = registry.runner_for(step.descriptor)
        if runner is None:
            return AdapterExecutionResult("unsupported", tuple(targets), tuple(adapters), (RecordResolutionIssue("missing_runner", "adapter descriptor has no local runner"),))
        reporting.step("dryml.records.adapter.run", "Running adapter step", data={"adapter": step.descriptor.name})
        try:
            source_store = _store_for_located(repo, current_record.ref) or store
            source_envelope = source_store.records.read_record(
                current_record.ref.record_id
            )
            require_product_integrity(source_store.records, source_envelope)
            if hasattr(current_record.record, "storage"):
                for storage_ref in current_record.record.storage:
                    source_store.records.resolve_storage_ref(storage_ref, record_id=current_record.ref.record_id)
            with ProductWriteSession(store.records) as session:
                context = AdapterExecutionContext(repo, store, step, current_record, current_repr, step.target, session)
                runner_output = _call_runner(runner, context)
                manifest = session.manifest()
                target_repr = _ensure_target_representation(store, step.target)
                target_record = _build_target_record(current_record.record, current_record.ref.record_id, target_repr.id, manifest, runner_output or {})
                reporting.step("dryml.records.product.write", "Writing product record", data={"representation_id": target_repr.id})
                target_result = session.commit_record(target_record.to_envelope())
            adapter_record = AdapterRecord(
                adapter=_adapter_payload(step.descriptor),
                operation_id=(step.descriptor.operation or {}).get("operation_id") if step.descriptor.operation else None,
                source_record_id=current_record.ref.record_id,
                source_representation_id=current_repr.id,
                target_record_id=target_result.located.record_id,
                target_representation_id=target_repr.id,
                produced_records=(target_result.located.record_id,),
                derived_from=(current_record.ref.record_id,),
            )
            reporting.step("dryml.records.adapter.record", "Writing adapter lineage record", data={"source_record_id": current_record.ref.record_id, "target_record_id": target_result.located.record_id})
            adapter_ref = store.records.write_record(adapter_record.to_envelope())
            targets.append(target_result.located)
            adapters.append(adapter_ref)
            current_record = LocatedTypedRecord(target_result.located, target_record)
            current_repr = target_repr
        except AdapterUnsupportedError as exc:
            return AdapterExecutionResult("unsupported", tuple(targets), tuple(adapters), (RecordResolutionIssue(exc.code, str(exc)),))
        except Exception as exc:
            return AdapterExecutionResult("failed", tuple(targets), tuple(adapters), (RecordResolutionIssue("adapter_failed", str(exc)),))
    return AdapterExecutionResult("ok", tuple(targets), tuple(adapters))


def adapter_descriptors_from_report(report: Any) -> tuple[AdapterDescriptor, ...]:
    """Extract adapter descriptors from a provider adapter-planning report."""

    payload = getattr(report, "report_payload", None) or {}
    adapters = payload.get("adapters") if isinstance(payload, Mapping) else None
    return tuple(AdapterDescriptor.from_json(item) for item in _json_sequence(adapters, "adapters"))


def _descriptors_from(registry: AdapterRegistry | None, descriptors: Iterable[AdapterDescriptor | Mapping[str, Any]] | None) -> tuple[AdapterDescriptor, ...]:
    result: list[AdapterDescriptor] = []
    if registry is not None:
        result.extend(registry.descriptors())
    if descriptors is not None:
        result.extend(item if isinstance(item, AdapterDescriptor) else AdapterDescriptor.from_json(item) for item in descriptors)
    return tuple(sorted(result, key=_descriptor_sort_key))


def _coerce_candidate(source: RepresentationCandidate | LocatedTypedRecord) -> RepresentationCandidate:
    if isinstance(source, RepresentationCandidate):
        return source
    raise RecordValidationError("adapter path sources must include representation specs")


def _descriptor_source_matches(descriptor: AdapterDescriptor, source: RepresentationSpec | RepresentationRequirement) -> bool:
    if isinstance(source, RepresentationSpec):
        return representation_satisfies(source, descriptor.source).compatible
    return _requirement_satisfies(source, descriptor.source)


def _requirement_satisfies(available: RepresentationRequirement, requested: RepresentationRequirement) -> bool:
    if requested.representation_id is not None and available.representation_id != requested.representation_id:
        return False
    if requested.kind is not None and available.kind != requested.kind:
        return False
    if requested.version is not None and available.version != requested.version:
        return False
    for key, value in requested.parameters.items():
        if available.parameters.get(key) != value:
            return False
    if set(requested.required_traits) - set(available.required_traits):
        return False
    if set(requested.storage_kinds) - set(available.storage_kinds):
        return False
    return True


def _requirement_from_spec(spec: RepresentationSpec) -> RepresentationRequirement:
    return RepresentationRequirement(kind=spec.kind, representation_id=spec.id, version=spec.version, parameters=spec.parameters, required_traits=spec.traits, storage_kinds=spec.storage_kinds)


def _ensure_target_representation(store: Any, requirement: RepresentationRequirement) -> RepresentationSpec:
    if requirement.representation_id is not None:
        if not store.records.has_spec(
            requirement.representation_id, family="representation"
        ):
            raise SpecValidationError(
                "exact adapter target representation spec is missing"
            )
        return RepresentationSpec(
            store.records.read_spec(
                requirement.representation_id, family="representation"
            )
        )
    if requirement.kind is None:
        raise SpecValidationError("adapter target requirement needs kind or existing representation_id")
    spec = make_representation_spec(
        requirement.kind,
        version=requirement.version,
        parameters=requirement.parameters,
        traits=requirement.required_traits,
        storage_kinds=requirement.storage_kinds or ("product-dir",),
    )
    store.records.write_spec(spec, family="representation")
    return RepresentationSpec(spec)


def _build_target_record(source: TypedRecord, source_record_id: str, representation_id: str, manifest: ProductManifest, runner_output: Mapping[str, Any]) -> TypedRecord:
    extra = dict(runner_output.get("payload") or {})
    storage = tuple(StorageRef.self_product(path=".", role=runner_output.get("storage_role") or "target-state") for _ in (0,))
    if isinstance(source, StoredStateRecord):
        return StoredStateRecord(
            subject_cdef_id=source.subject_cdef_id,
            representation_id=representation_id,
            storage=storage,
            owner_cdef_id=source.owner_cdef_id,
            owner_path=source.owner_path,
            state_role=source.state_role,
            manifest=manifest.to_json(),
            derived_from=(source_record_id,),
            realization_id=source.realization_id,
            output_slot=source.output_slot,
            extra={key: value for key, value in extra.items() if key != "derived_from"},
        )
    if isinstance(source, DataRecord):
        return DataRecord(representation_id=representation_id, storage=storage, subject_cdef_id=source.subject_cdef_id, operation_id=source.operation_id, data_role=source.data_role, manifest=manifest.to_json(), derived_from=(source_record_id,), realization_id=source.realization_id, output_slot=source.output_slot, extra=extra)
    if isinstance(source, ProgramRecord):
        return ProgramRecord(representation_id=representation_id, storage=storage, operation_id=source.operation_id, target=source.target, entrypoints=source.entrypoints, provider=source.provider, toolchain=source.toolchain, manifest=manifest.to_json(), derived_from=(source_record_id,), extra=extra)
    raise RecordValidationError("adapter source record kind is not supported")


def _call_runner(runner: AdapterRunner, context: AdapterExecutionContext) -> Mapping[str, Any] | None:
    try:
        sig = inspect.signature(runner)
        if len(sig.parameters) == 1:
            return runner(context)
    except (TypeError, ValueError):
        pass
    return runner(context=context, session=context.session, source_record=context.source_record, step=context.step)


def _json_sequence(value: Any, field_name: str) -> Any:
    if value is None:
        return ()
    if isinstance(value, str):
        raise RecordValidationError(f"adapter report {field_name} must be a JSON array, not a string")
    if not isinstance(value, (list, tuple)):
        raise RecordValidationError(f"adapter report {field_name} must be a JSON array", context={"type": type(value).__name__})
    return value


def _adapter_payload(descriptor: AdapterDescriptor) -> dict[str, Any]:
    payload = {"name": descriptor.name}
    if descriptor.version is not None:
        payload["version"] = descriptor.version
    if descriptor.provider_name is not None:
        payload["provider"] = descriptor.provider_name
    if descriptor.provider_version is not None:
        payload["provider_version"] = descriptor.provider_version
    return payload


def _store_for_located(repo: Any, ref: LocatedRecordRef) -> Any | None:
    for candidate in tuple(getattr(repo, "stores", ()) or (repo,)):
        try:
            if candidate.records._store_ref() == ref.store_ref:
                return candidate
        except Exception:
            continue
    return None


def _descriptor_sort_key(descriptor: AdapterDescriptor) -> tuple[Any, ...]:
    return (descriptor.name, descriptor.version or "", descriptor.provider_name or "", descriptor.provider_version or "", descriptor.cost)


def _plan_sort_key(plan: AdapterPlan) -> tuple[Any, ...]:
    return (plan.total_cost, len(plan.steps), tuple(_descriptor_sort_key(step.descriptor) for step in plan.steps), plan.source_record.ref.record_id if plan.source_record else "")


def _requirement_key(requirement: RepresentationRequirement) -> str:
    import json

    return json.dumps(requirement.to_json(), sort_keys=True, separators=(",", ":"))


def _record_product_bytes(record: TypedRecord) -> int:
    manifest = getattr(record, "manifest", None)
    if not isinstance(manifest, Mapping):
        return 0
    total_size = manifest.get("total_size")
    if type(total_size) is int and total_size >= 0:
        return total_size
    entries = manifest.get("entries")
    if not isinstance(entries, (list, tuple)):
        return 0
    return sum(item.get("size", 0) for item in entries if isinstance(item, Mapping) and type(item.get("size")) is int)


def _is_nonnegative_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def _freeze_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordValidationError("adapter descriptor field must be a mapping", context={"path": path, "type": type(value).__name__})
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise RecordValidationError("adapter descriptor field is not JSON-ready", context={"path": path, **exc.context}) from exc
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = [
    "AdapterDescriptor",
    "AdapterExecutionContext",
    "AdapterExecutionResult",
    "AdapterPlan",
    "AdapterRegistry",
    "AdapterSearchLimits",
    "AdapterStep",
    "AdapterUnsupportedError",
    "adapter_descriptors_from_report",
    "find_adapter_path",
    "run_adapter_plan",
]
