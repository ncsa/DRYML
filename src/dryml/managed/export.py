"""Crash-safe recipe and exact completed-realization transfer workflows."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from dryml.core2.definition import ConcreteDefinition, Definition
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.object import Object
from dryml.core2.repo import Repo, make_store
from dryml.core2.store.store import Store
from dryml.core2.utils.general import pickle_load
from dryml.formats.refs import format_cdef_id, parse_cdef_id
from dryml.records import (
    AdapterRecord,
    DataRecord,
    ExecutionRecord,
    RealizationRecord,
    RecordExportError,
    RecordPolicyOptions,
    StoredStateRecord,
    copy_record_closure,
    plan_record_closure,
    require_product_integrity,
    require_checkpoint_integrity,
)
from dryml.records.products import _fsync_directory, _fsync_tree, _trees_match

from .errors import ManagedStateError
from .locking import PlatformFileLock
from .refs import ManagedOutputRef
from .state import (
    ActivationEvent,
    GenerationControl,
    NamespaceState,
    OperationKey,
    RealizationState,
)
from .store import ManagedOperationStore, _write_json


TransferHistory = Literal["active", "all"]
TransferActivation = Literal["inactive", "if-absent"]


@dataclass(frozen=True, slots=True)
class RecipeExportReport:
    """Definitions installed by one definition-only recipe export."""

    root: ConcreteDefinition
    definition_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RealizationTransferReport:
    """Deterministic summary of one completed-realization transfer."""

    producer_cdef_id: str
    method: str
    realization_ids: tuple[str, ...]
    dependency_realization_ids: tuple[str, ...]
    records: tuple[str, ...]
    specs: tuple[tuple[str, str], ...]
    products: tuple[str, ...]
    definitions: tuple[str, ...]
    activated_realization_id: str | None


@dataclass(frozen=True, slots=True)
class _TransferNode:
    key: OperationKey
    declaration_fingerprint: str
    current_declaration_fingerprint: str
    generation_order: tuple[str, ...]
    state: RealizationState
    realization_record_id: str
    selected_activation_event: ActivationEvent | None = None


def export_recipe(
    value: Object | Definition | ConcreteDefinition,
    destination_store: Store | str | Path,
    *,
    main: bool = False,
) -> RecipeExportReport:
    """Install a complete logical definition closure without materialization.

    Existing definitions are adopted only when they decode to the same stable
    definition. Definition writes use the Store's atomic ``save_definition``
    boundary, and no Object state, records, products, or managed control is
    copied by this workflow.
    """

    destination = make_store(destination_store)
    repo = Repo(destination)
    try:
        root = repo.save_definition(value, main=main)
        if main:
            destination.commit()
        graph = ConcreteDefinitionGraph.for_query_index(root)
    finally:
        repo.close(flush=False)
    definition_ids = tuple(
        sorted(format_cdef_id(node.definition.stable_hash()) for node in graph.nodes())
    )
    return RecipeExportReport(root, definition_ids)


def transfer_realizations(
    source_store: Store | str | Path,
    destination_store: Store | str | Path,
    target: ManagedOutputRef,
    *,
    history: TransferHistory = "active",
    activate: TransferActivation = "if-absent",
) -> RealizationTransferReport:
    """Transfer exact completed bytes and lineage for one managed operation.

    ``history='active'`` snapshots the source's validated active realization;
    ``history='all'`` includes every retained completed realization across
    declaration generations. Exact consumed-result dependencies are
    recursively included. Destination products adopt only byte-identical
    collisions. Definitions and immutable records are installed before bounded
    sanitized control, and an optional compatible active pointer is installed
    last. Leases, owners, attempts, diagnostics, and source fence epochs are
    never copied.
    """

    if history not in {"active", "all"}:
        raise ValueError("history must be 'active' or 'all'")
    if activate not in {"inactive", "if-absent"}:
        raise ValueError("activate must be 'inactive' or 'if-absent'")
    if not isinstance(target, ManagedOutputRef):
        raise TypeError("target must be a ManagedOutputRef")
    source = make_store(source_store)
    destination = make_store(destination_store)
    source_managed = ManagedOperationStore(source, writable=False)
    key = OperationKey.from_producer(target.producer, target.method)
    namespace = source_managed._read_namespace(key, missing_ok=True)
    if namespace is None:
        raise ManagedStateError("source operation has no retained managed state")
    operation = source_managed.operation(
        key, namespace.current_declaration_fingerprint
    )
    source_active_event = operation.active_event(missing_ok=True)

    if history == "active":
        if source_active_event is None:
            raise ManagedStateError("source operation has no active completed realization")
        active_state = operation._read_realization(source_active_event.realization_id)
        if source_active_event.realization_record_id != active_state.realization_record_id:
            raise ManagedStateError(
                "source active event binds a different realization record"
            )
        root_states = (active_state,)
    else:
        root_states = tuple(
            state
            for state in source_managed.history(key)
            if state.status == "completed" and state.realization_record_id is not None
        )
        if not root_states:
            raise ManagedStateError("source operation has no completed realizations")

    nodes = _build_exact_closure(
        source_managed,
        key,
        root_states,
        root_active_event=source_active_event,
    )
    root_ids = tuple(state.realization_id for state in root_states)
    root_id_set = set(root_ids)
    dependency_ids = tuple(
        sorted(
            node.state.realization_id
            for node in nodes
            if node.state.realization_id not in root_id_set
        )
    )

    definition_ids = set(export_recipe(target.producer, destination).definition_ids)
    for producer_id in sorted({node.key.producer_cdef_id for node in nodes}):
        destination_definition = (
            Path(destination.object_dir_for_cdef_id(producer_id)) / "def.pkl"
        )
        definition = _read_definition(
            destination if destination_definition.is_file() else source,
            producer_id,
        )
        definition_ids.update(export_recipe(definition, destination).definition_ids)

    seed_records = tuple(
        sorted(
            {
                record_id
                for node in nodes
                for record_id in _node_record_ids(source, node)
            }
        )
    )
    record_plan = plan_record_closure(
        source,
        destination_store=destination,
        seed_records=seed_records,
        policy="closure",
        options=RecordPolicyOptions(include_products=True),
    )
    copy_record_closure(
        source,
        destination,
        seed_records=seed_records,
        policy="closure",
        options=RecordPolicyOptions(
            include_products=True,
            destination_collision="adopt-identical",
        ),
    )
    _copy_checkpoint_closure(source, destination, nodes)

    activation_event = None
    if source_active_event is not None and source_active_event.realization_id in root_id_set:
        activation_event = source_active_event
    activated = _install_control_closure(
        destination,
        nodes,
        root_key=key,
        root_realization_ids=frozenset(root_id_set),
        source_active_event=activation_event,
        activate=activate,
    )
    return RealizationTransferReport(
        producer_cdef_id=key.producer_cdef_id,
        method=key.method,
        realization_ids=root_ids,
        dependency_realization_ids=dependency_ids,
        records=record_plan.records,
        specs=record_plan.specs,
        products=record_plan.products,
        definitions=tuple(sorted(definition_ids)),
        activated_realization_id=activated,
    )


def _build_exact_closure(
    managed: ManagedOperationStore,
    root_key: OperationKey,
    root_states: tuple[RealizationState, ...],
    *,
    root_active_event: ActivationEvent | None,
) -> tuple[_TransferNode, ...]:
    queue = deque(
        (
            root_key,
            state,
            root_active_event
            if root_active_event is not None
            and root_active_event.realization_id == state.realization_id
            else None,
        )
        for state in root_states
    )
    nodes: dict[tuple[str, str, str], _TransferNode] = {}
    activation_events: dict[
        tuple[str, str, str], dict[int, ActivationEvent]
    ] = {}
    while queue:
        key, state, selected_activation_event = queue.popleft()
        node_key = (
            key.producer_cdef_id,
            key.method,
            state.realization_id,
        )
        if node_key in nodes:
            existing_event = nodes[node_key].selected_activation_event
            if (
                selected_activation_event is not None
                and existing_event is not None
                and selected_activation_event != existing_event
            ):
                raise ManagedStateError(
                    "exact transfer requires conflicting activations for one realization"
                )
            if selected_activation_event is not None and existing_event is None:
                nodes[node_key] = replace(
                    nodes[node_key],
                    selected_activation_event=selected_activation_event,
                )
            continue
        if state.status != "completed" or state.realization_record_id is None:
            raise ManagedStateError("exact transfer requires a completed realization record")
        namespace = managed._read_namespace(key, missing_ok=True)
        if namespace is None:
            raise ManagedStateError("exact lineage producer has no managed namespace")
        operation = managed.operation(key, state.declaration_fingerprint)
        authoritative = operation._read_realization(state.realization_id)
        if authoritative != state:
            raise ManagedStateError("realization state changed during transfer planning")
        realization = _validate_exact_realization(
            managed.store, operation, authoritative
        )
        node = _TransferNode(
            key,
            state.declaration_fingerprint,
            namespace.current_declaration_fingerprint,
            namespace.generations,
            authoritative,
            authoritative.realization_record_id,
            selected_activation_event,
        )
        nodes[node_key] = node
        for consumed in realization.consumed_records:
            consumed_key = OperationKey(consumed.producer_cdef_id, consumed.method)
            consumed_operation = managed.operation(
                consumed_key, consumed.declaration_fingerprint
            )
            operation_generation = (
                consumed.producer_cdef_id,
                consumed.method,
                consumed.declaration_fingerprint,
            )
            events_by_sequence = activation_events.get(operation_generation)
            if events_by_sequence is None:
                events_by_sequence = {
                    event.sequence: event
                    for event in consumed_operation._activation_events()
                }
                activation_events[operation_generation] = events_by_sequence
            event = events_by_sequence.get(consumed.activation_generation)
            if event is None or event.realization_id != consumed.realization_id:
                raise ManagedStateError(
                    "exact consumed lineage has no matching activation event"
                )
            consumed_state = consumed_operation._read_realization(
                consumed.realization_id
            )
            if event.realization_record_id != consumed_state.realization_record_id:
                raise ManagedStateError(
                    "exact consumed activation binds a different realization record"
                )
            consumed_realization = RealizationRecord.from_envelope(
                managed.store.records.read_record(consumed_state.realization_record_id)
            )
            if not any(
                output.record_id == consumed.record_id
                and output.slot == consumed.output_slot
                for output in consumed_realization.outputs
            ):
                raise ManagedStateError(
                    "exact consumed vector does not identify a realization output"
                )
            queue.append(
                (
                    consumed_key,
                    consumed_state,
                    event if selected_activation_event is not None else None,
                )
            )
    return tuple(
        sorted(
            nodes.values(),
            key=lambda node: (
                node.key.producer_cdef_id,
                node.key.method,
                node.state.sequence,
                node.state.realization_id,
            ),
        )
    )


def _validate_exact_realization(
    store: Store,
    operation: Any,
    state: RealizationState,
) -> RealizationRecord:
    try:
        realization = RealizationRecord.from_envelope(
            store.records.read_record(state.realization_record_id)
        )
        if realization.realization_id != state.realization_id:
            raise ManagedStateError("realization record binds a different realization")
        if realization.producer_cdef_id != operation.key.producer_cdef_id:
            raise ManagedStateError("realization record binds a different producer")
        if realization.method != operation.key.method:
            raise ManagedStateError("realization record binds a different method")
        if realization.declaration_fingerprint != operation.declaration_fingerprint:
            raise ManagedStateError("realization record binds a different declaration")
        if realization.attempt_ids != state.attempt_ids:
            raise ManagedStateError("realization record attempt lineage is inconsistent")
        if realization.checkpoint_head != state.checkpoint_head:
            raise ManagedStateError("realization checkpoint lineage is inconsistent")
        _checkpoint_source(operation, state)
        output_ids = set()
        for output in realization.outputs:
            store.records.read_spec(output.representation_id, family="representation")
            envelope = store.records.read_record(output.record_id)
            typed = (
                DataRecord.from_envelope(envelope)
                if output.record_kind == "data"
                else StoredStateRecord.from_envelope(envelope)
            )
            if (typed.realization_id, typed.output_slot) != (
                realization.realization_id,
                output.slot,
            ):
                raise ManagedStateError("managed output ownership is inconsistent")
            require_product_integrity(store.records, envelope)
            output_ids.add(output.record_id)
        execution = ExecutionRecord.from_envelope(
            store.records.read_record(realization.execution_record_id)
        )
        if execution.realization_id != realization.realization_id:
            raise ManagedStateError("execution record binds a different realization")
        if set(execution.produced_record_ids) != output_ids:
            raise ManagedStateError("execution produced lineage is inconsistent")
        if tuple(
            link.to_resolved()
            for link in execution.consumed_records
            if link.producer_cdef_id is not None
        ) != realization.consumed_records:
            raise ManagedStateError("execution consumed lineage is not exact")
        for consumed in realization.consumed_records:
            store.records.read_record(consumed.record_id)
        for consumed in execution.consumed_records:
            envelope = store.records.read_record(consumed.record_id)
            if consumed.producer_cdef_id is None:
                require_product_integrity(store.records, envelope)
        return realization
    except ManagedStateError:
        raise
    except Exception as exc:
        raise ManagedStateError("exact realization integrity validation failed") from exc


def _node_record_ids(store: Store, node: _TransferNode) -> tuple[str, ...]:
    realization = RealizationRecord.from_envelope(
        store.records.read_record(node.realization_record_id)
    )
    execution = ExecutionRecord.from_envelope(
        store.records.read_record(realization.execution_record_id)
    )
    record_ids = {
        node.realization_record_id,
        realization.execution_record_id,
        *(consumed.record_id for consumed in execution.consumed_records),
    }
    for output in realization.outputs:
        record_ids.update(
            _output_representation_record_ids(store, realization.realization_id, output)
        )
    return tuple(sorted(record_ids))


def _output_representation_record_ids(
    store: Store,
    realization_id: str,
    output: Any,
) -> tuple[str, ...]:
    """Validate and return all representations and adapter lineage for one output."""

    try:
        candidates: dict[str, DataRecord | StoredStateRecord] = {}
        record_ids: set[str] = set()
        for ref in store.records.find_records(
            kind=output.record_kind,
            realization_id=realization_id,
            output_slot=output.slot,
        ):
            envelope = store.records.read_record(ref.record_id)
            typed = (
                DataRecord.from_envelope(envelope)
                if output.record_kind == "data"
                else StoredStateRecord.from_envelope(envelope)
            )
            if (typed.realization_id, typed.output_slot) != (
                realization_id,
                output.slot,
            ):
                raise ManagedStateError(
                    "derived output representation ownership is inconsistent"
                )
            store.records.read_spec(
                typed.representation_id, family="representation"
            )
            require_product_integrity(store.records, envelope)
            candidates[ref.record_id] = typed
            record_ids.add(ref.record_id)
            record_ids.update(typed.derived_from)

        if output.record_id not in candidates:
            raise ManagedStateError(
                "exact realization output is absent from its representation set"
            )
        if candidates[output.record_id].representation_id != output.representation_id:
            raise ManagedStateError(
                "exact realization output representation is inconsistent"
            )

        derived_ids = set(candidates) - {output.record_id}
        if not derived_ids:
            return tuple(sorted(record_ids))

        adapters_by_target: dict[str, list[AdapterRecord]] = {
            record_id: [] for record_id in derived_ids
        }
        for ref in store.records.find_records(kind="adapter"):
            envelope = store.records.read_record(ref.record_id)
            payload = envelope.get("payload") or {}
            target_id = payload.get("target_record_id")
            if target_id not in derived_ids:
                continue
            adapter = AdapterRecord.from_envelope(envelope)
            source = candidates.get(adapter.source_record_id)
            target = candidates[target_id]
            if source is None:
                raise ManagedStateError(
                    "adapter-derived output leaves its exact realization slot"
                )
            if adapter.status != "ok":
                raise ManagedStateError(
                    "adapter-derived output has non-successful lineage"
                )
            if adapter.source_record_id not in target.derived_from:
                raise ManagedStateError(
                    "adapter source is absent from derived output lineage"
                )
            if (
                adapter.source_representation_id != source.representation_id
                or adapter.target_representation_id != target.representation_id
            ):
                raise ManagedStateError(
                    "adapter representation lineage is inconsistent"
                )
            adapters_by_target[target_id].append(adapter)
            record_ids.add(ref.record_id)
            record_ids.update(adapter.derived_from)
            record_ids.update(adapter.produced_records)

        reachable = {output.record_id}
        remaining = set(derived_ids)
        while remaining:
            newly_reachable = {
                target_id
                for target_id in remaining
                if any(
                    adapter.source_record_id in reachable
                    for adapter in adapters_by_target[target_id]
                )
            }
            if not newly_reachable:
                raise ManagedStateError(
                    "derived output representation has no exact adapter lineage"
                )
            reachable.update(newly_reachable)
            remaining.difference_update(newly_reachable)
        return tuple(sorted(record_ids))
    except ManagedStateError:
        raise
    except Exception as exc:
        raise ManagedStateError(
            "exact derived representation integrity validation failed"
        ) from exc


def _copy_checkpoint_closure(
    source: Store,
    destination: Store,
    nodes: tuple[_TransferNode, ...],
) -> None:
    destination_managed = ManagedOperationStore(destination)
    for node in nodes:
        if node.state.checkpoint_head is None:
            continue
        source_operation = ManagedOperationStore(source, writable=False).operation(
            node.key, node.declaration_fingerprint
        )
        source_root = _checkpoint_source(source_operation, node.state)
        realization = RealizationRecord.from_envelope(
            source.records.read_record(node.realization_record_id)
        )
        destination_operation = destination_managed.operation(
            node.key, node.declaration_fingerprint
        )
        workspace = (
            destination_operation.attempts_dir
            / f"{0:020d}-{realization.completed_attempt_id}"
        )
        target = workspace / "checkpoints" / node.state.checkpoint_head
        _copy_exact_tree(source_root, target)
        require_checkpoint_integrity(target, node.state.checkpoint_head)


def _checkpoint_source(
    operation: Any,
    state: RealizationState,
) -> Path | None:
    if state.checkpoint_head is None:
        return None
    matches = tuple(
        operation.attempts_dir.glob(f"*/checkpoints/{state.checkpoint_head}")
    )
    if not matches:
        raise ManagedStateError("completed realization checkpoint payload is missing")
    try:
        for path in matches:
            require_checkpoint_integrity(path, state.checkpoint_head)
    except Exception as exc:
        raise ManagedStateError("completed realization checkpoint integrity failed") from exc
    if any(not _trees_match(matches[0], path) for path in matches[1:]):
        raise ManagedStateError("checkpoint identity resolves to conflicting bytes")
    return matches[0]


def _copy_exact_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        if _trees_match(source, destination):
            return
        raise RecordExportError(
            "destination checkpoint already exists with different bytes"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".copying-{destination.name}-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, staging)
        _fsync_tree(staging)
        try:
            staging.replace(destination)
        except OSError:
            if destination.exists() and _trees_match(source, destination):
                shutil.rmtree(staging, ignore_errors=True)
            else:
                raise RecordExportError(
                    "destination checkpoint appeared during copy"
                )
        _fsync_directory(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _read_definition(store: Store, cdef_id: str) -> ConcreteDefinition:
    parsed = parse_cdef_id(cdef_id)
    path = Path(store.object_dir_for_cdef_id(cdef_id)) / "def.pkl"
    if not path.is_file():
        raise RecordExportError(
            "source producer definition is missing", context={"cdef_id": cdef_id}
        )
    try:
        definition = pickle_load(path)
    except Exception as exc:
        raise RecordExportError(
            "source producer definition could not be read", context={"cdef_id": cdef_id}
        ) from exc
    if not isinstance(definition, ConcreteDefinition):
        raise RecordExportError("source producer definition is malformed")
    if definition.stable_hash() != parsed.digest:
        raise RecordExportError("source producer definition identity mismatch")
    return definition


def _install_control_closure(
    destination: Store,
    nodes: tuple[_TransferNode, ...],
    *,
    root_key: OperationKey,
    root_realization_ids: frozenset[str],
    source_active_event: ActivationEvent | None,
    activate: TransferActivation,
) -> str | None:
    managed = ManagedOperationStore(destination)
    by_key: dict[OperationKey, list[_TransferNode]] = {}
    for node in nodes:
        by_key.setdefault(node.key, []).append(node)

    for key in sorted(by_key, key=lambda item: (item.producer_cdef_id, item.method)):
        group = by_key[key]
        operation_dir = managed._operation_dir(key)
        with PlatformFileLock(operation_dir / "owner.lock"):
            namespace = managed._read_namespace(key, missing_ok=True)
            if namespace is None:
                current = group[0].current_declaration_fingerprint
                transferred = {
                    node.declaration_fingerprint for node in group
                } | {current}
                generations = tuple(
                    fingerprint
                    for fingerprint in group[0].generation_order
                    if fingerprint in transferred
                )
                namespace = NamespaceState(key, current, generations, 0)
            else:
                source_order = group[0].generation_order
                generations = tuple(
                    dict.fromkeys(
                        [
                            *namespace.generations,
                            *(
                                fingerprint
                                for fingerprint in source_order
                                if any(
                                    node.declaration_fingerprint == fingerprint
                                    for node in group
                                )
                            ),
                        ]
                    )
                )
                namespace = replace(namespace, generations=generations)
            _write_json(managed._namespace_path(key), namespace.to_json())
            fingerprints = {
                node.declaration_fingerprint for node in group
            } | {namespace.current_declaration_fingerprint}
            for fingerprint in sorted(fingerprints):
                operation = managed.operation(key, fingerprint)
                control = operation._read_control(missing_ok=True)
                if control is None:
                    _write_json(
                        operation.control_path,
                        GenerationControl(fingerprint, namespace.fence_epoch).to_json(),
                    )
            for node in group:
                operation = managed.operation(key, node.declaration_fingerprint)
                sanitized = replace(
                    node.state,
                    current_attempt_id=None,
                    diagnostics=(),
                )
                _write_json(
                    operation._realization_path(sanitized.realization_id),
                    sanitized.to_json(),
                    immutable=True,
                )
            for fingerprint in sorted(fingerprints):
                operation = managed.operation(key, fingerprint)
                history = operation.history()
                latest = max(
                    history,
                    key=lambda state: (state.sequence, state.realization_id),
                    default=None,
                )
                control = operation._read_control()
                history_next_sequence = (
                    1 if latest is None else latest.sequence + 1
                )
                next_sequence = control.next_realization_sequence
                if next_sequence is None:
                    next_sequence = history_next_sequence
                else:
                    next_sequence = max(next_sequence, history_next_sequence)
                _write_json(
                    operation.control_path,
                    replace(
                        control,
                        next_realization_sequence=next_sequence,
                        latest_realization_id=(
                            None if latest is None else latest.realization_id
                        ),
                        reserved_realization_id=None,
                    ).to_json(),
                )

    activation_nodes = tuple(
        node
        for node in nodes
        if node.selected_activation_event is not None
        and not (
            activate == "inactive"
            and node.key == root_key
            and node.state.realization_id in root_realization_ids
        )
    )
    requirements: dict[OperationKey, tuple[str, int]] = {}
    for node in activation_nodes:
        assert node.selected_activation_event is not None
        requirement = (
            node.state.realization_id,
            node.selected_activation_event.sequence,
        )
        previous = requirements.setdefault(node.key, requirement)
        if previous != requirement:
            raise ManagedStateError(
                "exact transfer requires conflicting active selections for one operation"
            )

    activated_root = None
    for node in sorted(
        activation_nodes,
        key=lambda item: (
            item.key == root_key
            and item.state.realization_id in root_realization_ids,
            item.key.producer_cdef_id,
            item.key.method,
        ),
    ):
        installed = _install_compatible_activation(managed, node)
        if (
            installed
            and source_active_event is not None
            and node.key == root_key
            and node.state.realization_id == source_active_event.realization_id
        ):
            activated_root = node.state.realization_id
    return activated_root


def _install_compatible_activation(
    managed: ManagedOperationStore,
    node: _TransferNode,
) -> bool:
    event_source = node.selected_activation_event
    if event_source is None:
        return False
    operation = managed.operation(node.key, node.declaration_fingerprint)
    with PlatformFileLock(operation._lock_path):
        namespace = managed._read_namespace(node.key)
        assert namespace is not None
        if namespace.current_declaration_fingerprint != node.declaration_fingerprint:
            return False
        control = operation._read_control()
        if control.pending_realization_id is not None:
            return False
        existing_active = operation.active_event(missing_ok=True)
        if existing_active is not None:
            return (
                existing_active.realization_id == node.state.realization_id
                and existing_active.sequence == event_source.sequence
                and existing_active.realization_record_id
                == node.realization_record_id
            )
        same_sequence = tuple(
            event
            for event in operation._activation_events()
            if event.sequence == event_source.sequence
        )
        if same_sequence:
            if (
                len(same_sequence) != 1
                or same_sequence[0].realization_id != node.state.realization_id
                or same_sequence[0].realization_record_id
                != node.realization_record_id
            ):
                raise ManagedStateError(
                    "destination activation sequence conflicts with exact snapshot"
                )
            event = same_sequence[0]
        else:
            identity = (
                f"{node.key.producer_cdef_id}\0{node.key.method}\0"
                f"{node.state.realization_id}\0{event_source.sequence}"
            )
            event = ActivationEvent(
                activation_id="activation-v1-"
                + hashlib.sha256(identity.encode()).hexdigest()[:32],
                declaration_fingerprint=node.declaration_fingerprint,
                sequence=event_source.sequence,
                realization_id=node.state.realization_id,
                previous_realization_id=None,
                fence_epoch=1,
                realization_record_id=node.realization_record_id,
            )
            _write_json(operation._activation_path(event), event.to_json(), immutable=True)
        _write_json(operation.active_pointer_path, event.to_json())
    return True


__all__ = [
    "RecipeExportReport",
    "RealizationTransferReport",
    "export_recipe",
    "transfer_realizations",
]
