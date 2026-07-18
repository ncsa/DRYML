"""Concurrency-stable read-only resolution of logical managed outputs."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

from dryml.core2.store.store import Store
from dryml.core2.symbol import resolve_symbol
from dryml.records import RealizationRecord, ResolvedRecord, require_product_integrity

from .descriptor import ManagedMethod
from .errors import (
    ConcurrentManagedActivationError,
    ManagedStateError,
    MissingManagedOutputError,
)
from .refs import ManagedOutputRef
from .state import OperationKey, declaration_fingerprint
from .store import ManagedOperationStore, resolve_managed_store


def resolve_inputs(
    refs: Iterable[ManagedOutputRef],
    *,
    store: Store | None = None,
    repo: Any | None = None,
    max_attempts: int = 3,
) -> tuple[ResolvedRecord, ...]:
    """Resolve one ordered input vector with a bounded double collect.

    Every collection uses the same selected Store and preserves caller order.
    A changed collection retries from the beginning; no mixed vector is ever
    returned or persisted.
    """

    refs = tuple(refs)
    if any(not isinstance(ref, ManagedOutputRef) for ref in refs):
        raise TypeError("managed inputs must be ManagedOutputRef values")
    if type(max_attempts) is not int or max_attempts < 1 or max_attempts > 16:
        raise ValueError("max_attempts must be between 1 and 16")
    selected = resolve_managed_store(repo, store=store)
    for _attempt in range(max_attempts):
        first = _collect_input_vector(refs, selected)
        second = _collect_input_vector(refs, selected)
        if first == second:
            _validate_vector_integrity(second, selected)
            return first
    raise ConcurrentManagedActivationError(
        f"managed inputs did not yield a stable active vector after {max_attempts} attempts"
    )


def resolve_output(
    ref: ManagedOutputRef,
    *,
    store: Store | None = None,
    repo: Any | None = None,
) -> ResolvedRecord:
    """Resolve one logical output without materializing or computing its producer."""

    return resolve_inputs((ref,), store=store, repo=repo)[0]


def _collect_input_vector(
    refs: tuple[ManagedOutputRef, ...],
    store: Store,
) -> tuple[ResolvedRecord, ...]:
    return tuple(_collect_output(ref, store) for ref in refs)


def _collect_output(ref: ManagedOutputRef, store: Store) -> ResolvedRecord:
    producer = ref.producer
    key = OperationKey.from_producer(producer, ref.method)
    try:
        cls = resolve_symbol(producer.cls)
        descriptor = inspect.getattr_static(cls, ref.method)
    except (AttributeError, ImportError) as exc:
        raise MissingManagedOutputError(
            f"managed output producer has no current declaration for {ref.method!r}"
        ) from exc
    if not isinstance(descriptor, ManagedMethod):
        raise MissingManagedOutputError(
            f"managed output producer method {ref.method!r} is not managed"
        )
    outputs = descriptor.output_declarations(producer)
    if outputs.get(ref.slot) is None:
        raise MissingManagedOutputError(
            f"managed output slot {ref.slot!r} is absent from the current declaration"
        )
    fingerprint = declaration_fingerprint(
        ref.method,
        descriptor.declaration,
        producer=producer,
    )
    managed_store = ManagedOperationStore(store)
    namespace = managed_store._read_namespace(key, missing_ok=True)
    if namespace is None:
        raise MissingManagedOutputError("managed output has no active realization")
    if namespace.current_declaration_fingerprint != fingerprint:
        raise MissingManagedOutputError("managed output declaration generation is not current")
    operation = managed_store.operation(key, fingerprint)
    event = operation.active_event(missing_ok=True)
    if event is None or event.realization_record_id is None:
        raise MissingManagedOutputError("managed output has no complete active realization")
    try:
        realization = RealizationRecord.from_envelope(
            store.records.read_record(event.realization_record_id)
        )
    except Exception as exc:
        raise ManagedStateError("active realization record could not be validated") from exc
    if realization.realization_id != event.realization_id:
        raise ManagedStateError("active event and realization record disagree")
    output = next((item for item in realization.outputs if item.slot == ref.slot), None)
    if output is None:
        raise MissingManagedOutputError(
            f"active realization does not contain output slot {ref.slot!r}"
        )
    store.records.read_record(output.record_id)
    return ResolvedRecord(
        producer_cdef_id=key.producer_cdef_id,
        method=key.method,
        declaration_fingerprint=fingerprint,
        activation_generation=event.sequence,
        realization_id=event.realization_id,
        output_slot=ref.slot,
        record_id=output.record_id,
    )


def _validate_vector_integrity(vector: tuple[ResolvedRecord, ...], store: Store) -> None:
    for record_id in dict.fromkeys(item.record_id for item in vector):
        require_product_integrity(store.records, store.records.read_record(record_id))


__all__ = ["resolve_inputs", "resolve_output"]
