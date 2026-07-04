"""Small fake execution helpers for tests and examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dryml.records.execution import ExecutionRecord


def fake_execution_record(
    *,
    operation_id: str,
    backend: Mapping[str, Any] | None = None,
    status: str = "ok",
    execution_kind: str = "python",
    consumed_records: Sequence[str | Mapping[str, Any]] = (),
    produced_records: Sequence[str | Mapping[str, Any]] = (),
    **fields: Any,
) -> ExecutionRecord:
    """Build a metadata-only fake execution record for tests/examples."""

    return ExecutionRecord(
        execution_kind=execution_kind,
        operation_id=operation_id,
        backend=backend or {"name": "dryml.fake", "kind": "fake"},
        status=status,
        consumed_records=tuple(consumed_records),
        produced_records=tuple(produced_records),
        **fields,
    )


__all__ = ["fake_execution_record"]
