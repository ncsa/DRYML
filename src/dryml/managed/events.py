"""Bounded invocation-local managed operation events and progress snapshots."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


EVENT_KINDS = frozenset({
    "started",
    "progress",
    "safe_point",
    "checkpoint",
    "completed",
    "failed",
    "interrupted",
})
MAX_EVENT_PAYLOAD_ITEMS = 32
MAX_PROGRESS_METRICS = 16


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """One bounded progress observation suitable for status and callbacks."""

    current: int | float
    total: int | float | None = None
    message: str | None = None
    metrics: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _number(self.current, "current")
        if self.current < 0:
            raise ValueError("progress current must be non-negative")
        if self.total is not None:
            _number(self.total, "total")
            if self.total < 0 or self.current > self.total:
                raise ValueError("progress total must be non-negative and not below current")
        if self.message is not None and (
            not isinstance(self.message, str) or len(self.message) > 256
        ):
            raise ValueError("progress message must be a string of at most 256 characters")
        if not isinstance(self.metrics, Mapping) or len(self.metrics) > MAX_PROGRESS_METRICS:
            raise ValueError(f"progress metrics must contain at most {MAX_PROGRESS_METRICS} entries")
        metrics = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name or len(name) > 64:
                raise ValueError("progress metric names must be bounded non-empty strings")
            _number(value, f"metric {name}")
            metrics[name] = value
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    @property
    def fraction(self) -> float | None:
        """Return normalized completion, or ``None`` without a positive total."""

        if self.total in {None, 0}:
            return None
        return float(self.current) / float(self.total)

    def to_json(self) -> dict[str, Any]:
        """Return the strict bounded JSON representation."""

        return {
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any] | None) -> "ProgressSnapshot | None":
        """Decode persisted progress, preserving ``None`` for no observation."""

        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {"current", "total", "message", "metrics"}:
            raise ValueError("progress snapshot fields are malformed")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class OperationEvent:
    """One bounded coordinator event emitted by managed operation code."""

    sequence: int
    kind: str
    progress_snapshot: ProgressSnapshot | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("event sequence must be a positive integer")
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unsupported event kind {self.kind!r}")
        if self.progress_snapshot is not None and not isinstance(self.progress_snapshot, ProgressSnapshot):
            raise TypeError("event progress must be a ProgressSnapshot")
        if self.kind == "progress" and self.progress_snapshot is None:
            # A generic progress marker remains useful to callback tests and
            # implementations that do not expose numeric work units.
            pass
        payload = _bounded_payload(self.payload)
        object.__setattr__(self, "payload", MappingProxyType(payload))

    @classmethod
    def progress_event(
        cls,
        sequence: int,
        *,
        current: int | float,
        total: int | float | None = None,
        message: str | None = None,
        metrics: Mapping[str, int | float] | None = None,
    ) -> "OperationEvent":
        """Build a numeric progress event."""

        return cls(
            sequence,
            "progress",
            progress_snapshot=ProgressSnapshot(current, total, message, metrics or {}),
        )

    # Keep the concise authoring spelling while retaining the ``progress`` field.
    progress = progress_event


class EventBuffer:
    """Bound event history while replacing superseded progress observations."""

    def __init__(self, *, max_events: int = 32):
        if type(max_events) is not int or max_events < 1 or max_events > 1024:
            raise ValueError("max_events must be between 1 and 1024")
        self.max_events = max_events
        self._events: deque[OperationEvent] = deque(maxlen=max_events)

    def append(self, event: OperationEvent) -> None:
        """Append an event, coalescing all prior progress observations."""

        if not isinstance(event, OperationEvent):
            raise TypeError("event must be an OperationEvent")
        if event.kind == "progress":
            self._events = deque(
                (item for item in self._events if item.kind != "progress"),
                maxlen=self.max_events,
            )
        self._events.append(event)

    def snapshot(self) -> tuple[OperationEvent, ...]:
        """Return the current bounded event history."""

        return tuple(self._events)


def _number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"progress {name} must be a finite number")


def _bounded_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or len(value) > MAX_EVENT_PAYLOAD_ITEMS:
        raise ValueError("event payload must be a bounded mapping")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 64:
            raise ValueError("event payload keys must be bounded non-empty strings")
        if isinstance(item, (list, tuple, dict, set)):
            raise ValueError("event payload values must be bounded scalar values")
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("event payload values must be JSON scalar values")
        if isinstance(item, str) and len(item) > 256:
            raise ValueError("event payload strings must contain at most 256 characters")
        result[key] = item
    return result


__all__ = ["EVENT_KINDS", "EventBuffer", "OperationEvent", "ProgressSnapshot"]
