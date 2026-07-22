from __future__ import annotations

import pytest

from dryml.managed import EventBuffer, OperationEvent, ProgressSnapshot


def test_progress_snapshot_is_bounded_and_validated():
    progress = ProgressSnapshot(current=2, total=5, message="batch", metrics={"loss": 0.5})

    assert progress.fraction == pytest.approx(0.4)
    assert progress.to_json()["metrics"] == {"loss": 0.5}
    with pytest.raises(ValueError):
        ProgressSnapshot(current=6, total=5)
    with pytest.raises(ValueError):
        ProgressSnapshot(current=1, metrics={str(index): index for index in range(17)})


def test_event_buffer_coalesces_progress_and_bounds_non_progress_history():
    events = EventBuffer(max_events=8)
    for index in range(1000):
        events.append(OperationEvent.progress(index + 1, current=index, total=1000))
    for index in range(20):
        events.append(OperationEvent(index + 1001, "safe_point"))

    snapshot = events.snapshot()
    assert len(snapshot) == 8
    assert sum(event.kind == "progress" for event in snapshot) <= 1
    assert snapshot[-1].kind == "safe_point"


def test_event_rejects_unbounded_or_unknown_payloads():
    with pytest.raises(ValueError, match="kind"):
        OperationEvent(1, "unknown")
    with pytest.raises(ValueError, match="payload"):
        OperationEvent(1, "progress", payload={"items": list(range(100))})
