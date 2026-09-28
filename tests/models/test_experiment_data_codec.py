"""Closed ExperimentData v1 JSON codec contracts."""

from pathlib import Path

import pytest

from dryml.core import Repo, Serializable
from dryml.models import ExperimentData, ExperimentDataError


class CodecSubject(Serializable):
    """Reference-only checkpoint fixture for history payload tests."""


def _row(tmp_path):
    repo = Repo(tmp_path / "store")
    checkpoint = repo.save_object(CodecSubject(repo=repo), deep_capture=True)
    history = ExperimentData.get_or_create(checkpoint.object_projection(), repo=repo)
    history.add_row(
        expected_artifacts=("one", "two"), state_ref=checkpoint,
        prev_state_ref=None, time=0, examples_seen=0, scalar_values={"value": True},
    )
    return repo, checkpoint, history


def test_codec_round_trip_and_rejected_decode_leave_rows_unchanged(tmp_path):
    """Validate the complete JSON tree before replacing retained rows."""

    repo, checkpoint, history = _row(tmp_path)
    payload = tmp_path / "payload"
    payload.mkdir()
    history.save_state_to_dir_imp(payload, codec="pkl")
    restored = ExperimentData(checkpoint.object_projection())
    restored.restore_state_from_dir_imp(payload, codec="pkl")
    assert restored.data.to_dict("records") == history.data.to_dict("records")

    before = restored.data.to_dict("records")
    Path(payload, "experiment_data.json").write_text(
        '{"format":"dryml-experiment-data","format":"wrong"}', encoding="utf-8",
    )
    with pytest.raises(ExperimentDataError, match="Duplicate JSON"):
        restored.restore_state_from_dir_imp(payload, codec="pkl")
    assert restored.data.to_dict("records") == before
    repo.close(flush=False)


def test_codec_rejects_unknown_status_and_incomplete_completed_row(tmp_path):
    """Reject status variants before they can be interpreted as success."""

    repo, checkpoint, history = _row(tmp_path)
    payload = tmp_path / "payload"
    payload.mkdir()
    history.save_state_to_dir_imp(payload, codec="pkl")
    raw = Path(payload, "experiment_data.json").read_text(encoding="utf-8")
    raw = raw.replace('"evaluation_status":"pending"', '"evaluation_status":"completed"')
    Path(payload, "experiment_data.json").write_text(raw, encoding="utf-8")
    with pytest.raises(ExperimentDataError, match="Completed rows"):
        ExperimentData(checkpoint.object_projection()).restore_state_from_dir_imp(payload, codec="pkl")
    repo.close(flush=False)
