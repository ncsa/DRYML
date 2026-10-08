"""Closed ExperimentData legacy readers and v3 writer contracts."""

import json
from pathlib import Path

import pytest

from dryml.core import Repo, Serializable, StateRef
from dryml.core.cardinality import Cardinality
from dryml.core.dtype import DType
from dryml.data import GeneratorDataset
from dryml.models import ExperimentData, ExperimentDataError


class CodecSubject(Serializable):
    """Reference-only checkpoint fixture for history payload tests."""


class SemanticCodecSubject(Serializable):
    """Checkpoint fixture with semantic values retained in its definition."""

    def __init__(self, cardinality, dtype):
        self.dataset = GeneratorDataset(_empty_values, cardinality=cardinality)
        self.dtype = dtype


def _empty_values():
    return iter(())


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
    assert json.loads(Path(payload, "experiment_data.json").read_text())["version"] == 3
    restored = ExperimentData(checkpoint.object_projection())
    restored.restore_state_from_dir_imp(payload, codec="pkl")
    assert restored.data.to_dict("records") == history.data.to_dict("records")

    Path(payload, "experiment_data.json").write_text(
        json.dumps(history._payload(version=2)), encoding="utf-8",
    )
    legacy = ExperimentData(checkpoint.object_projection())
    legacy.restore_state_from_dir_imp(payload, codec="pkl")
    assert legacy.data.to_dict("records") == history.data.to_dict("records")

    before = restored.data.to_dict("records")
    Path(payload, "experiment_data.json").write_text(
        '{"format":"dryml-experiment-data","format":"wrong"}', encoding="utf-8",
    )
    with pytest.raises(ExperimentDataError, match="Duplicate JSON"):
        restored.restore_state_from_dir_imp(payload, codec="pkl")
    assert restored.data.to_dict("records") == before
    with pytest.raises(ExperimentDataError, match="version"):
        history._payload(version=True)
    repo.close(flush=False)


def test_history_round_trips_state_ref_with_semantic_definition_values(tmp_path):
    """History persists checkpoints with Dataset cardinality and canonical dtype."""

    repo = Repo(tmp_path / "store")
    checkpoint = repo.save_object(
        SemanticCodecSubject(Cardinality.INFINITE, DType("float", 32)),
        deep_capture=True,
    )
    history = ExperimentData.get_or_create(checkpoint.object_projection(), repo=repo)
    history.add_row(
        expected_artifacts=(), state_ref=checkpoint, prev_state_ref=None,
        time=0, examples_seen=0,
    )
    payload = tmp_path / "payload"
    payload.mkdir()

    history.save_state_to_dir_imp(payload, codec="pkl")
    restored = ExperimentData(checkpoint.object_projection())
    restored.restore_state_from_dir_imp(payload, codec="pkl")

    assert isinstance(restored.data.iloc[0].state_ref, StateRef)
    assert restored.data.iloc[0].state_ref == checkpoint
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
