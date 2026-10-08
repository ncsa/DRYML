"""Closed ExperimentData v1 reader and v2 writer contracts."""

import json
from pathlib import Path

import pytest

from dryml.core import Repo, Serializable, StateRef
from dryml.core.cardinality import Cardinality
from dryml.core.factory import FactorySpec
from dryml.core.freeze import FrozenDict, FrozenList, FrozenSet
from dryml.core.symbol import SourceSpec
from dryml.core.tensor_spec import TensorSpec
from dryml.data import GeneratorDataset
from dryml.models import ExperimentData, ExperimentDataError
from dryml.models.experiment_data import _reference_from_json, _reference_to_json


class CodecSubject(Serializable):
    """Reference-only checkpoint fixture for history payload tests."""


class CardinalityCodecSubject(Serializable):
    """Checkpoint fixture with cardinality retained in a nested Dataset definition."""

    def __init__(self, cardinality):
        self.dataset = GeneratorDataset(_empty_values, cardinality=cardinality)


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
    assert json.loads(Path(payload, "experiment_data.json").read_text())["version"] == 2
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


def test_v2_factory_reference_codec_preserves_frozen_identity_and_v1_stays_closed():
    """V2 round-trips native factories without broadening the v1 grammar."""

    source = SourceSpec.from_source(
        "def build(*args, **kwargs):\n    return args, kwargs\n",
        kind="function",
        name="build",
    )
    factory = FactorySpec(
        source,
        [1, {"members": {2, 3}}],
        frozenset(("left", "right")),
        spec=TensorSpec("float32", shape=(1,), backend="jax"),
    )

    restored = _reference_from_json(_reference_to_json(factory), version=2)

    assert restored == factory
    assert isinstance(restored.args[0], FrozenList)
    assert isinstance(restored.args[0][1], FrozenDict)
    assert isinstance(restored.args[0][1]["members"], FrozenSet)
    assert isinstance(restored.kwargs, FrozenDict)
    legacy_factory = FactorySpec(source, 1, label="legacy")
    assert _reference_from_json(
        _reference_to_json(legacy_factory), version=1,
    ) == legacy_factory
    with pytest.raises(ExperimentDataError, match="not valid in v1"):
        _reference_from_json({"type": "frozen_list", "items": []}, version=1)


@pytest.mark.parametrize(
    "cardinality",
    (
        Cardinality.finite(0), Cardinality.finite(12),
        Cardinality.INFINITE, Cardinality.UNKNOWN,
    ),
)
def test_v2_reference_codec_round_trips_cardinality(cardinality):
    """V2 preserves all Cardinality variants while v1 remains closed."""

    encoded = _reference_to_json(cardinality)

    assert _reference_from_json(encoded, version=2) == cardinality
    with pytest.raises(ExperimentDataError, match="not valid in v1"):
        _reference_from_json(encoded, version=1)


@pytest.mark.parametrize(
    "encoded",
    (
        {"type": "cardinality", "kind": "finite", "value": "01"},
        {"type": "cardinality", "kind": "finite", "value": "-1"},
        {"type": "cardinality", "kind": "infinite", "value": "0"},
        {"type": "cardinality", "kind": "other", "value": None},
    ),
)
def test_v2_reference_codec_rejects_malformed_cardinality(encoded):
    """Cardinality records admit only canonical values for their declared kind."""

    with pytest.raises(ExperimentDataError, match="cardinality|canonical"):
        _reference_from_json(encoded, version=2)


def test_history_round_trips_state_ref_with_nested_generator_cardinality(tmp_path):
    """History persists checkpoints whose Dataset definition is explicitly infinite."""

    repo = Repo(tmp_path / "store")
    checkpoint = repo.save_object(
        CardinalityCodecSubject(Cardinality.INFINITE), deep_capture=True,
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
