"""Read source-controlled qualification inputs without live payloads."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from dryml.artifacts import Value
from dryml.managed import managed_operation
from dryml.models import ExperimentData, TrainState, TrainingObservation
from dryml.models import ExperimentDataError
from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore
from tests.qualification_fixture_support import (
    assert_value_fixture_has_no_executable_pickle_opcodes,
    verify_qualification_fixture_manifest,
)


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "qualification_reader_v1"
pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")


def _generated_fixture_bytes(root: Path) -> dict[str, bytes]:
    """Return generated fixture payload bytes, excluding the generator source."""

    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "generate.py"
    }


class FixtureValue(Value):
    """Concrete reader used only to verify the closed Value receipt fixture."""

    @property
    def ready(self) -> bool:
        """Return whether the Value reader installed its complete receipt."""

        return self._value_is_present()

    @managed_operation()
    def compute(self, *, managed):
        """Reject computation because this fixture covers persisted receipts only."""

        raise AssertionError("fixture reader must not compute")


def test_experiment_data_v1_fixture_is_rejected_without_partial_install(tmp_path):
    """Old history/reference authority is rejected intact; current rows survive."""

    verify_qualification_fixture_manifest(FIXTURE_ROOT)
    payload_path = FIXTURE_ROOT / "experiment_data.json"
    payload = json.loads(payload_path.read_text(encoding="ascii"))
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    reference = repo.save_object(Serializable())
    history = ExperimentData(reference.object_view())
    before = history._payload()
    with pytest.raises(ExperimentDataError, match="version"):
        history.restore_state_from_dir_imp(FIXTURE_ROOT, codec="pkl")
    assert history._payload() == before
    verify_qualification_fixture_manifest(FIXTURE_ROOT)
    assert payload["version"] == 1


def test_value_receipt_fixture_round_trips_actual_value_reader_input(tmp_path):
    """Restore and re-save the committed ``value.pkl`` receipt through Value APIs."""

    receipt = FIXTURE_ROOT / "artifact_value" / "value.pkl"
    verify_qualification_fixture_manifest(FIXTURE_ROOT)
    assert_value_fixture_has_no_executable_pickle_opcodes(receipt)
    value = FixtureValue()
    value.restore_state_from_dir_imp(FIXTURE_ROOT / "artifact_value", codec="pkl")
    value.save_state_to_dir_imp(tmp_path, codec="pkl")

    assert value.value() == "fixture-result"
    assert (tmp_path / "value.pkl").read_bytes() == receipt.read_bytes()


def test_current_history_preserves_original_fixture_facts_and_missing_columns(tmp_path):
    """Keep old fidelity obligations in a current run-aware SavedNode fixture."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    reference = repo.save_object(Serializable())
    history = ExperimentData(reference.object_view())
    history.add_row(expected_artifacts=("metric",), state_ref=reference,
                    prev_state_ref=reference, time=0, examples_seen=81,
                    eval_artifacts={"metric": reference}, evaluation_status="completed",
                    scalar_values={"explicit_null": None, "integer": 7, "flag": True}, row_key="v2:fixture:1")
    history.add_row(expected_artifacts=("missing_metric",), state_ref=reference,
                    prev_state_ref=reference, time=1, examples_seen=81, row_key="v2:fixture:2")
    payload = tmp_path / "payload"
    payload.mkdir()
    current = history._payload()
    current["columns"].append({"name": "missing_metric", "kind": "scalar"})
    (payload / "experiment_data.json").write_text(json.dumps(current))
    restored = ExperimentData(reference.object_view())
    restored.restore_state_from_dir_imp(payload, codec="pkl")
    row = restored._rows["v2:fixture:1"]
    assert row["facts"]["state_ref"].graph_equal(reference)
    assert row["facts"]["prev_state_ref"].graph_equal(reference)
    assert row["facts"]["examples_seen"] == 81
    assert row["evaluation_status"] == "completed"
    assert dict(row["eval_artifacts"])["metric"].graph_equal(reference)
    expected = {"explicit_null": None, "integer": 7, "flag": True}
    assert {name: row["scalars"][name] for name in expected} == expected
    from dryml.models.experiment_data import _MISSING
    assert row["scalars"].get("missing_metric", _MISSING) is _MISSING
    assert "missing_metric" in restored._columns
    assert json.loads((payload / "experiment_data.json").read_text())["version"] == 4


def test_train_state_old_named_and_tuple_vectors_are_rejected_atomically():
    """Old progress layouts cannot manufacture current run-aware TrainState."""

    verify_qualification_fixture_manifest(FIXTURE_ROOT)
    fixture = json.loads((FIXTURE_ROOT / "train_state.json").read_text(encoding="ascii"))
    state_data = dict(fixture["state"])
    state_data["pending_observation"] = TrainingObservation(
        **state_data["pending_observation"]
    )
    restored = TrainState()
    before = restored.__getstate__()
    for old in (state_data, (2, 3, TrainState.training)):
        with pytest.raises(ValueError, match="version"):
            restored.__setstate__(old)
        assert restored.__getstate__() == before
    restored.pending_observation = TrainingObservation(**fixture["state"]["pending_observation"])
    current = restored.__getstate__()
    fresh = TrainState.__new__(TrainState)
    fresh.__setstate__(current)

    assert fixture["fixture"] == "dryml-train-state-named-state-vector"
    assert fixture["fixture_version"] == 1
    assert fresh.pending_observation.training_loss == 2.0
    assert fresh.__getstate__() == current
    assert fresh.epoch_loss_numerator == 0.0
    assert fresh.epoch_loss_denominator == 0
    assert fresh.examples_seen == fresh.loss_denominator == 0


def test_qualification_frozen_inputs_remain_immutable_after_reader_refusal(tmp_path):
    """Frozen incompatible inputs remain byte-identical across repeat refusal."""
    committed = _generated_fixture_bytes(FIXTURE_ROOT)

    test_experiment_data_v1_fixture_is_rejected_without_partial_install(tmp_path)
    verify_qualification_fixture_manifest(FIXTURE_ROOT)
    assert _generated_fixture_bytes(FIXTURE_ROOT) == committed
