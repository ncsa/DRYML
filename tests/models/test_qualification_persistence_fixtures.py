"""Read source-controlled qualification inputs without live payloads."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from dryml.artifacts import Value
from dryml.managed import managed_operation
from dryml.models import ExperimentData, TrainState, TrainingObservation
from tests.qualification_fixture_support import (
    assert_value_fixture_has_no_executable_pickle_opcodes,
    verify_qualification_fixture_manifest,
)


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "qualification_reader_v1"


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


def test_experiment_data_v1_fixture_round_trips_actual_reader_input_without_payloads(tmp_path):
    """Decode, canonically re-encode, and re-decode one closed v1 history row."""

    verify_qualification_fixture_manifest(FIXTURE_ROOT)
    payload_path = FIXTURE_ROOT / "experiment_data.json"
    payload = json.loads(payload_path.read_text(encoding="ascii"))
    encoded_state = payload["rows"][0]["facts"]["state_ref"]
    from dryml.models.experiment_data import _decode_ref

    reference = _decode_ref(encoded_state)
    history = ExperimentData(reference.object_projection())
    history.restore_state_from_dir_imp(FIXTURE_ROOT, codec="pkl")
    history.save_state_to_dir_imp(tmp_path, codec="pkl")
    restored = ExperimentData(reference.object_projection())
    restored.restore_state_from_dir_imp(tmp_path, codec="pkl")

    row = restored._rows["v1:fixture:1"]
    assert row["facts"]["state_ref"] == row["facts"]["prev_state_ref"] == reference
    assert row["facts"]["examples_seen"] == 81
    assert row["evaluation_status"] == "completed"
    assert dict(row["eval_artifacts"]) == {"metric": reference}
    assert row["scalars"] == {"explicit_null": None, "integer": 7, "flag": True}
    assert restored._columns[-1] == "missing_metric"
    assert (tmp_path / "experiment_data.json").read_bytes() == payload_path.read_bytes()


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


def test_train_state_named_state_vector_restores_observation_and_historical_defaults():
    """Restore the documented named-state compatibility vector without pickle claims."""

    verify_qualification_fixture_manifest(FIXTURE_ROOT)
    fixture = json.loads((FIXTURE_ROOT / "train_state.json").read_text(encoding="ascii"))
    state_data = dict(fixture["state"])
    state_data["pending_observation"] = TrainingObservation(
        **state_data["pending_observation"]
    )
    restored = TrainState.__new__(TrainState)
    restored.__setstate__(state_data)
    historical = TrainState.__new__(TrainState)
    historical.__setstate__((2, 3, TrainState.training))

    assert fixture["fixture"] == "dryml-train-state-named-state-vector"
    assert fixture["fixture_version"] == 1
    assert restored.pending_observation.training_loss == 2.0
    assert restored.__getstate__() == state_data
    assert historical.examples_seen == historical.loss_denominator == 0


def test_qualification_fixture_generator_reproduces_the_committed_bytes(tmp_path):
    """Regenerate into a clean directory without changing committed reader inputs."""

    repository = FIXTURE_ROOT.parents[2]
    output = tmp_path / "generated"
    committed = _generated_fixture_bytes(FIXTURE_ROOT)

    subprocess.run(
        [
            sys.executable,
            str(FIXTURE_ROOT / "generate.py"),
            "--output",
            str(output),
        ],
        cwd=repository,
        check=True,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
    )

    verify_qualification_fixture_manifest(output)
    assert _generated_fixture_bytes(FIXTURE_ROOT) == committed
    assert _generated_fixture_bytes(output) == committed
