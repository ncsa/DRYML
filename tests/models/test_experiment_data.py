"""Public ExperimentData history contracts."""

import math

import pandas as pd
import pytest

from dryml.core import Repo, Serializable
from dryml.core.cardinality import Cardinality
from dryml.models import ExperimentData, ExperimentDataError


class HistorySubject(Serializable):
    """Small checkpoint carrier used without loading an Experiment framework."""

    def __init__(self, value=0):
        self.value = value


def _history(tmp_path):
    repo = Repo(tmp_path / "store")
    checkpoint = repo.save_object(HistorySubject(1, repo=repo), deep_capture=True)
    return repo, checkpoint, ExperimentData.get_or_create(
        checkpoint.object_projection(), repo=repo,
    )

def test_experiment_data_is_publicly_available():
    """Expose the ordinary persisted history Object from ``dryml.models``."""

    from dryml.models import ExperimentData

    assert ExperimentData.__name__ == "ExperimentData"


def test_empty_history_exposes_the_v1_table_columns(tmp_path):
    """Keep the analysis schema visible before the first checkpoint row."""

    repo, _, history = _history(tmp_path)

    assert list(history.data.columns) == [
        "row_key", "prev_row_key", "state_ref", "prev_state_ref", "time",
        "examples_seen", "total_parameters", "trainable_parameters",
        "dataset_size_kind", "dataset_size", "training_loss",
        "expected_artifacts", "artifact_inputs", "eval_artifacts",
        "evaluation_status", "failed_artifact",
    ]
    repo.close(flush=False)


def test_rows_preserve_missing_null_big_ints_and_detached_data(tmp_path):
    """Keep typed table values independent from callers and pandas inference."""

    repo, checkpoint, history = _history(tmp_path)
    row_key = history.add_row(
        expected_artifacts=("metric",), state_ref=checkpoint, prev_state_ref=None,
        time=1, examples_seen=10**100, dataset_size=Cardinality.unknown(),
        scalar_values={"metric": None, "count": 10**120},
    )
    history.update_row(
        row_key, eval_artifacts={"metric": checkpoint},
        scalar_values={"metric": None}, evaluation_status="completed",
    )

    data = history.data
    assert data.loc[0, "examples_seen"] == 10**100
    assert data.loc[0, "metric"] is None
    assert data.loc[0, "count"] == 10**120
    assert data.loc[0, "total_parameters"] is pd.NA
    data.loc[0, "eval_artifacts"]["metric"] = None
    assert history.data.loc[0, "eval_artifacts"]["metric"] == checkpoint
    assert row_key == data.loc[0, "row_key"]
    repo.close(flush=False)


def test_rows_reject_wrong_association_nonfinite_and_reserved_scalar_names(tmp_path):
    """Validate row facts before retained history can change."""

    repo, checkpoint, history = _history(tmp_path)
    other = repo.save_object(HistorySubject(2, repo=repo), deep_capture=True)
    with pytest.raises(ExperimentDataError, match="does not project"):
        history.add_row(
            expected_artifacts=(), state_ref=other, prev_state_ref=None,
            time=1, examples_seen=0,
        )
    with pytest.raises(ExperimentDataError, match="finite"):
        history.add_row(
            expected_artifacts=(), state_ref=checkpoint, prev_state_ref=None,
            time=1, examples_seen=0, scalar_values={"score": math.nan},
        )
    with pytest.raises(ExperimentDataError, match="reserved"):
        history.add_row(
            expected_artifacts=(), state_ref=checkpoint, prev_state_ref=None,
            time=1, examples_seen=0, scalar_values={"state_ref": 1},
        )
    assert len(history.data) == 0
    assert ExperimentData.scalar_column(r"a.b\\c", "x.y") == r"a\.b\\\\c.x\.y"
    repo.close(flush=False)


def test_updates_keep_inputs_separate_and_enforce_status_and_row_bounds(tmp_path):
    """Retain initial Artifact evidence apart from completed result references."""

    repo, checkpoint, history = _history(tmp_path)
    row_key = history.add_row(
        expected_artifacts=("one", "two"), state_ref=checkpoint,
        prev_state_ref=None, time=1, examples_seen=0,
    )
    history.update_row(
        row_key, artifact_inputs={"one": checkpoint}, eval_artifacts={"one": checkpoint},
        scalar_values={"one": 1}, evaluation_status="failed", failed_artifact="two",
    )
    record = history.data.loc[0]
    assert record["artifact_inputs"] == {"one": checkpoint}
    assert record["eval_artifacts"] == {"one": checkpoint}
    with pytest.raises(ExperimentDataError, match="Completed rows"):
        history.update_row(
            row_key, eval_artifacts={}, scalar_values={}, evaluation_status="completed",
        )
    history.update_row(
        row_key, eval_artifacts={"two": checkpoint}, scalar_values={"two": 2},
        evaluation_status="completed",
    )
    assert history.data.loc[0, "evaluation_status"] == "completed"
    with pytest.raises(ExperimentDataError, match="row limit"):
        history.add_row(
            expected_artifacts=tuple(f"artifact-{index}" for index in range(4_097)),
            state_ref=checkpoint, prev_state_ref=None, time=2, examples_seen=0,
        )
    repo.close(flush=False)
