"""Generate the small, trusted Stage 5+7 v1 reader fixtures.

Run with ``PYTHONPATH=src python tests/fixtures/stage5_7_v1/generate.py --output
<empty-directory>`` from the repository root after intentionally changing the
documented payload contract. Omitting ``--output`` deliberately regenerates the
source-controlled fixture directory.
The fixtures contain only closed JSON records or a dill encoding of a plain Value
envelope; they do not embed model, Dataset, Store, or executable user payloads.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from dryml.core import StateRef, Template, TemplateBundle
from dryml.core.cardinality import Cardinality
from dryml.core.utils.general import pickle_save
from dryml.models import ExperimentData, ParameterCounts


ROOT = Path(__file__).parent
REFERENCE = {
    "codec_version": 1,
    "object": {
        "codec_version": 1,
        "definition": {
            "codec_version": 2,
            "root": "n0",
            "nodes": [{
                "label": "n0",
                "cls": {"module": "dryml.core.object", "qualname": "Serializable"},
                "parameters": {"kind": "dict", "items": []},
                "stateful_role": True,
            }],
        },
        "objects": [{
            "path": {"schema_version": 3, "segments": []},
            "object_id": {
                "namespace": ["fixture"],
                "nonce": "00000000-0000-0000-0000-000000000001",
            },
        }],
    },
    "states": [{
        "path": {"schema_version": 3, "segments": []},
        "state": "pkl-" + "0" * 64,
    }],
}


def _output_root(destination: Path | str | None) -> Path:
    """Return a deliberate fixture destination, rejecting nonempty explicit paths."""

    if destination is None:
        return ROOT
    output = Path(destination)
    if output.exists():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise ValueError("An explicit fixture output directory must be empty and non-symlinked.")
    else:
        output.mkdir(parents=True)
    return output


def _write_json(path: Path, payload: object, *, pretty: bool = False) -> None:
    """Write one ASCII JSON payload with platform-independent newline bytes."""

    options = {"ensure_ascii": True, "allow_nan": False}
    if pretty:
        options.update(indent=2, sort_keys=True)
        content = json.dumps(payload, **options) + "\n"
    else:
        options["separators"] = (",", ":")
        content = json.dumps(payload, **options)
    path.write_bytes(content.encode("ascii"))


def main(destination: Path | str | None = None) -> None:
    """Write deterministic closed reader inputs and their provenance manifest.

    Args:
        destination: Optional empty non-symlinked output directory. Omitting it
            intentionally updates the source-controlled fixture directory.

    Raises:
        ValueError: If an explicit destination is nonempty, a symlink, or not a
            directory.
    """

    output = _output_root(destination)

    reference = StateRef.from_data(REFERENCE)
    history = ExperimentData(reference.object_projection())
    history.add_row(
        expected_artifacts=("metric",),
        state_ref=reference,
        prev_state_ref=reference,
        time=123,
        examples_seen=81,
        parameters=ParameterCounts(10, 6),
        dataset_size=Cardinality.finite(81),
        training_loss=2.0,
        eval_artifacts={"metric": reference},
        scalar_values={"explicit_null": None, "integer": 7, "flag": True},
        evaluation_status="completed",
        row_key="v1:fixture:1",
        prev_row_key="v1:fixture:0",
    )
    payload = history._payload()
    # v1 represents an omitted scalar cell with a declared-but-absent column.
    payload["columns"].append({"name": "missing_metric", "kind": "scalar"})
    history_path = output / "experiment_data.json"
    _write_json(history_path, payload)

    value_dir = output / "artifact_value"
    value_dir.mkdir(exist_ok=True)
    value_path = value_dir / "value.pkl"
    pickle_save({
        "format": "dryml.artifacts.value",
        "version": 1,
        "present": True,
        "result": "fixture-result",
    }, value_path)

    train_state = {
        "epoch": 2,
        "step": 3,
        "phase": "training",
        "examples_seen": 81,
        "loss_numerator": 0.0,
        "loss_denominator": 0,
        "next_batch": 4,
        "target_epoch": 3,
        "pending_epoch_postlude": None,
        "pending_epoch_postlude_phase": None,
        "pending_epoch_metrics": None,
        "safe_point_sequence": 1,
        "pending_observation": {
            "examples_seen": 81,
            "loss_numerator": 162.0,
            "loss_denominator": 81,
            "epoch": 2,
            "next_batch": 4,
            "sequence": 1,
        },
        "pending_observation_time": 0,
        "pending_observation_prev_state_ref": None,
        "pending_observation_prev_row_key": None,
        "pending_observation_attempt_id": "fixture-attempt",
        "pending_observation_terminal": False,
    }
    train_path = output / "train_state.json"
    _write_json(train_path, {
        "fixture": "dryml-train-state-named-state-vector",
        "fixture_version": 1,
        "state": train_state,
    }, pretty=True)

    template_path = output / "template_bundle.json"
    bundle = TemplateBundle({"score": Template.from_value(1)})
    _write_json(template_path, bundle.to_data())

    entries = {
        "artifact_value/value.pkl": {
            "format": "dryml.artifacts.value",
            "sha256": sha256(value_path.read_bytes()).hexdigest(),
            "version": 1,
        },
        "experiment_data.json": {
            "format": "dryml-experiment-data",
            "sha256": sha256(history_path.read_bytes()).hexdigest(),
            "version": 1,
        },
        "template_bundle.json": {
            "format": "dryml-template",
            "kind": "template-bundle",
            "sha256": sha256(template_path.read_bytes()).hexdigest(),
            "version": 1,
        },
        "train_state.json": {
            "fixture": "dryml-train-state-named-state-vector",
            "sha256": sha256(train_path.read_bytes()).hexdigest(),
            "fixture_version": 1,
        },
    }
    _write_json(
        output / "manifest.json",
        {
            "schema": "dryml-stage5_7-fixtures",
            "version": 1,
            "generator": "generate.py",
            "provenance": "Closed code-free reader vectors; TrainState is a named __setstate__ compatibility vector, not a portable pickle codec claim.",
            "fixtures": entries,
        },
        pretty=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="empty directory for generated fixtures")
    main(parser.parse_args().output)
