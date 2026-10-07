"""Native training-input bridges retain Dataset-owned traversal semantics."""

from __future__ import annotations

import json
import numpy as np
import os
from pathlib import Path
import pytest
import subprocess
import sys

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Batch, Dataset, Take


class TrackingDataset(Dataset):
    """Re-iterable paired source that records each cursor's lifecycle."""

    def __init__(self, values):
        self.values = tuple(values)
        self.opens = 0
        self.closes = []
        super().__init__(spec=(
            TensorSpec("float32", shape=(1,), backend="numpy"),
            TensorSpec("float32", shape=(1,), backend="numpy"),
        ))

    def __iter__(self):
        identifier = self.opens
        self.opens += 1
        try:
            yield from self.values
        finally:
            self.closes.append(identifier)

    def __len__(self):
        return Cardinality.finite(len(self.values))


def _batches():
    source = TrackingDataset((
        (np.asarray([0.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32)),
        (np.asarray([1.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32)),
        (np.asarray([2.0], dtype=np.float32), np.asarray([4.0], dtype=np.float32)),
    ))
    return source, Batch(source, 2)


def test_prepared_dataset_plans_once_reopens_independent_closeable_cursors():
    """Preparation is inert and each prepared traversal owns a fresh graph cursor."""
    from dryml.data.native import PreparedDataset

    source, dataset = _batches()
    prepared = dataset.prepare()

    assert isinstance(prepared, PreparedDataset)
    assert prepared.execution_level == "stream"
    assert source.opens == 0
    first = prepared.iterator()
    second = prepared.iterator()
    assert next(first)[0].shape == (2, 1)
    assert next(second)[0].shape == (2, 1)
    first.close()
    second.close()

    assert sorted(source.closes) == [0, 1]


def test_prepared_dataset_reports_eager_fallback_for_unqualified_operator():
    """Unqualified Dataset operators retain eager iteration rather than failing a bridge."""
    from dryml.data.native import PreparedDataset

    source, dataset = _batches()
    prepared = PreparedDataset(Take(dataset, 1))

    assert prepared.execution_level == "eager"
    cursor = prepared.iterator()
    assert next(cursor)[0].shape == (2, 1)
    cursor.close()
    assert source.closes == [0]


def test_native_training_cursor_closes_on_preparation_failure_without_progress_side_effects():
    """A native-input conversion failure closes its cursor before any trainer can update."""
    from dryml.data.native import PreparedDataset

    class FailingPreparation:
        def prepare(self, x, y):
            del x, y
            raise RuntimeError("conversion failed")

    source, dataset = _batches()
    cursor = PreparedDataset(dataset).training_batches(FailingPreparation())

    with pytest.raises(RuntimeError, match="conversion failed"):
        next(cursor)

    assert source.closes == [0]


def test_training_input_modules_defer_framework_and_legacy_adapter_imports():
    """The common prepared seam and plugin modules stay dependency-light until used."""
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
import dryml.data.native
import dryml.tf.training_data
import dryml.torch.training_data
import dryml.jax.training_data
names = ('tensorflow', 'torch', 'jax', 'dryml.data.tf', 'dryml.data.torch')
print(json.dumps(sorted(name for name in names if name in sys.modules)))
""",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(result.stdout) == []


def test_tensorflow_native_dataset_preserves_prepared_batches_and_closes_source():
    """A real tf.data consumer receives each authored batch without rebatching."""
    tf = pytest.importorskip("tensorflow")
    from dryml.data.native import PreparedDataset
    from dryml.models.tf import BasicTraining
    from dryml.models.utils import TrainingPreparation
    from dryml.tf.training_data import as_training_dataset

    source, dataset = _batches()
    preparation = TrainingPreparation.from_specs(
        BasicTraining(verbose=0), dataset.spec[0], dataset.spec[1], "tf"
    )
    native = as_training_dataset(PreparedDataset(dataset), preparation)
    values = list(native.as_numpy_iterator())

    assert [x.shape for x, _ in values] == [(2, 1), (1, 1)]
    assert [y[:, 0].tolist() for _, y in values] == [[0.0, 2.0], [4.0]]
    assert source.closes == [0]
    assert all(value.dtype == np.dtype("float32") for pair in values for value in pair)


def test_torch_iterable_dataset_and_single_process_loader_preserve_batches():
    """Torch receives authored batches without a second loader batch axis."""
    torch = pytest.importorskip("torch")
    from dryml.data.native import PreparedDataset
    from dryml.models.torch import Training
    from dryml.models.utils import TrainingPreparation
    from dryml.torch.training_data import as_data_loader, as_training_dataset

    source, dataset = _batches()
    preparation = TrainingPreparation.from_specs(
        Training(verbose=0), dataset.spec[0], dataset.spec[1], "torch"
    )
    native = as_training_dataset(PreparedDataset(dataset), preparation)
    assert isinstance(native, torch.utils.data.IterableDataset)
    loader = as_data_loader(native)
    values = list(loader)

    assert [tuple(x.shape) for x, _ in values] == [(2, 1), (1, 1)]
    assert [y[:, 0].tolist() for _, y in values] == [[0.0, 2.0], [4.0]]
    assert source.closes == [0]

    source, dataset = _batches()
    preparation = TrainingPreparation.from_specs(
        Training(verbose=0), dataset.spec[0], dataset.spec[1], "torch"
    )
    cursor = iter(as_training_dataset(PreparedDataset(dataset), preparation))
    next(cursor)
    cursor.close()
    assert source.closes == [0]
