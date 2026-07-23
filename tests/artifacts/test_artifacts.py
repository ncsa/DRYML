import numpy as np
import pytest

import dryml.artifacts
from dryml.artifacts import Scalar, ScalarAvg
from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.records import DataRecord


def test_scalar_artifact_compute_value_and_reload(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    scalar = Scalar(3.5)

    repo.save_object(scalar)
    assert scalar.compute(repo=repo) == 3.5
    assert scalar.value == 3.5

    repo2 = Repo(stores=DirStore(store.base_dir))
    loaded = repo2.load_object(scalar.definition)

    assert loaded.value == 3.5


def test_scalar_avg_computes_dataset_mean(tmp_path):
    store = DirStore(tmp_path / "store")
    ds = ArrayDataset(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    avg = ScalarAvg(ds)

    result = avg.compute(store=store)

    assert result.action == "start"
    assert avg.read(store=store) == pytest.approx(2.0)
    record = DataRecord.from_envelope(
        store.records.read_record(result.outputs["value"].record_id)
    )
    assert record.output_slot == "value"


def test_scalar_avg_compute_without_store_remains_direct():
    ds = ArrayDataset(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    avg = ScalarAvg(ds)

    assert avg.compute() == pytest.approx(2.0)
    assert avg.value == pytest.approx(2.0)


def test_scalar_avg_reload_reads_managed_value_without_object_state(tmp_path):
    store = DirStore(tmp_path / "store")
    ds = ArrayDataset(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    avg = ScalarAvg(ds)

    avg.compute(store=store)
    Repo(store).save_definition(avg.definition)

    repo2 = Repo(stores=DirStore(store.base_dir))
    loaded = repo2.load(avg.definition, restore_state=False)
    assert loaded.read(store=repo2.stores[0]) == pytest.approx(2.0)


def test_draft_accuracy_artifact_is_not_public():
    assert not hasattr(dryml.artifacts, "Accuracy")
