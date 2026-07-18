import numpy as np
import pytest

from dryml.artifacts import Accuracy, Scalar, ScalarAvg
from dryml.core2 import Repo
from dryml.core2.store.dir import DirStore
from dryml.core2.tensor_spec import TensorSpec
from dryml.data import ArgMax, ArrayDataset, Map, Pipe, Project, Select
from dryml.models import Model


class ParityClassifier(Model):
    def __init__(self):
        self.output_spec = TensorSpec("float32", shape=(2,), backend="numpy")

    def __call__(self, x):
        labels = np.asarray(x, dtype=np.int64) % 2
        return np.eye(2, dtype=np.float32)[labels]


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
    repo = Repo(stores=store)
    ds = ArrayDataset(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    avg = ScalarAvg(ds)

    repo.save_object(avg)

    assert avg.compute(repo=repo) == pytest.approx(2.0)
    assert avg.value == pytest.approx(2.0)


def test_scalar_avg_compute_without_store_caches_value():
    repo = Repo()
    ds = ArrayDataset(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    avg = ScalarAvg(ds)

    assert avg.compute(repo=repo) == pytest.approx(2.0)
    assert avg.value == pytest.approx(2.0)


def test_scalar_avg_compute_before_save_persists_cached_value(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    ds = ArrayDataset(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    avg = ScalarAvg(ds)

    assert avg.compute(repo=repo) == pytest.approx(2.0)
    repo.save_object(avg)

    repo2 = Repo(stores=DirStore(store.base_dir))
    loaded = repo2.load_object(avg.definition)
    assert loaded.value == pytest.approx(2.0)


def test_accuracy_computes_projected_argmax_pipeline(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    x = np.array([0, 1, 2, 3], dtype=np.int64)
    y = np.array([0, 1, 0, 0], dtype=np.int64)
    ds = ArrayDataset((x, y))
    predictions = Map(
        ds,
        Project(
            Pipe(Select(0), ParityClassifier(), ArgMax()),
            Select(1),
        ),
    )
    accuracy = Accuracy(predictions, path_x=0, path_y=1)

    repo.save_object(accuracy)

    assert accuracy.compute(repo=repo) == pytest.approx(0.75)
    assert accuracy.value == pytest.approx(0.75)


def test_accuracy_compute_without_store_caches_value():
    x = np.array([0, 1, 2, 3], dtype=np.int64)
    y = np.array([0, 1, 0, 0], dtype=np.int64)
    ds = ArrayDataset((x, y))
    predictions = Map(
        ds,
        Project(
            Pipe(Select(0), ParityClassifier(), ArgMax()),
            Select(1),
        ),
    )
    accuracy = Accuracy(predictions, path_x=0, path_y=1)

    assert accuracy.compute(repo=Repo()) == pytest.approx(0.75)
    assert accuracy.value == pytest.approx(0.75)
