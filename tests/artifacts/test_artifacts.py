import numpy as np
import pytest

from dryml.artifacts import CachedDataset, Scalar, ScalarAvg
from dryml.core2 import ConfigRef, Repo
from dryml.core2.repo import default_repo
from dryml.core2.store.dir import DirStore
from dryml.data import ArrayDataset, NpyFileDataset


def test_scalar_artifact_compute_read_and_reload(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    scalar = Scalar(3.5)

    repo.save_object(scalar)
    assert scalar.compute(repo=repo) == 3.5
    assert scalar.read(repo=repo) == 3.5

    repo2 = Repo(stores=DirStore(store.base_dir))
    loaded = repo2.load_object(scalar.definition)

    assert loaded.read(repo=repo2) == 3.5


def test_scalar_avg_computes_dataset_mean(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    ds = ArrayDataset(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    avg = ScalarAvg(ds)

    repo.save_object(avg)

    assert avg.compute(repo=repo) == pytest.approx(2.0)
    assert avg.read(repo=repo) == pytest.approx(2.0)


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
    assert accuracy.read(repo=repo) == pytest.approx(0.75)


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
    assert accuracy.read(repo=Repo()) == pytest.approx(0.75)


def test_cached_dataset_writes_npy_files_for_npy_file_dataset(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    source = ArrayDataset(np.array([[1, 2], [3, 4]], dtype=np.int32))
    cached = CachedDataset(source)

    repo.save_object(cached)
    cached.compute(repo=repo)

    files = sorted(p.name for p in tmp_path.glob("store/objects/*/*/*.npy"))
    assert files == ["00000000.npy", "00000001.npy"]

    repo.set_config("cache.root", repo.location(cached))
    ds = NpyFileDataset(ConfigRef("cache.root"), repo=repo)
    assert [item.tolist() for item in ds] == [[1, 2], [3, 4]]


def test_cached_dataset_supports_default_repo_location_flow(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo()
    source = ArrayDataset(np.array([[5, 6]], dtype=np.int32))
    cached = CachedDataset(source)

    with default_repo(repo):
        cached.save(store=store)
        cached.compute()
        repo.set_config("cache.root", cached.location)
        ds = NpyFileDataset(ConfigRef("cache.root"))

    assert [item.tolist() for item in ds] == [[5, 6]]
