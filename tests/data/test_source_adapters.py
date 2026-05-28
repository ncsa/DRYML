import numpy as np

from dryml.core2 import ConfigRef, Repo
from dryml.core2.tensor_spec import TensorSpec
from dryml.data import NpyFileDataset


def test_npy_file_dataset_loads_sorted_files(tmp_path):
    np.save(tmp_path / "b.npy", np.array([2, 3], dtype=np.int32))
    np.save(tmp_path / "a.npy", np.array([0, 1], dtype=np.int32))

    ds = NpyFileDataset(tmp_path)

    assert ds.spec == TensorSpec("int32", shape=(2,), backend="numpy")
    assert [item.tolist() for item in ds] == [[0, 1], [2, 3]]


def test_npy_file_dataset_accepts_config_ref_root(tmp_path):
    np.save(tmp_path / "x.npy", np.array([4, 5], dtype=np.int32))
    repo = Repo(config={"data.root": str(tmp_path)})

    ds = NpyFileDataset(ConfigRef("data.root"), repo=repo)

    assert ds.root == tmp_path
    assert [item.tolist() for item in ds] == [[4, 5]]
