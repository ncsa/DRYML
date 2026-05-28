import builtins
import sys
import types

import numpy as np
import pytest

from dryml.core2 import ConfigRef, Repo
from dryml.core2.cardinality import Cardinality
from dryml.core2.tensor_spec import TensorSpec
from dryml.data import TFDSAdapter, NpyFileDataset


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


class _Cardinality:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class _FakeTFDS:
    def __init__(self, items, cardinality=None):
        self.items = tuple(items)
        self._cardinality = len(self.items) if cardinality is None else cardinality

    def as_numpy_iterator(self):
        return iter(self.items)

    def cardinality(self):
        return _Cardinality(self._cardinality)


def _install_fake_tfds(monkeypatch, dataset):
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return dataset

    module = types.SimpleNamespace(load=load)
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", module)
    return calls


def test_tfds_adapter_numpy_mode_infers_spec_and_iterates(monkeypatch):
    items = [
        (np.zeros((28, 28, 1), dtype=np.uint8), np.int64(0)),
        (np.ones((28, 28, 1), dtype=np.uint8), np.int64(1)),
    ]
    calls = _install_fake_tfds(monkeypatch, _FakeTFDS(items))

    ds = TFDSAdapter("mnist", split="train[:2]", as_supervised=True, as_numpy=True)

    assert calls[0][0] == ("mnist",)
    assert calls[0][1]["split"] == "train[:2]"
    assert calls[0][1]["as_supervised"] is True
    assert ds.spec == (
        TensorSpec("uint8", shape=(28, 28, 1), backend="numpy"),
        TensorSpec("int64", shape=(), backend="numpy"),
    )
    assert ds.__len__() == Cardinality.finite(2)
    assert [int(y) for _, y in ds] == [0, 1]


def test_tfds_adapter_real_mnist_numpy_mode():
    pytest.importorskip("tensorflow_datasets")

    ds = TFDSAdapter("mnist", split="train[:2]", as_supervised=True, as_numpy=True)

    assert ds.spec[0] == TensorSpec("uint8", shape=(28, 28, 1), backend="numpy")
    assert ds.spec[1] == TensorSpec("int64", shape=(), backend="numpy")
    assert ds.__len__() == Cardinality.finite(2)
    assert len(list(ds)) == 2


def test_tfds_adapter_numpy_mode_does_not_import_tensorflow(monkeypatch):
    items = [(np.zeros((2,), dtype=np.float32), np.int64(0))]
    _install_fake_tfds(monkeypatch, _FakeTFDS(items))
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "dryml.tf":
            raise AssertionError("TFDSAdapter(as_numpy=True) should not import dryml.tf.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    ds = TFDSAdapter("fake", as_supervised=True, as_numpy=True)

    assert ds.spec[0] == TensorSpec("float32", shape=(2,), backend="numpy")


def test_tfds_adapter_maps_unknown_and_infinite_cardinality(monkeypatch):
    items = [(np.zeros((2,), dtype=np.float32), np.int64(0))]

    _install_fake_tfds(monkeypatch, _FakeTFDS(items, cardinality=-2))
    assert TFDSAdapter("fake", as_supervised=True, as_numpy=True).__len__() == Cardinality.UNKNOWN

    _install_fake_tfds(monkeypatch, _FakeTFDS(items, cardinality=-1))
    assert TFDSAdapter("fake", as_supervised=True, as_numpy=True).__len__() == Cardinality.INFINITE
