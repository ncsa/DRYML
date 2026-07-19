from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from dryml.artifacts import CachedDataset
from dryml.core2.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.data.tf.cache import TensorFlowCacheView


def test_tensorflow_cache_view_is_lazy_and_iterates_parquet(tmp_path):
    tf = pytest.importorskip("tensorflow")
    store = DirStore(tmp_path / "store")
    cached = CachedDataset(ArrayDataset(np.arange(12, dtype=np.float32).reshape(6, 2)))
    cached.compute(store=store, representation="numpy-sequence", shard_rows=2)

    view = cached.tensorflow_view(store=store, representation="parquet")

    assert isinstance(view, TensorFlowCacheView)
    assert view.support().status == "ok"
    values = list(view)
    assert all(isinstance(value, tf.Tensor) for value in values)
    assert [value.numpy().tolist() for value in values] == [
        value.tolist() for value in np.arange(12, dtype=np.float32).reshape(6, 2)
    ]


def test_tensorflow_cache_module_does_not_import_tensorflow():
    source = """
import sys
import dryml.data.tf.cache
assert 'tensorflow' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_tensorflow_cache_view_reports_missing_dependency(monkeypatch):
    from dryml.data.tf import cache

    monkeypatch.setattr(cache, "_framework_available", lambda _name: False)
    view = TensorFlowCacheView(object())

    support = view.support()
    assert support.status == "unsupported"
    assert support.issues[0].code == "optional_dependency_missing"
