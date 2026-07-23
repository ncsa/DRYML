from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from dryml.artifacts import CachedDataset
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.data.torch.cache import TorchCacheView


def test_torch_cache_view_is_lazy_and_iterates_numpy(tmp_path):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "Tensor"):
        sys.modules.pop("torch", None)
        pytest.skip("PyTorch is not installed")
    store = DirStore(tmp_path / "store")
    cached = CachedDataset(ArrayDataset(np.arange(12, dtype=np.float32).reshape(6, 2)))
    cached.compute(store=store, representation="numpy-sequence", shard_rows=2)

    view = cached.torch_view(store=store)

    assert isinstance(view, TorchCacheView)
    assert view.support().status == "ok"
    values = list(view)
    assert all(isinstance(value, torch.Tensor) for value in values)
    assert [value.cpu().numpy().tolist() for value in values] == [
        value.tolist() for value in np.arange(12, dtype=np.float32).reshape(6, 2)
    ]


def test_torch_cache_module_does_not_import_torch():
    source = """
import sys
import dryml.data.torch.cache
assert 'torch' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_torch_cache_view_reports_missing_dependency(monkeypatch):
    from dryml.data.torch import cache

    monkeypatch.setattr(cache, "_framework_available", lambda _name: False)
    view = TorchCacheView(object())

    support = view.support()
    assert support.status == "unsupported"
    assert support.issues[0].code == "optional_dependency_missing"
