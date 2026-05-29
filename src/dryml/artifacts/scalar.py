from __future__ import annotations

import os
from typing import Any, Iterable

import numpy as np

from dryml.core2.utils.general import pickle_load, pickle_save
from dryml.core2.utils.recurse import iter_leaves

from .base import Artifact


_SCALAR_FILENAME = "value.pkl"


def _scalar_path(location: str) -> str:
    return os.path.join(location, _SCALAR_FILENAME)


def _write_scalar(location: str, value: Any) -> None:
    os.makedirs(location, exist_ok=True)
    pickle_save(value, _scalar_path(location))


def _read_scalar(location: str) -> Any:
    return pickle_load(_scalar_path(location))


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class Scalar(Artifact):
    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    def compute(self, repo=None, *, store=None):
        location = self._location(repo, store=store, require_exists=True)
        _write_scalar(location, self.value)
        return self.value

    def read(self, repo=None, *, store=None):
        try:
            location = self._location(repo, store=store)
        except RuntimeError:
            return self.value
        path = _scalar_path(location)
        if os.path.exists(path):
            return _read_scalar(location)
        return self.value


class ScalarAgg(Artifact):
    def __init__(self, src):
        super().__init__()
        self.src = src

    def aggregate(self, values: Iterable[Any]):
        raise NotImplementedError

    def compute(self, repo=None, *, store=None):
        value = self.aggregate(iter(self.src))
        _write_scalar(self._location(repo, store=store, require_exists=True), value)
        return value

    def read(self, repo=None, *, store=None):
        return _read_scalar(self._location(repo, store=store, require_exists=True))


class ScalarAvg(ScalarAgg):
    def aggregate(self, values: Iterable[Any]) -> float:
        total = 0.0
        count = 0
        for item in values:
            for leaf in iter_leaves(item):
                arr = _as_numpy(leaf)
                if arr.size == 0:
                    continue
                total += float(np.sum(arr))
                count += int(arr.size)

        if count == 0:
            raise ValueError("Cannot average an empty scalar source.")
        return total / count


Scalar.__module__ = "dryml.artifacts"
ScalarAgg.__module__ = "dryml.artifacts"
ScalarAvg.__module__ = "dryml.artifacts"
