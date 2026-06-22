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


def _read_scalar_if_present(location: str):
    path = _scalar_path(location)
    if not os.path.exists(path):
        return None, False
    return _read_scalar(location), True


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalize_path(path):
    if isinstance(path, (tuple, list)):
        return tuple(path)
    return (path,)


def _select_path(value: Any, path) -> Any:
    result = value
    for idx in _normalize_path(path):
        result = result[idx]
    return result


class Scalar(Artifact):
    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        _write_scalar(dest_dir, self.value)

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None = None):
        value, exists = _read_scalar_if_present(src_dir)
        if exists:
            self.value = value

    def compute(self, repo=None, *, store=None):
        return self.value


class ScalarAgg(Artifact):
    def __init__(self, src):
        super().__init__()
        self.src = src

    def aggregate(self, values: Iterable[Any]):
        raise NotImplementedError

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        if hasattr(self, "value"):
            _write_scalar(dest_dir, self.value)

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None = None):
        value, exists = _read_scalar_if_present(src_dir)
        if exists:
            self.value = value

    def compute(self, repo=None, *, store=None):
        value = self.aggregate(iter(self.src))
        self.value = value
        return value


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


class Accuracy(ScalarAgg):
    def __init__(self, src, *, path_x=0, path_y=1):
        super().__init__(src)
        self.path_x = path_x
        self.path_y = path_y

    def aggregate(self, values: Iterable[Any]) -> float:
        num_correct = 0
        num_total = 0
        for item in values:
            x = _as_numpy(_select_path(item, self.path_x))
            y = _as_numpy(_select_path(item, self.path_y))
            if x.shape != y.shape:
                raise ValueError(
                    f"Accuracy requires matching shapes, got {x.shape} and {y.shape}."
                )

            matches = np.asarray(x == y)
            num_correct += int(np.sum(matches))
            num_total += int(matches.size)

        if num_total == 0:
            raise ValueError("Cannot compute Accuracy on an empty source.")
        return num_correct / num_total


Scalar.__module__ = "dryml.artifacts"
ScalarAgg.__module__ = "dryml.artifacts"
ScalarAvg.__module__ = "dryml.artifacts"
Accuracy.__module__ = "dryml.artifacts"
