from __future__ import annotations

from pathlib import Path

import numpy as np

from dryml.core2.utils.general import pickle_save

from .base import Artifact


_CACHE_META_FILENAME = "cache.pkl"


class CachedDataset(Artifact):
    def __init__(
            self,
            src,
            *,
            pattern: str = "{index:08d}.npy",
            allow_pickle: bool = False):
        super().__init__()
        self.src = src
        self.pattern = pattern
        self.allow_pickle = allow_pickle

    def _item_path(self, root: Path, index: int) -> Path:
        rel = Path(self.pattern.format(index=index))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("CachedDataset pattern must stay inside the artifact location.")
        if rel.suffix != ".npy":
            rel = rel.with_suffix(".npy")
        return root / rel

    def compute(self, repo=None, *, store=None) -> str:
        location = self._location(repo, store=store, require_exists=True)
        root = Path(location)
        root.mkdir(parents=True, exist_ok=True)

        for path in root.rglob("*.npy"):
            path.unlink()

        count = 0
        for index, item in enumerate(self.src):
            path = self._item_path(root, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, item, allow_pickle=self.allow_pickle)
            count += 1

        pickle_save({"count": count}, root / _CACHE_META_FILENAME)
        return location

CachedDataset.__module__ = "dryml.artifacts"
