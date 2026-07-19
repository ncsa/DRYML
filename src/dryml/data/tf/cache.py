"""Dependency-lazy TensorFlow views over managed cache representations."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from dryml.core2.utils.recurse import map_leaves
from dryml.data.cache import (
    CacheViewIssue,
    CacheViewSupport,
    CacheViewUnavailableError,
    framework_support,
    iter_cache_representation,
)


@dataclass(frozen=True, slots=True)
class TensorFlowCacheView:
    """Lazy iterable yielding TensorFlow tensors from one active cache."""

    dataset: Any
    repo: Any = None
    store: Any = None
    representation: Any = "numpy-sequence"

    def support(self) -> CacheViewSupport:
        """Report TensorFlow availability without importing it."""

        return framework_support(
            "tensorflow",
            "TensorFlow",
            "tf",
            available=_framework_available("tensorflow"),
        )

    def __iter__(self):
        """Resolve the active representation and import TensorFlow on demand."""

        support = self.support()
        if support.status != "ok":
            raise CacheViewUnavailableError(support)
        from dryml.runtime import import_configured_framework

        tf = import_configured_framework("tensorflow")
        _located, _record, root, spec = self.dataset.representation_record(
            self.representation, repo=self.repo, store=self.store
        )
        for value in iter_cache_representation(root, spec.kind):
            yield map_leaves(value, tf.convert_to_tensor)


def _framework_available(name: str) -> bool:
    """Compatibility seam for dependency-absence tests."""

    return importlib.util.find_spec(name) is not None


__all__ = [
    "CacheViewIssue",
    "CacheViewSupport",
    "CacheViewUnavailableError",
    "TensorFlowCacheView",
]
