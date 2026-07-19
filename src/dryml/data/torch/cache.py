"""Dependency-lazy PyTorch views over managed cache representations."""

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
class TorchCacheView:
    """Lazy iterable yielding PyTorch tensors from one active cache."""

    dataset: Any
    repo: Any = None
    store: Any = None
    representation: Any = "numpy-sequence"

    def support(self) -> CacheViewSupport:
        """Report PyTorch availability without importing it."""

        return framework_support(
            "torch",
            "PyTorch",
            "torch",
            available=_framework_available("torch"),
        )

    def __iter__(self):
        """Resolve the active representation and import PyTorch on demand."""

        support = self.support()
        if support.status != "ok":
            raise CacheViewUnavailableError(support)
        from dryml.runtime import import_configured_framework

        torch = import_configured_framework("torch")
        _located, _record, root, spec = self.dataset.representation_record(
            self.representation, repo=self.repo, store=self.store
        )
        for value in iter_cache_representation(root, spec.kind):
            yield map_leaves(value, torch.as_tensor)


def _framework_available(name: str) -> bool:
    """Compatibility seam for dependency-absence tests."""

    return importlib.util.find_spec(name) is not None


__all__ = [
    "CacheViewIssue",
    "CacheViewSupport",
    "CacheViewUnavailableError",
    "TorchCacheView",
]
