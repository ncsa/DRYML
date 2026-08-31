from __future__ import annotations

from .base import Artifact

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

    def compute(self, repo=None, *, store=None) -> str:
        """Reject the retired mutable object-directory cache protocol.

        Args:
            repo: Retired Store-root cache selector.
            store: Retired Store-root cache selector.

        Raises:
            RuntimeError: Always, because current Stores expose immutable local
                states rather than mutable object directories.
        """

        raise RuntimeError(
            "CachedDataset mutable Store-root caches are retired; use an explicit external cache."
        )

CachedDataset.__module__ = "dryml.artifacts"
