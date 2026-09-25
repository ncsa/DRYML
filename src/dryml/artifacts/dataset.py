from __future__ import annotations

from .base import Artifact

class CachedDataset(Artifact):
    """Abstract Stage 4 placeholder for a future cached Dataset Artifact.

    The retired mutable Store-root cache protocol is intentionally unavailable.
    This class supplies no constructor, cache state, computation, or readiness
    implementation, so it remains abstract until its later design unit lands.
    """

    pass

CachedDataset.__module__ = "dryml.artifacts"
