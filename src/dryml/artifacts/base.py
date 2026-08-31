from __future__ import annotations

from dryml.core import Serializable


class Artifact(Serializable):
    """Base class for repo-backed computed payloads."""

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        pass

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        pass

    def compute(self):
        raise NotImplementedError

Artifact.__module__ = "dryml.artifacts"
