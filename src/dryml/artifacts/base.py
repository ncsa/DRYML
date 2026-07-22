from __future__ import annotations

from dryml.core2 import Serializable


class Artifact(Serializable):
    """Logical Object base for managed, record-backed computed results."""

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        pass

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None = None):
        pass

    def compute(self):
        raise NotImplementedError

    def exists(self, repo=None, *, store=None) -> bool:
        """Return whether this Artifact has a completed active managed result."""

        operation = self.compute
        status = getattr(operation, "status", None)
        if status is None:
            return False
        try:
            return status(repo=repo, store=store).active_realization_id is not None
        except RuntimeError:
            return False


Artifact.__module__ = "dryml.artifacts"
