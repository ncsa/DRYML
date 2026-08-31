from __future__ import annotations

import os

from dryml.core import Serializable


class Artifact(Serializable):
    """Base class for repo-backed computed payloads."""

    def _location(self, repo=None, *, store=None, require_exists: bool = False) -> str:
        """Resolve this Artifact's persisted location through explicit authority.

        Args:
            repo: Optional active Repo.
            store: Optional Store override.
            require_exists: Whether the selected location must already exist.

        Returns:
            The selected object directory.

        Raises:
            RuntimeError: If no explicit Repo or retained Store affinity exists.
        """

        from dryml.core.repo import get_default_repo

        if repo is None:
            repo = get_default_repo()
        if repo is None:
            if store is not None:
                raise RuntimeError("Artifact location with a Store override requires an explicit Repo.")
            return self.location
        return repo.location(self, store=store, require_exists=require_exists)

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        pass

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        pass

    def compute(self):
        raise NotImplementedError

    def exists(self) -> bool:
        try:
            return os.path.exists(self._location())
        except RuntimeError:
            return False


Artifact.__module__ = "dryml.artifacts"
