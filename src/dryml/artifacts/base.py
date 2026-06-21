from __future__ import annotations

import os

from dryml.core2 import Serializable


class Artifact(Serializable):
    """Base class for repo-backed computed payloads."""

    def _location(self, repo=None, *, store=None, require_exists: bool = False) -> str:
        from dryml.core2.repo import get_default_repo

        if repo is None:
            repo = get_default_repo()
        return repo.location(self, store=store, require_exists=require_exists)

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        pass

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None = None):
        pass

    def compute(self):
        raise NotImplementedError

    def exists(self) -> bool:
        try:
            return os.path.exists(self._location())
        except RuntimeError:
            return False


Artifact.__module__ = "dryml.artifacts"
