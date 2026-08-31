from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkspaceHandle:
    """Opaque handle for a realized Object's local workspace path.

    Attributes:
        root: Base path for the owning Repo/session on this machine.
        rel: Token-neutral per-realization subpath beneath ``root``.
    """

    root: str        # base path for this repo/session on this machine
    rel: str         # per-object subpath (e.g. cdefhash or instance id)

    def path(self) -> str:
        """Return the filesystem path represented by this handle.

        Returns:
            The path formed by joining the handle root and relative path.
        """

        return os.path.join(self.root, self.rel)


class WorkspaceManager:
    """Allocate token-neutral, realization-scoped Object workspaces.

    Private CDef node keys are retained only in process-local maps. Generated
    labels, rather than those private keys or realization tokens, form paths.
    """

    def __init__(self, base_dir: str):
        """Create a manager rooted at ``base_dir``.

        Args:
            base_dir: Existing or creatable directory that contains all managed
                local workspaces.

        Side Effects:
            Creates ``base_dir`` when it does not already exist.
        """

        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def alloc(
            self,
            cdef_hash: str,
            *,
            scope: Any | None = None,
            node_key: object | None = None) -> WorkspaceHandle:
        """Allocate an isolated workspace for one realized graph node.

        Args:
            cdef_hash: Structural definition digest used only as a readable
                directory prefix.
            scope: Private realization scope supplying token-neutral labels.
            node_key: Private CDef-node token. It is retained only in an
                in-memory map and never included in the returned path.

        Returns:
            A handle rooted beneath this manager's base directory.
        """

        if scope is None or node_key is None:
            return WorkspaceHandle(self.base_dir, cdef_hash)
        rel = os.path.join(
            cdef_hash,
            scope.workspace_label,
            scope.workspace_node_label(node_key),
        )
        return WorkspaceHandle(self.base_dir, rel)
