from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class WorkspaceHandle:
    root: str        # base path for this repo/session on this machine
    rel: str         # per-object subpath (e.g. cdefhash or instance id)

    def path(self) -> str:
        return os.path.join(self.root, self.rel)


class WorkspaceManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def alloc(self, cdef_hash: str) -> WorkspaceHandle:
        # instance_id lets you support "fresh instance" even with same cdef
        rel = cdef_hash
        return WorkspaceHandle(self.base_dir, rel)
