from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CachePolicy = Literal["none", "weak", "strong"]
# "weak": keep weak ref so GC can reclaim (default for incidental objects)
# "strong": keep strong ref for pinned objects / things intended to be saved
# "none": do not cache (rare, but useful for strict “fresh” eval runs)


LiveReusePolicy = Literal["matching", "greedy", "never"]
"""Policy controlling whether exact StateRef loads may reuse live objects."""


RepoGraphMissingPolicy = Literal["raise", "skip", "load"]
RepoGraphOrder = Literal["pre", "post"]


@dataclass(frozen=True, slots=True)
class RepoGraphOptions:
    include_root: bool = True
    order: RepoGraphOrder = "post"
    missing: RepoGraphMissingPolicy = "raise"
    dedupe: bool = True
