from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


InstancePolicy = Literal["reuse", "new"]
# "reuse": return cached instance if present
# "new": always construct a fresh instance (even if cached)


CachePolicy = Literal["none", "weak", "strong"]
# "weak": keep weak ref so GC can reclaim (default for incidental objects)
# "strong": keep strong ref for pinned objects / things intended to be saved
# "none": do not cache (rare, but useful for strict “fresh” eval runs)


LiveReusePolicy = Literal["matching", "greedy", "never"]
"""Policy controlling whether exact StateRef loads may reuse live objects."""


@dataclass(frozen=True, slots=True)
class RepoLoadOptions:
    instance: InstancePolicy = "reuse"
    restore_state: bool = True
    build_missing: bool = False
    reuse_weak: bool = True
    cache: CachePolicy = "weak"
    revision: Any = None


RepoGraphMissingPolicy = Literal["raise", "skip", "load"]
RepoGraphOrder = Literal["pre", "post"]


@dataclass(frozen=True, slots=True)
class RepoGraphOptions:
    load: RepoLoadOptions = field(default_factory=RepoLoadOptions)
    include_root: bool = True
    order: RepoGraphOrder = "post"
    missing: RepoGraphMissingPolicy = "raise"
    dedupe: bool = True
