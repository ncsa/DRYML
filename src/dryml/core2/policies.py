from typing import Literal


InstancePolicy = Literal["reuse", "new"]
# "reuse": return cached instance if present
# "new": always construct a fresh instance (even if cached)


CachePolicy = Literal["none", "weak", "strong"]
# "weak": keep weak ref so GC can reclaim (default for incidental objects)
# "strong": keep strong ref for pinned objects / things intended to be saved
# "none": do not cache (rare, but useful for strict “fresh” eval runs)
