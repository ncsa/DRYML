from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DefLink:
    """Parent-slot edge wrapper for definition and exact-reference targets."""

    kind: Any
    target: Any

    def __post_init__(self) -> None:
        from .canonical import freeze_link_target
        from .cdef_graph import EdgeKind

        if not isinstance(self.kind, EdgeKind):
            raise TypeError(f"DefLink kind must be an EdgeKind, got {type(self.kind).__name__}.")

        object.__setattr__(self, "target", freeze_link_target(self.target))


def Ref(target: Any) -> DefLink:
    """Return a non-materializing reference edge wrapper for ``target``."""

    from .cdef_graph import EdgeKind

    return DefLink(EdgeKind.REF, target)


def Mat(target: Any) -> DefLink:
    """Return an explicit materializing edge wrapper for ``target``."""

    from .cdef_graph import EdgeKind

    return DefLink(EdgeKind.MATERIALIZE, target)
