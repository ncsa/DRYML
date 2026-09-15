"""Persistent DefLink structure and lazy shared Ref/Mat vocabulary access."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DefLink:
    """Parent-slot edge wrapper for finalized targets or transient assertions.

    Args:
        kind: The graph edge kind represented by this link.
        target: A finalized link target. Public construction retains the legacy
            finalized representation for persisted data.

    Raises:
        TypeError: If ``kind`` is not an ``EdgeKind`` or a finalized target is
            unsupported.

    Transient assertions are created only through the shared ``Ref(value)`` and
    ``Mat(value)`` vocabulary. They retain their live source until signature
    normalization finalizes them and cannot be hashed, encoded, or pickled.
    """

    kind: Any
    target: Any
    _finalized: bool = field(default=True, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        from .canonical import freeze_link_target
        from .cdef_graph import EdgeKind

        if not isinstance(self.kind, EdgeKind):
            raise TypeError(f"DefLink kind must be an EdgeKind, got {type(self.kind).__name__}.")
        if self._finalized:
            object.__setattr__(self, "target", freeze_link_target(self.target))

    @property
    def is_finalized(self) -> bool:
        """Whether this link is safe to admit into persistent identity data.

        Returns:
            ``True`` for canonical persisted links and ``False`` for a live
            caller assertion awaiting signature normalization.
        """

        return self._finalized

    @classmethod
    def finalized(cls, kind: Any, target: Any) -> "DefLink":
        """Create a validated canonical link for decoding and finalized output.

        Args:
            kind: The canonical ``EdgeKind``.
            target: A supported persisted link target.

        Returns:
            A finalized immutable link with the historical encoded shape.
        """

        return cls(kind, target)

    @classmethod
    def assertion(cls, kind: Any, target: Any) -> "DefLink":
        """Create an unresolved live assertion owned by signatures.

        Args:
            kind: The asserted delivery edge kind.
            target: The original caller value, retained without freezing.

        Returns:
            A transient link which must be finalized by ``dryml.core.signatures``.

        Raises:
            TypeError: If ``kind`` is not an ``EdgeKind``.
        """

        from .cdef_graph import EdgeKind

        if not isinstance(kind, EdgeKind):
            raise TypeError(f"DefLink kind must be an EdgeKind, got {type(kind).__name__}.")
        result = object.__new__(cls)
        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "target", target)
        object.__setattr__(result, "_finalized", False)
        return result

    def __reduce__(self):
        """Serialize only finalized canonical links.

        Returns:
            The historical constructor reduction for a finalized link.

        Raises:
            TypeError: If a live assertion reaches a pickle boundary.
        """

        if not self._finalized:
            raise TypeError("Unresolved DefLink assertions cannot be pickled.")
        return type(self), (self.kind, self.target)


def __getattr__(name: str) -> Any:
    """Lazily expose the shared Ref/Mat vocabulary without an import cycle.

    Args:
        name: Requested legacy module attribute.

    Returns:
        The shared callable/subscriptable vocabulary singleton.

    Raises:
        AttributeError: If ``name`` is not a supported vocabulary member.
    """

    if name not in {"Ref", "Mat"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .signatures import Mat, Ref

    return {"Ref": Ref, "Mat": Mat}[name]
