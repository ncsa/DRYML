"""Generic graph reduction and reachability helpers, without domain policy."""

from __future__ import annotations

from typing import Any, Hashable

from ..types import is_namedtuple
from .transformer import GraphCtx


def fold_closure_masks(children, masks):
    """Union distinct children's reachability masks and detect branch overlap.

    Args:
        children (Iterable[Hashable]): Distinct direct targets in typed edge order.
        masks (Mapping[Hashable, int]): Already folded, inclusive child closures.
    Returns:
        tuple[int, bool]: Union mask and whether distinct closures intersect.
    Raises:
        KeyError: If a dependency's mask has not been folded.
    Side Effects:
        None. The predicate depends only on incidence, never on local values,
        hashes, traversal demand or cache state. Domain role labels remain owned
        by the caller's typed edge records.
    """
    closure, overlap = 0, False
    for child in children:
        mask = masks[child]
        overlap |= bool(closure & mask)
        closure |= mask
    return closure, overlap


class GraphHashError(Exception):
    pass


class GraphHasher:
    """
    Generic one-input graph hasher / reducer.

    Subclasses override:
      - is_atomic
      - hash_atomic
      - dispatch
      - memo_key / should_track_cycle if needed
    """

    def hash(self, obj: Any, ctx: GraphCtx | None = None) -> str:
        if ctx is None:
            ctx = GraphCtx()

        active_ids = ctx.state.setdefault("_active_ids", set())

        if self.is_atomic(obj, ctx):
            return self.hash_atomic(obj, ctx)

        memo_key = self.memo_key(obj, ctx)
        if memo_key is not None and memo_key in ctx.memo:
            return ctx.memo[memo_key]

        track_cycle = self.should_track_cycle(obj, ctx)
        oid = id(obj)
        if track_cycle:
            if oid in active_ids:
                raise self.cycle_error(obj, ctx)
            active_ids.add(oid)

        try:
            out = self.dispatch(obj, ctx)
            if memo_key is not None:
                ctx.memo[memo_key] = out
            return out
        finally:
            if track_cycle:
                active_ids.remove(oid)

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return False

    def hash_atomic(self, obj: Any, ctx: GraphCtx) -> str:
        raise TypeError(
            f"{type(self).__name__} does not know how to hash atomic {type(obj).__name__}"
        )

    def memo_key(self, obj: Any, ctx: GraphCtx) -> Hashable | None:
        return id(obj)

    def should_track_cycle(self, obj: Any, ctx: GraphCtx) -> bool:
        return (
            isinstance(obj, (dict, list, tuple, set))
            or is_namedtuple(obj)
        )

    def cycle_error(self, obj: Any, ctx: GraphCtx) -> Exception:
        return GraphHashError(
            f"Cycle detected while hashing {type(obj).__name__} at {ctx.path_str()}"
        )

    def dispatch(self, obj: Any, ctx: GraphCtx) -> str:
        raise TypeError(
            f"{type(self).__name__} cannot hash {type(obj).__name__} at {ctx.path_str()}"
        )
