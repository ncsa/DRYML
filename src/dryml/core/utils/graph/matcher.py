from __future__ import annotations

from typing import Any, Hashable

from ..types import is_namedtuple
from .transformer import GraphCtx


class GraphMatchError(Exception):
    pass


class GraphMatcher:
    """
    Generic two-graph matcher.

    This is the sibling of GraphVisitor / GraphTransformer for cases where
    recursion depends on *pairs* of nodes instead of a single node.
    """

    def match(self, selector: Any, target: Any, ctx: GraphCtx | None = None) -> bool:
        if ctx is None:
            ctx = GraphCtx()

        active_pairs = ctx.state.setdefault("_active_pairs", set())

        if self.is_atomic_pair(selector, target, ctx):
            return self.match_atomic(selector, target, ctx)

        pair_key = self.memo_key(selector, target, ctx)
        if pair_key is not None and pair_key in ctx.memo:
            return ctx.memo[pair_key]

        track_cycle = self.should_track_cycle(selector, target, ctx)
        pair_id = (id(selector), id(target))
        if track_cycle:
            if pair_id in active_pairs:
                raise self.cycle_error(selector, target, ctx)
            active_pairs.add(pair_id)

        try:
            out = self.dispatch(selector, target, ctx)
            if pair_key is not None:
                ctx.memo[pair_key] = out
            return out
        finally:
            if track_cycle:
                active_pairs.remove(pair_id)

    # ------------------------------------------------------------------
    # Policy hooks
    # ------------------------------------------------------------------

    def is_atomic_pair(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        return False

    def match_atomic(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        return selector == target

    def memo_key(
        self, selector: Any, target: Any, ctx: GraphCtx
    ) -> Hashable | None:
        return None

    def should_track_cycle(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        return (
            isinstance(selector, (dict, list, tuple, set))
            or is_namedtuple(selector)
            or isinstance(target, (dict, list, tuple, set))
            or is_namedtuple(target)
        )

    def cycle_error(self, selector: Any, target: Any, ctx: GraphCtx) -> Exception:
        return GraphMatchError(
            f"Cycle detected while matching {type(selector).__name__} "
            f"against {type(target).__name__} at {ctx.path_str()}"
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        if isinstance(selector, dict) and isinstance(target, dict):
            return self.match_dict(selector, target, ctx)

        if is_namedtuple(selector) and is_namedtuple(target):
            return self.match_namedtuple(selector, target, ctx)

        if isinstance(selector, tuple) and isinstance(target, tuple):
            return self.match_tuple(selector, target, ctx)

        if isinstance(selector, list) and isinstance(target, list):
            return self.match_list(selector, target, ctx)

        if isinstance(selector, set) and isinstance(target, set):
            return self.match_set(selector, target, ctx)

        return self.match_other(selector, target, ctx)

    # ------------------------------------------------------------------
    # Generic container matchers
    # ------------------------------------------------------------------

    def match_dict(
        self, selector: dict[Any, Any], target: dict[Any, Any], ctx: GraphCtx
    ) -> bool:
        if set(selector.keys()) != set(target.keys()):
            return False

        for k in selector.keys():
            if not self.match(selector[k], target[k], ctx.child(k if isinstance(k, (str, int)) else str(k))):
                return False
        return True

    def match_namedtuple(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        if type(selector) is not type(target):
            return False
        return self.match_tuple(selector, target, ctx)

    def match_tuple(
        self, selector: tuple[Any, ...], target: tuple[Any, ...], ctx: GraphCtx
    ) -> bool:
        if len(selector) != len(target):
            return False

        for i, (sel_v, tgt_v) in enumerate(zip(selector, target)):
            if not self.match(sel_v, tgt_v, ctx.child(i)):
                return False
        return True

    def match_list(
        self, selector: list[Any], target: list[Any], ctx: GraphCtx
    ) -> bool:
        if len(selector) != len(target):
            return False

        for i, (sel_v, tgt_v) in enumerate(zip(selector, target)):
            if not self.match(sel_v, tgt_v, ctx.child(i)):
                return False
        return True

    def match_set(
        self, selector: set[Any], target: set[Any], ctx: GraphCtx
    ) -> bool:
        # Greedy matching. Good enough for the generic utility.
        if len(selector) != len(target):
            return False

        unmatched = list(target)
        for i, sel_v in enumerate(selector):
            found = None
            for j, tgt_v in enumerate(unmatched):
                if self.match(sel_v, tgt_v, ctx.child(i)):
                    found = j
                    break
            if found is None:
                return False
            unmatched.pop(found)

        return True

    def match_other(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        return type(selector) is type(target) and selector == target
