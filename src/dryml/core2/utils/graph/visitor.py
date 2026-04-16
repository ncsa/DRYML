from __future__ import annotations

from typing import Any, Hashable

from ..types import is_namedtuple
from .transformer import GraphCtx, GraphTransformError


class GraphVisitor:
    """
    Generic stateful graph visitor.

    This base class only knows how to recurse through:
      - dict
      - list
      - tuple
      - set
      - namedtuple instances

    Everything else is delegated to `visit_other`.

    Subclasses typically override:
      - is_atomic / visit_atomic
      - memo_key
      - should_track_cycle
      - visit_other
      - optionally specific container handlers
    """

    def visit(self, obj: Any, ctx: GraphCtx | None = None) -> None:
        if ctx is None:
            ctx = GraphCtx()

        if self.is_atomic(obj, ctx):
            self.visit_atomic(obj, ctx)
            return

        track_cycle = self.should_track_cycle(obj, ctx)
        oid = id(obj)
        if track_cycle and oid in ctx.active_ids:
            raise self.cycle_error(obj, ctx)

        memo_key = self.memo_key(obj, ctx)
        if memo_key is not None:
            if memo_key in ctx.memo:
                return
            ctx.memo[memo_key] = True

        if track_cycle:
            ctx.active_ids.add(oid)

        try:
            self.dispatch(obj, ctx)
        finally:
            if track_cycle:
                ctx.active_ids.remove(oid)

    # ------------------------------------------------------------------
    # Policy hooks
    # ------------------------------------------------------------------

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return False

    def visit_atomic(self, obj: Any, ctx: GraphCtx) -> None:
        return None

    def memo_key(self, obj: Any, ctx: GraphCtx) -> Hashable | None:
        return None

    def should_track_cycle(self, obj: Any, ctx: GraphCtx) -> bool:
        return (
            isinstance(obj, (dict, list, tuple, set))
            or is_namedtuple(obj)
        )

    def cycle_error(self, obj: Any, ctx: GraphCtx) -> Exception:
        return GraphTransformError(
            f"Cycle detected for {type(obj).__name__} at {ctx.path_str()}"
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, obj: Any, ctx: GraphCtx) -> None:
        if isinstance(obj, dict):
            self.visit_dict(obj, ctx)
            return

        if is_namedtuple(obj):
            self.visit_namedtuple(obj, ctx)
            return

        if isinstance(obj, tuple):
            self.visit_tuple(obj, ctx)
            return

        if isinstance(obj, list):
            self.visit_list(obj, ctx)
            return

        if isinstance(obj, set):
            self.visit_set(obj, ctx)
            return

        self.visit_other(obj, ctx)

    # ------------------------------------------------------------------
    # Generic container visitors
    # ------------------------------------------------------------------

    def visit_dict_keys(self, obj: dict[Any, Any], ctx: GraphCtx) -> bool:
        """
        Whether dict keys should be recursively visited.

        Default is False to match the plain tree helpers, which recurse over
        dict values only.
        """
        return False

    def visit_dict(self, obj: dict[Any, Any], ctx: GraphCtx) -> None:
        do_keys = self.visit_dict_keys(obj, ctx)

        for k, v in obj.items():
            if do_keys:
                self.visit(k, ctx.child("<key>"))

            value_path = k if isinstance(k, (str, int)) else str(k)
            self.visit(v, ctx.child(value_path))

    def visit_namedtuple(self, obj: Any, ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_tuple(self, obj: tuple[Any, ...], ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_list(self, obj: list[Any], ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_set(self, obj: set[Any], ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(f"<set:{i}>"))

    def visit_other(self, obj: Any, ctx: GraphCtx) -> None:
        raise TypeError(
            f"{type(self).__name__} cannot visit {type(obj).__name__} "
            f"at {ctx.path_str()}"
        )
