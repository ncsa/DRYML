from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Hashable, TypeAlias

from ..types import is_namedtuple


PathPart: TypeAlias = str | int


@dataclass(slots=True, frozen=True)
class GraphCtx:
    """
    Generic traversal context shared across recursive graph operations.

    Notes
    -----
    - `path` is immutable and extended for child traversals.
    - `memo`, `active_ids`, and `state` are shared across children.
    - `state` is intentionally untyped here so callers can stash whatever
      operation-specific data they need without baking policy into utils.
    """
    path: tuple[PathPart, ...] = ()
    memo: dict[Hashable, Any] = field(default_factory=dict)
    active_ids: set[int] = field(default_factory=set)
    state: dict[str, Any] = field(default_factory=dict)

    def child(self, part: PathPart) -> "GraphCtx":
        return replace(self, path=self.path + (part,))

    def with_state(self, **kwargs: Any) -> "GraphCtx":
        new_state = dict(self.state)
        new_state.update(kwargs)
        return replace(self, state=new_state)

    def path_str(self) -> str:
        if not self.path:
            return "<root>"
        return "/".join(map(str, self.path))


class GraphTransformError(Exception):
    pass


class GraphTransformer:
    """
    Generic stateful graph-to-graph transformer.

    This base class only knows how to recurse through:
      - dict
      - list
      - tuple
      - set
      - namedtuple instances

    Everything else is delegated to `transform_other`.

    Subclasses typically override:
      - is_atomic / transform_atomic
      - memo_key
      - should_track_cycle
      - transform_other
      - optionally specific container handlers
    """

    def transform(self, obj: Any, ctx: GraphCtx | None = None) -> Any:
        if ctx is None:
            ctx = GraphCtx()

        if self.is_atomic(obj, ctx):
            return self.transform_atomic(obj, ctx)

        memo_key = self.memo_key(obj, ctx)
        if memo_key is not None and memo_key in ctx.memo:
            return ctx.memo[memo_key]

        track_cycle = self.should_track_cycle(obj, ctx)
        oid = id(obj)
        if track_cycle:
            if oid in ctx.active_ids:
                raise self.cycle_error(obj, ctx)
            ctx.active_ids.add(oid)

        try:
            out = self.dispatch(obj, ctx)
            if memo_key is not None:
                ctx.memo[memo_key] = out
            return out
        finally:
            if track_cycle:
                ctx.active_ids.remove(oid)

    # ------------------------------------------------------------------
    # Policy hooks
    # ------------------------------------------------------------------

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return False

    def transform_atomic(self, obj: Any, ctx: GraphCtx) -> Any:
        return obj

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

    def dispatch(self, obj: Any, ctx: GraphCtx) -> Any:
        if isinstance(obj, dict):
            return self.transform_dict(obj, ctx)

        if is_namedtuple(obj):
            return self.transform_namedtuple(obj, ctx)

        if isinstance(obj, tuple):
            return self.transform_tuple(obj, ctx)

        if isinstance(obj, list):
            return self.transform_list(obj, ctx)

        if isinstance(obj, set):
            return self.transform_set(obj, ctx)

        return self.transform_other(obj, ctx)

    # ------------------------------------------------------------------
    # Generic container transforms
    # ------------------------------------------------------------------

    def transform_dict_keys(self, obj: dict[Any, Any], ctx: GraphCtx) -> bool:
        """
        Whether dict keys should be recursively transformed.

        Default is False to match the plain tree helpers, which recurse over
        dict values only.
        """
        return False

    def transform_dict_key(self, key: Any, ctx: GraphCtx) -> Any:
        return self.transform(key, ctx.child("<key>"))

    def transform_dict_value(self, key: Any, value: Any, ctx: GraphCtx) -> Any:
        value_path = key if isinstance(key, (str, int)) else str(key)
        return self.transform(value, ctx.child(value_path))

    def transform_dict(self, obj: dict[Any, Any], ctx: GraphCtx) -> Any:
        out: dict[Any, Any] = {}
        do_keys = self.transform_dict_keys(obj, ctx)

        for k, v in obj.items():
            rk = self.transform_dict_key(k, ctx) if do_keys else k
            rv = self.transform_dict_value(k, v, ctx)
            out[rk] = rv

        return out

    def transform_namedtuple(self, obj: Any, ctx: GraphCtx) -> Any:
        return type(obj)(*(self.transform(v, ctx.child(i)) for i, v in enumerate(obj)))

    def transform_tuple(self, obj: tuple[Any, ...], ctx: GraphCtx) -> Any:
        return tuple(self.transform(v, ctx.child(i)) for i, v in enumerate(obj))

    def transform_list(self, obj: list[Any], ctx: GraphCtx) -> Any:
        return [self.transform(v, ctx.child(i)) for i, v in enumerate(obj)]

    def transform_set(self, obj: set[Any], ctx: GraphCtx) -> Any:
        return {
            self.transform(v, ctx.child(f"<set:{i}>"))
            for i, v in enumerate(obj)
        }

    def transform_other(self, obj: Any, ctx: GraphCtx) -> Any:
        raise TypeError(
            f"{type(self).__name__} cannot transform {type(obj).__name__} "
            f"at {ctx.path_str()}"
        )
