from contextvars import ContextVar
from functools import wraps
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from ..errors import CycleError
from .types import _ATOMIC_TYPES, is_namedtuple


def cycle_detect(arg_pos=0, kwarg_name=None, should_track=None):
    if arg_pos is not None and kwarg_name is not None:
        raise ValueError("Specify only one of arg_pos or kwarg_name")
    if arg_pos is None and kwarg_name is None:
        raise ValueError("Specify one of arg_pos or kwarg_name")

    if should_track is None:
        def should_track(val):
            return not isinstance(val, _ATOMIC_TYPES + (type,))

    def decorator(f):
        path_var = ContextVar(f"cycle_path_{id(f)}", default=None)

        def val_getter(args, kwargs):
            return args[arg_pos] if arg_pos is not None else kwargs[kwarg_name]

        @wraps(f)
        def wrapper(*args, **kwargs):
            val = val_getter(args, kwargs)

            if not should_track(val):
                return f(*args, **kwargs)

            path = path_var.get()
            if path is None:
                path = set()

            oid = id(val)
            if oid in path:
                raise CycleError(
                    msg=(
                        f"Val/type that tripped: {type(val)}/{val} "
                        f"oid: {oid} path_oids: {path}"
                    )
                )

            new_path = set(path)
            new_path.add(oid)
            token = path_var.set(new_path)
            try:
                return f(*args, **kwargs)
            finally:
                path_var.reset(token)

        return wrapper

    return decorator


T = TypeVar("T")


def map_leaves(
        x: Any,
        leaf_fn: Callable[[Any], T],
        pred: Callable[[Any], bool]|None = None) -> Any:
    """
    Recursively map `leaf_fn` over the leaves of a plain Python tree.
    If pred is specified, only applies the function to the leaves that
    satisfy the predicate.

    Supported container nodes:
      - dict          (maps over values, preserves keys)
      - list
      - tuple
      - namedtuple instances

    Everything else is treated as a leaf.
    """
    if isinstance(x, dict):
        return {k: map_leaves(v, leaf_fn, pred=pred) for k, v in x.items()}

    if is_namedtuple(x):
        return type(x)(*(map_leaves(v, leaf_fn, pred=pred) for v in x))

    if isinstance(x, tuple):
        return tuple(map_leaves(v, leaf_fn, pred=pred) for v in x)

    if isinstance(x, list):
        return [map_leaves(v, leaf_fn, pred=pred) for v in x]

    if pred is None or pred(x):
        return leaf_fn(x)
    else:
        return x


def iter_leaves(
        x: Any,
        pred: Callable[[Any], bool]|None=None) -> Iterator[Any]:
    """
    Yield leaves from a plain Python tree in left-to-right traversal order.
    if pred is specified, only yields the leaves matching the predicate.
    """
    if isinstance(x, dict):
        for v in x.values():
            yield from iter_leaves(v, pred=pred)
        return

    if is_namedtuple(x):
        for v in x:
            yield from iter_leaves(v, pred=pred)
        return

    if isinstance(x, (tuple, list)):
        for v in x:
            yield from iter_leaves(v, pred=pred)
        return

    if pred is None or pred(x):
        yield x


def first_leaf(x: Any, pred: Callable[[Any], bool]|None=None) -> Any:
    """
    Return the first leaf encountered in a left-to-right traversal.

    Raises
    ------
    ValueError
        If the tree is empty (for example {}, [], (), or an empty namedtuple).
    """
    try:
        return next(iter_leaves(x, pred=pred))
    except StopIteration as e:
        raise ValueError("Cannot get first leaf from an empty tree, or no leaves matched the predicate.") from e


def leaf_values(x: Any, pred: Callable[[Any], bool]|None=None) -> list[Any]:
    """
    Collect all leaves into a list.
    """
    return list(iter_leaves(x, pred=pred))
