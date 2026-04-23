from contextvars import ContextVar
from functools import wraps
from collections.abc import Callable, Iterator, Sequence
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


def map_leaf_groups(
        xs: Sequence[Any],
        leaf_fn: Callable[[list[Any]], T]) -> Any:
    """
    Recursively combine a non-empty sequence of same-structured Python trees.

    The output mirrors the input element structure, but each leaf position is
    replaced by ``leaf_fn(...)`` applied to the list of corresponding leaf
    values across all input trees.

    Supported container nodes:
      - dict          (matched by exact key order)
      - list          (matched by length)
      - tuple         (matched by length)
      - namedtuple    (matched by exact type and length)

    Everything else is treated as a leaf.

    Examples
    --------
    >>> map_leaf_groups(
    ...     [{'a': 1, 'b': (2, 3)}, {'a': 4, 'b': (5, 6)}],
    ...     list)
    {'a': [1, 4], 'b': ([2, 5], [3, 6])}
    """
    xs = list(xs)
    if not xs:
        raise ValueError("Cannot combine an empty sequence of trees.")

    def rec(vals: list[Any], path: tuple[Any, ...]) -> Any:
        x0 = vals[0]

        if isinstance(x0, dict):
            keys0 = tuple(x0.keys())
            for x in vals[1:]:
                if not isinstance(x, dict):
                    raise TypeError(
                        f"Structure mismatch at {path}: expected dict, got {type(x)!r}"
                    )
                if tuple(x.keys()) != keys0:
                    raise ValueError(
                        f"Dict key mismatch at {path}: "
                        f"expected keys {keys0}, got {tuple(x.keys())}"
                    )
            return {
                k: rec([x[k] for x in vals], path + (k,))
                for k in keys0
            }

        if is_namedtuple(x0):
            typ0 = type(x0)
            n0 = len(x0)
            for x in vals[1:]:
                if not is_namedtuple(x):
                    raise TypeError(
                        f"Structure mismatch at {path}: expected namedtuple {typ0!r}, "
                        f"got {type(x)!r}"
                    )
                if type(x) is not typ0:
                    raise TypeError(
                        f"Namedtuple type mismatch at {path}: "
                        f"expected {typ0!r}, got {type(x)!r}"
                    )
                if len(x) != n0:
                    raise ValueError(
                        f"Namedtuple length mismatch at {path}: "
                        f"expected {n0}, got {len(x)}"
                    )
            return typ0(*(rec([x[i] for x in vals], path + (i,)) for i in range(n0)))

        if isinstance(x0, tuple):
            n0 = len(x0)
            for x in vals[1:]:
                if is_namedtuple(x) or not isinstance(x, tuple):
                    raise TypeError(
                        f"Structure mismatch at {path}: expected plain tuple, got {type(x)!r}"
                    )
                if len(x) != n0:
                    raise ValueError(
                        f"Tuple length mismatch at {path}: expected {n0}, got {len(x)}"
                    )
            return tuple(rec([x[i] for x in vals], path + (i,)) for i in range(n0))

        if isinstance(x0, list):
            n0 = len(x0)
            for x in vals[1:]:
                if not isinstance(x, list):
                    raise TypeError(
                        f"Structure mismatch at {path}: expected list, got {type(x)!r}"
                    )
                if len(x) != n0:
                    raise ValueError(
                        f"List length mismatch at {path}: expected {n0}, got {len(x)}"
                    )
            return [rec([x[i] for x in vals], path + (i,)) for i in range(n0)]

        return leaf_fn(vals)

    return rec(xs, ())


def zip_leaves(xs: Sequence[Any]) -> Any:
    """
    Recursively transpose a non-empty sequence of same-structured trees so
    that each leaf becomes a list of corresponding leaf values.

    Example
    -------
    >>> zip_leaves([{'a': 1, 'b': (2, 3)}, {'a': 4, 'b': (5, 6)}])
    {'a': [1, 4], 'b': ([2, 5], [3, 6])}
    """
    return map_leaf_groups(xs, list)
