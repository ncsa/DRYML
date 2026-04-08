from contextvars import ContextVar
from functools import wraps

from ..errors import CycleError


_ATOMIC_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    bytes,
    str,
)


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
