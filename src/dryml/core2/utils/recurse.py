from ..errors import CycleError

# decorator to detect cycles in recursive function calls
def cycle_detect(arg_pos=0, kwarg_name=None):
    def decorator(f):
        f._path_oids = set()

        if arg_pos is not None and kwarg_name is not None:
            raise ValueError("Specify only one of arg_pos or kwarg_name")
        if arg_pos is None and kwarg_name is None:
            raise ValueError("Specify one of arg_pos or kwarg_name")

        def val_getter(*args, **kwargs):
            return args[arg_pos] if arg_pos is not None else kwargs[kwarg_name]

        def wrapper(*args, **kwargs):
            val = val_getter(*args, **kwargs)
            oid = id(val)
            if oid in f._path_oids:
                # Only raise if this is NOT a type.
                if not isinstance(val, type):
                    raise CycleError()
            f._path_oids.add(oid)
            try:
                result = f(*args, **kwargs)
            finally:
                f._path_oids.remove(oid)
            return result

        return wrapper
    return decorator
