from dataclasses import dataclass
import inspect

@dataclass(frozen=True)
class CallableInfo:
    """
    Normalized type to hold information about a callable
    """
    original: object
    func: object
    bound_self: object | None
    signature: inspect.Signature
    qualname: str | None
    module: str | None
    is_bound_method: bool
    is_function: bool
    is_callable_instance: bool


def analyze_callable(obj) -> CallableInfo:
    """
    Method to take an arbitrary python callable object and deduce information about it.
    """
    if inspect.ismethod(obj):
        func = obj.__func__
        bound_self = obj.__self__
        return CallableInfo(
            original=obj,
            func=func,
            bound_self=bound_self,
            signature=inspect.signature(func),
            qualname=getattr(func, "__qualname__", None),
            module=getattr(func, "__module__", None),
            is_bound_method=True,
            is_function=False,
            is_callable_instance=False,
        )

    if inspect.isfunction(obj):
        return CallableInfo(
            original=obj,
            func=obj,
            bound_self=None,
            signature=inspect.signature(obj),
            qualname=getattr(obj, "__qualname__", None),
            module=getattr(obj, "__module__", None),
            is_bound_method=False,
            is_function=True,
            is_callable_instance=False,
        )

    if callable(obj):
        call = getattr(type(obj), "__call__", None)
        if call is None:
            raise TypeError(f"Callable object {obj!r} has no type.__call__")
        return CallableInfo(
            original=obj,
            func=call,
            bound_self=obj,
            signature=inspect.signature(call),
            qualname=getattr(call, "__qualname__", None),
            module=getattr(call, "__module__", None),
            is_bound_method=False,
            is_function=False,
            is_callable_instance=True,
        )

    raise TypeError(f"Object {obj!r} is not callable")
