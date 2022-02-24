"""
A module for tracking the currently available computing context
"""


from contextlib import contextmanager
from typing import Type


_current_context = None


def context():
    global _current_context
    return _current_context


contexts = {
}


class ResourcesUnavailableError(Exception):
    """
    Signals a context is unable to allocate necessary resources
    """
    def __init__(self, msg):
        super().__init__(msg)


class ContextAlreadyActiveError(Exception):
    """
    Signals a context is already active.
    """
    def __init__(self, msg):
        super().__init__(msg)


class ComputeContext(object):
    def acquire_context(self):
        print("base acquire context being run")
        global _current_context
        if _current_context is not None:
            raise ContextAlreadyActiveError()
        _current_context = self
        print(f"current context: {self}")

    def release_context(self):
        print("base release context being run")
        global _current_context
        _current_context = None


def make_context_manager(ctx_cls: Type):
    @contextmanager
    def context_manager(*args, **kwargs):
        ctx_obj = ctx_cls(*args, **kwargs)
        ctx_obj.acquire_context()
        try:
            yield ctx_obj
        finally:
            ctx_obj.release_context()

    return context_manager


def register_context_manager(name: str, ctx_cls: Type):
    if name in contexts:
        raise ValueError(f"Context with name {name} already exists!")
    contexts[name] = make_context_manager(ctx_cls)


register_context_manager('default', ComputeContext)
