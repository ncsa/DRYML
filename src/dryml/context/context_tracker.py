"""
A module for tracking the currently available computing context
"""


from contextlib import contextmanager
from typing import Type


_current_context = None


contexts = {
}


def context():
    global _current_context
    return _current_context


def get_context_class(ctx_name):
    global contexts
    return contexts[ctx_name][0]


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


class WrongContextError(Exception):
    """
    Signals the wrong context is active.
    """
    def __init__(self, msg):
        super().__init__(msg)


class NoContextError(Exception):
    """
    Signals there is no context active.
    """
    def __init__(self, msg):
        super().__init__(msg)


class ContextIncompatibilityError(Exception):
    """
    Signals there is no single context which satisfies
    all requirements.
    """
    def __init__(self, msg):
        super().__init__(msg)


class ComputeContext(object):
    def acquire_context(self):
        global _current_context
        if _current_context is not None:
            raise ContextAlreadyActiveError()
        _current_context = self

    def release_context(self):
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
    contexts[name] = (ctx_cls, make_context_manager(ctx_cls))


register_context_manager('default', ComputeContext)


def consolidate_contexts(ctx_name_list):
    """
    Find a single context which satisfies all listed context
    requirements.
    """

    # Build list of unique context names
    ctx_name_list = list(set(ctx_name_list))
    ctx_cls_list = list(map(
        lambda name: get_context_class(name),
        ctx_name_list))

    # Get MRO for each class
    ctx_cls_mros = list(map(
        lambda cls: cls.mro(),
        ctx_cls_list))

    # Compute how many context classes from the
    # ctx_cls_list are contained in each mro

    def mro_count(mro, ctx_cls_list):
        return len(list(filter(lambda cls: cls in mro, ctx_cls_list)))

    ctx_cls_mro_contain_count = list(map(
        lambda mro: mro_count(mro, ctx_cls_list),
        ctx_cls_mros))

    # Find maximum
    max_v = ctx_cls_mro_contain_count[0]
    max_i = 0
    for i in range(len(ctx_cls_mro_contain_count)):
        if ctx_cls_mro_contain_count[i] > max_v:
            max_i = i
            max_v = ctx_cls_mro_contain_count[i]

    if max_v < len(ctx_cls_list):
        raise ContextIncompatibilityError(
            "Was unable to find a single context..")

    return ctx_name_list[max_i]
