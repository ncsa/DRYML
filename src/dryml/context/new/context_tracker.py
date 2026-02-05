"""
A module for tracking the currently available computing context
"""


from contextlib import contextmanager
from typing import Type, Union, Optional
from dryml.core.utils import is_nonstring_iterable
from collections import UserDict
import multiprocessing
import re


# Create the global resource pool for this process
_context_manager = None


def default_context_loader():
    from dryml.context import ComputeContext
    return ComputeContext


def tf_context_loader():
    from dryml.context.tf import TFComputeContext
    return TFComputeContext


def torch_context_loader():
    from dryml.context.torch import TorchComputeContext
    return TorchComputeContext


context_loaders = {
    'default': default_context_loader,
    'tf': tf_context_loader,
    'torch': torch_context_loader,
}


class context_map(UserDict):
    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            ctx_cls = context_loaders[key]()
            self.data[key] = (ctx_cls, make_context_manager(ctx_cls))
            return self.data[key]


contexts = context_map({})


def context():
    global _context_manager
    return _context_manager


def get_context_class(ctx_name):
    global contexts
    return contexts[ctx_name][0]


def get_context_manager(ctx_name):
    global contexts
    return contexts[ctx_name][1]


class ContextContainer(object):
    """
    Manages the creation of a compute context objects.
    """
    def __init__(self, resource_requests: Optional[dict] = {'default': {}}):
        self.resource_requests = resource_requests
        self.contexts = {}
        self.known_objects = []

    def acquire_context(self):
        # Check that there isn't another manager active already
        global _context_manager
        if _context_manager is not None:
            raise ContextAlreadyActiveError()

        # Acquire needed contexts and resources
        for ctx_name in self.resource_requests:
            # Acquire each context in turn
            ctx_cls = get_context_class(ctx_name)
            ctx = ctx_cls(resource_request=self.resource_requests[ctx_name])
            ctx.acquire_context()
            self.contexts[ctx_name] = ctx

        # Set the global context
        _context_manager = self

    def unload_objects(self, repo=None):
        # First, we'll build a tree of known objects.
        from dryml.core2.util import get_unique_objects
        obj_list = get_unique_objects(self.known_objects)

        # First, we'll save each serializable object if we have a repo.
        if repo is not None:
            repo.save_object(obj_list)

        # Now we'll unload all `Defer` objects
        from dryml.core2.util import apply_func
        apply_func(obj_list, lambda obj: obj.__unload__())

    def release_context(self):
        # Release each contained context
        for ctx_name in self.contexts:
            ctx = self.contexts[ctx_name]
            ctx.release_context()

        global _context_manager
        # Remove current_context
        _context_manager = None

    def add_object(self, obj):
        from dryml.core2.util import get_unique_objects
        from dryml.core2.object import Remember
        if not isinstance(obj, Remember):
            raise TypeError("Can only add Remember objects to the context tracker.")

        obj_dict = get_unique_objects(self.known_objects)

        if obj.definition() in obj_dict:
            # The object is already here, return
            return True

        # We don't see this object already, so add it.
        self.known_objects.append(obj)

        ## Load the object
        #if isinstance(obj, )
        
        from dryml.core2.object import Object
        if type(obj) is not Object:
            TypeError("Can only activate Objects for computation.")
        self.activated_object_map[id(obj)] = obj

    def remove_activated_object(self, obj):
        from dryml.core.object import Object
        if type(obj) is not Object:
            TypeError("Can only activate Objects for computation.")
        del self.activated_object_map[id(obj)]

    def contains_activated_object(self, obj):
        from dryml.core.object import Object
        if type(obj) is not Object:
            TypeError("Can only activate Objects for computation.")
        if id(obj) in self.activated_object_map:
            return True
        else:
            return False

    def satisfies(self, ctx_reqs):
        for ctx_name in ctx_reqs:
            ctx_cls = get_context_class(ctx_name)
            found_satisfier = False
            for c_name in self.contexts:
                c_cls = get_context_class(c_name)
                if ctx_cls in c_cls.mro():
                    if self.contexts[c_name] \
                           .allocation.satisfies(ctx_reqs[ctx_name]):
                        found_satisfier = True
                        break
            if not found_satisfier:
                return False
        return True

    def get_num_gpus_tf(self):
        if 'tf' not in self.contexts:
            raise WrongContextError("No TF Context active.")
        return self.contexts['tf'].allocation.num_gpus

    def get_torch_devices(self):
        if 'torch' not in self.contexts:
            raise WrongContextError("No Torch Context active.")
        return self.contexts['torch'].compute_devices()


def set_context(ctx_reqs):
    """
    Set a context globally. Good for local compute sessions
    """
    ctx_manager = ContextContainer(resource_requests=ctx_reqs)
    ctx_manager.acquire_context()


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


ContextManager = make_context_manager(ContextContainer)


def consolidate_contexts(ctx_name_list):
    """
    Find a collection of contexts satisfies all listed context
    requirements.
    """

    # Add default context in case no context names are given
    ctx_name_list.append('default')

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

    # Find context with maximum
    max_v = ctx_cls_mro_contain_count[0]
    max_i = 0
    for i in range(len(ctx_cls_mro_contain_count)):
        if ctx_cls_mro_contain_count[i] > max_v:
            max_i = i
            max_v = ctx_cls_mro_contain_count[i]
    max_ctx_name = ctx_name_list[max_i]

    # form list of leftover contexts which aren't
    # covered by the max context
    max_ctx_mro = ctx_cls_mros[max_i]
    leftover_ctx_names = []

    for i in range(len(ctx_name_list)):
        if ctx_cls_list[i] not in max_ctx_mro:
            leftover_ctx_names.append(ctx_name_list[i])

    if len(leftover_ctx_names) > 0:
        return [max_ctx_name] + consolidate_contexts(leftover_ctx_names)
    else:
        return [max_ctx_name]


def get_object_context_hints(obj):
    # TODO: allow instance hints as well?
    hints = []
    for cls in obj.__class__.__mro__:
        if hasattr(cls, '__compute_context_hint__'):
            hints.append(cls.__compute_context_hint__)
    return hints


def get_context_requirements(objs):
    """
    Set a context appropriate for the object or set of objects
    """
    from dryml.core2.object import Object
    from dryml.core2.util import get_unique_objects

    ic(objs)

    if issubclass(type(objs), Object):
        objs = [objs]

    if not is_nonstring_iterable(objs):
        raise ValueError(
            "set_appropriate_context only supports single "
            "Objects or an iterable of Objects.")

    ctx_reqs = {}

    objs_dict = get_unique_objects(objs)

    ic(objs)

    hints = []

    for obj in objs_dict.values():
        hints += get_object_context_hints(obj)

    context_reqs = {}
    for hint in hints:
        # Turn hints into requirements if needed, otherwise we assume they're already formatted correctly
        # TODO: Move this to a dedicated method?
        if type(hint) is str:
            hint = {hint: {}}

        for hint_k in hint:
            if hint_k not in context_reqs:
                context_reqs[hint_k] = [hint[hint_k]]
            else:
                context_reqs[hint_k].append(hint[hint_k])

    for ctx_name in context_reqs:
        ctx_reqs[ctx_name] = combine_requests(context_reqs[ctx_name])

    return ctx_reqs
