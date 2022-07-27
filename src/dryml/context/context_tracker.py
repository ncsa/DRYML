"""
A module for tracking the currently available computing context
"""


from contextlib import contextmanager
from typing import Type, Union, Optional
from dryml.utils import is_nonstring_iterable
from collections import UserDict
import multiprocessing
import re


specific_resource_re = re.compile('^(cpu|gpu)/([0-9]+)$')


class InsufficientResourcesError(Exception):
    pass


class ResourceRequest(UserDict):
    """
    A request of specific resources.
    """

    def __init__(
            self, *args, **res_map):
        super().__init__()
        for arg in args:
            if type(arg) is not dict:
                raise TypeError("Non dict arguments not supported")
            self.data.update(**arg)
        self.data.update(**res_map)
        if len(self.data.keys()):
            # Nothing passed, by default only get one cpu.
            self['num_cpus'] = 1

    def __setitem__(self, key, val):
        if type(val) not in [float, int]:
            raise ValueError(
                "Can't set a non-float or non-int resource fraction")
        if key in ['num_gpus', 'num_cpus']:
            self.data[key] = val
        else:
            spec_resource_match = specific_resource_re.match(key)
            if not spec_resource_match:
                raise KeyError(f"Resource key {key} not valid.")
            self.data[key] = val


def combine_requests(requests: [ResourceRequest]):
    # In the future I might expand functionality
    # here to do combinations other than max.
    all_keys = []
    for req in requests:
        all_keys += list(req.keys())

    result_request = ResourceRequest()

    for key in all_keys:
        max_val = max(map(
            lambda r: r[key],
            filter(lambda r: key in r, requests)))
        result_request[key] = max_val

    return result_request


def resource_request_builder(req_dict):
    return ResourceRequest(req_dict)


class ResourceAllocation(UserDict):
    def __init__(
            self, **resource_map):
        super().__init__()
        self.data.update(**resource_map)

    @property
    def gpus(self):
        return list(filter(lambda k: 'gpu' in k, self.keys()))

    @property
    def num_gpus(self):
        return len(self.gpus)

    @property
    def cpus(self):
        return list(filter(lambda k: 'cpu' in k, self.keys()))

    @property
    def num_cpus(self):
        return len(self.cpus)

    def satisfies(self, request: Optional[Union[ResourceRequest, dict]] = {}):
        # Handle construction of the resource request object
        if type(request) is dict:
            request = ResourceRequest(request)

        # Get list of specific request keys
        specific_keys = list(filter(
            lambda k: k not in ['num_cpus', 'num_gpus'],
            request.keys()))
        for key in specific_keys:
            if self[key] < request[key]:
                return False

        # Get remaining resource keys
        remaining_resource_keys = list(filter(
            lambda k: k not in specific_keys,
            self.keys()))

        # Check if we have enough to satisfy num_gpu/num_cpus
        if 'num_gpus' in request:
            if request['num_gpus'] == -1:
                # All remaining gpus are requested. for now,
                # don't apply any constraint
                pass
            else:
                num_gpu_keys = len(list(filter(
                    lambda k: 'gpu' in k,
                    remaining_resource_keys)))
                if num_gpu_keys < request['num_gpus']:
                    return False
        if 'num_cpus' in request:
            if request['num_cpus'] == -1:
                # All remaining cpus are requested. for now,
                # don't apply any constraint
                pass
            else:
                num_cpu_keys = len(list(filter(
                    lambda k: 'cpu' in k,
                    remaining_resource_keys)))
                if num_cpu_keys < request['num_cpus']:
                    return False

        return True


class ResourcePool(object):
    """
    For now, the cpu entries of a resource pool are essentially dummy
    variables. They aren't really doing anything. This can be expanded
    later to handle parallelized algorithms

    Perhaps for now, 'num_gpus' should indicate how many 'full' gpus
    """
    def __init__(self, num_cpus=None, num_gpus=None, _test=False):
        self.resource_map = {}
        if not _test:
            total_num_cpus = len(list(range(multiprocessing.cpu_count())))
            if num_cpus is None:
                num_cpus = total_num_cpus
            elif num_cpus > total_num_cpus:
                raise InsufficientResourcesError("There are not enough cpus!")
        else:
            if num_cpus is None:
                raise ValueError(
                    "In testing, you must pass a value to 'num_cpus'.")

        for i in range(num_cpus):
            self.resource_map[f"cpu/{i}"] = 1.

        if not _test:
            import GPUtil
            total_num_gpus = len(GPUtil.getGPUs())
            if num_gpus is None:
                num_gpus = len(GPUtil.getGPUs())
            elif num_gpus > total_num_gpus:
                raise InsufficientResourcesError("There are not enough gpus!")
        else:
            if num_gpus is None:
                raise ValueError(
                    "In testing, you must pass a value to 'num_gpus'.")

        for i in range(num_gpus):
            self.resource_map[f"gpu/{i}"] = 1.

    def __repr__(self):
        return self.resource_map.__repr__()

    def __str__(self):
        return self.resource_map.__str__()

    @property
    def cpus(self):
        return list(filter(lambda k: 'cpu' in k, self.resource_map.keys()))

    @property
    def num_cpus(self):
        return len(self.cpus)

    @property
    def gpus(self):
        return list(filter(lambda k: 'gpu' in k, self.resource_map.keys()))

    @property
    def num_gpus(self):
        return len(self.gpus)

    def request(self, resource_request: Union[ResourceRequest, dict]):
        # Handle construction of the resource request object
        if type(resource_request) is dict:
            resource_request = ResourceRequest(resource_request)

        alloc = ResourceAllocation()
        # First, we should allocate the specific resource requests
        for key in resource_request:
            spec_match = specific_resource_re.match(key)
            if spec_match is not None:
                avail_val = self.resource_map[key]
                ask_val = resource_request[key]
                if ask_val > avail_val:
                    self.release(alloc)
                    raise InsufficientResourcesError(
                        f"{key} doesn't have enough availability")
                alloc[key] = ask_val
                self.resource_map[key] = avail_val-ask_val

        if 'num_gpus' in resource_request and \
                resource_request['num_gpus'] != -1:
            gpu_keys = list(filter(
                lambda s: 'gpu' in s,
                self.resource_map.keys()))
            gpu_keys = sorted(
                gpu_keys,
                key=lambda k: self.resource_map[k],
                reverse=True)
            for i in range(resource_request['num_gpus']):
                gpu_key = gpu_keys[i]
                if self.resource_map[gpu_key] < 1.:
                    self.release(alloc)
                    raise InsufficientResourcesError(
                        "Couldn't find a gpu with full availability!")
                alloc[gpu_key] = 1.
                self.resource_map[gpu_key] = 0.
        else:
            if 'num_gpus' in resource_request:
                if resource_request['num_gpus'] == -1:
                    # Allocate all remaining gpus that aren't yet allocated.
                    already_allocated_gpu_keys = alloc.gpus
                    available_gpu_keys = self.gpus
                    remaining_gpu_keys = list(filter(
                        lambda k: k not in already_allocated_gpu_keys,
                        available_gpu_keys))
                    remaining_gpu_keys = list(filter(
                        lambda k: self.resource_map[k] == 1.,
                        remaining_gpu_keys))
                    for gpu_key in remaining_gpu_keys:
                        alloc[gpu_key] = 1.
                        self.resource_map[gpu_key] = 0.
                else:
                    # Report error
                    self.release(alloc)
                    raise ValueError(
                        f"'num_gpus' value {resource_request['num_gpus']}"
                        "not supported.")

        # Next, allocate general resource requests
        if 'num_cpus' in resource_request and \
                resource_request['num_cpus'] != -1:
            cpu_keys = list(filter(
                lambda s: 'cpu' in s,
                self.resource_map.keys()))
            cpu_keys = sorted(
                cpu_keys,
                key=lambda k: self.resource_map[k],
                reverse=True)
            for i in range(resource_request['num_cpus']):
                cpu_key = cpu_keys[i]
                if self.resource_map[cpu_key] < 1.:
                    self.release(alloc)
                    raise InsufficientResourcesError(
                        "Couldn't find a cpu with full availability!")
                alloc[cpu_key] = 1.
                self.resource_map[cpu_key] = 0.
        else:
            if 'num_cpus' in resource_request:
                if resource_request['num_cpus'] == -1:
                    already_allocated_cpu_keys = alloc.cpus
                    available_cpu_keys = self.cpus
                    remaining_cpu_keys = list(filter(
                        lambda k: k not in already_allocated_cpu_keys,
                        available_cpu_keys))
                    remaining_cpu_keys = list(filter(
                        lambda k: self.resource_map[k] == 1.,
                        remaining_cpu_keys))
                    for cpu_key in remaining_cpu_keys:
                        alloc[cpu_key] = 1.
                        self.resource_map[cpu_key] = 0.
                else:
                    self.release(alloc)
                    raise ValueError(
                        f"'num_cpus' value {resource_request['num_cpus']} "
                        "not supported.")

        return alloc

    def release(self, alloc: ResourceAllocation):
        for key in alloc:
            self.resource_map[key] += alloc[key]


# Create the global resource pool for this process
_resource_pool = ResourcePool()
_context_manager = None


def default_context_loader():
    from dryml.context import ComputeContext
    return ComputeContext


def tf_context_loader():
    from dryml.context.tf import TFComputeContext
    return TFComputeContext


context_loaders = {
    'default': default_context_loader,
    'tf': tf_context_loader,
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


class ResourcesUnavailableError(Exception):
    """
    Signals a context is unable to allocate necessary resources
    """
    pass


class ContextAlreadyActiveError(Exception):
    """
    Signals a context is already active.
    """
    pass


class WrongContextError(Exception):
    """
    Signals the wrong context is active.
    """
    pass


class NoContextError(Exception):
    """
    Signals there is no context active.
    """
    pass


class ContextIncompatibilityError(Exception):
    """
    Signals there is no single context which satisfies
    all requirements.
    """
    pass


class ComputeContext(object):
    def __init__(
            self,
            resource_request: Optional[Union[ResourceRequest, dict]] = {}):
        if type(resource_request) is dict:
            resource_request = ResourceRequest(resource_request)
        self.resource_request = resource_request
        self.allocation = None

    def acquire_context(self):
        global _resource_pool
        # Acquire allocation
        self.allocation = _resource_pool.request(self.resource_request)

    def release_context(self):
        global _resource_pool
        # Release allocation
        _resource_pool.release(self.allocation)
        self.allocation = None


class ContextContainer(object):
    """
    Manages the creation of a compute context objects.
    """
    def __init__(self, resource_requests: Optional[dict] = {'default': {}}):
        self.resource_requests = resource_requests
        self.contexts = {}
        self.activated_object_map = {}

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

    def deactivate_objects(self, save_cache=None):
        if save_cache is None:
            from dryml.save_cache import SaveCache
            save_cache = SaveCache()
        while len(self.activated_object_map) > 0:
            ids = list(self.activated_object_map.keys())
            obj_id = ids[0]
            obj = self.activated_object_map[obj_id]
            obj.compute_deactivate(save_cache=save_cache)

    def release_context(self):
        # Deactivate each tracked object
        self.deactivate_objects()

        # Release each contained context
        for ctx_name in self.contexts:
            ctx = self.contexts[ctx_name]
            ctx.release_context()

        global _context_manager
        # Remove current_context
        _context_manager = None

    def add_activated_object(self, obj):
        from dryml.dry_object import DryObject
        if type(obj) is not DryObject:
            TypeError("Can only activate DryObjects for computation.")
        self.activated_object_map[id(obj)] = obj

    def remove_activated_object(self, obj):
        from dryml.dry_object import DryObject
        if type(obj) is not DryObject:
            TypeError("Can only activate DryObjects for computation.")
        del self.activated_object_map[id(obj)]

    def contains_activated_object(self, obj):
        from dryml.dry_object import DryObject
        if type(obj) is not DryObject:
            TypeError("Can only activate DryObjects for computation.")
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
            raise RuntimeError("No TF Context active.")
        return self.contexts['tf'].allocation.num_gpus


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


def get_context_requirements(objs):
    """
    Set a context appropriate for the object or set of objects
    """
    from dryml import DryObject

    if issubclass(type(objs), DryObject):
        objs = [objs]

    if not is_nonstring_iterable(objs):
        raise ValueError(
            "set_appropriate_context only supports single "
            "DryObjects or an iterable of DryObjects.")

    ctx_reqs = {}

    for obj in objs:
        obj_reqs = obj.dry_context_requirements()
        for ctx_name in obj_reqs:
            if ctx_name in ctx_reqs:
                ctx_reqs[ctx_name].append(obj_reqs[ctx_name])
            else:
                ctx_reqs[ctx_name] = [obj_reqs[ctx_name]]

    for ctx_name in ctx_reqs:
        ctx_reqs[ctx_name] = combine_requests(ctx_reqs[ctx_name])

    return ctx_reqs
