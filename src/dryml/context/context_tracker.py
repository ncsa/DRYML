from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Type
import multiprocessing

from dryml.core2.utils.general import get_unique_concrete_definitions
from .resource_spec import (
    ResourceAllocation,
    ResourceSpec,
    combine_resource_specs,
    normalize_compute_reqs,
)


class ContextError(Exception):
    pass


class InsufficientResourcesError(ContextError):
    pass


class ContextAlreadyActiveError(ContextError):
    pass


class NoContextError(ContextError):
    pass


class WrongContextError(ContextError):
    pass


class ContextIncompatibilityError(ContextError):
    pass


class ContextBootstrapError(ContextError):
    pass


def _detect_num_gpus() -> int:
    try:
        import GPUtil
        return len(GPUtil.getGPUs())
    except Exception:
        pass
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _detect_total_memory_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


class ResourcePool:
    def __init__(
        self,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        total_memory_bytes: int | None = None,
    ):
        if num_cpus is None:
            num_cpus = multiprocessing.cpu_count()
        if num_gpus is None:
            num_gpus = _detect_num_gpus()
        if total_memory_bytes is None:
            total_memory_bytes = _detect_total_memory_bytes()

        self.capacity: dict[str, float] = {}
        self.available: dict[str, float] = {}
        self.total_memory_bytes = total_memory_bytes
        self.available_memory_bytes = total_memory_bytes

        for i in range(num_cpus):
            key = f"cpu/{i}"
            self.capacity[key] = 1.0
            self.available[key] = 1.0

        for i in range(num_gpus):
            key = f"gpu/{i}"
            self.capacity[key] = 1.0
            self.available[key] = 1.0

    def request(self, spec: Mapping[str, Any] | ResourceSpec | None) -> ResourceAllocation:
        spec = ResourceSpec.from_user(spec)
        alloc = ResourceAllocation()

        try:
            # specific fractional requests first
            for key, need in spec.specific.items():
                have = self.available.get(key, 0.0)
                if need > have:
                    raise InsufficientResourcesError(
                        f"Requested {need} of {key}, only {have} available"
                    )
                alloc.add(key, need)
                self.available[key] = have - need

            # full GPUs
            gpu_keys = [
                key for key, avail in self.available.items()
                if key.startswith("gpu/")
                and key not in alloc.assigned
                and avail >= 1.0
            ]
            if len(gpu_keys) < spec.num_gpus:
                raise InsufficientResourcesError(
                    f"Requested {spec.num_gpus} GPUs, only {len(gpu_keys)} available"
                )
            for key in gpu_keys[:spec.num_gpus]:
                alloc.add(key, 1.0)
                self.available[key] = 0.0

            # full CPUs
            cpu_keys = [
                key for key, avail in self.available.items()
                if key.startswith("cpu/")
                and key not in alloc.assigned
                and avail >= 1.0
            ]
            if len(cpu_keys) < spec.num_cpus:
                raise InsufficientResourcesError(
                    f"Requested {spec.num_cpus} CPUs, only {len(cpu_keys)} available"
                )
            for key in cpu_keys[:spec.num_cpus]:
                alloc.add(key, 1.0)
                self.available[key] = 0.0

            if spec.memory_bytes is not None and self.available_memory_bytes is not None:
                if spec.memory_bytes > self.available_memory_bytes:
                    raise InsufficientResourcesError(
                        f"Requested {spec.memory_bytes} bytes, "
                        f"only {self.available_memory_bytes} available"
                    )
                self.available_memory_bytes -= spec.memory_bytes
                alloc.memory_bytes = spec.memory_bytes

            return alloc

        except Exception:
            self.release(alloc)
            raise

    def release(self, alloc: ResourceAllocation | None) -> None:
        if alloc is None:
            return

        for key, value in alloc.assigned.items():
            self.available[key] = min(
                self.capacity[key],
                self.available[key] + value,
            )

        if alloc.memory_bytes is not None and self.available_memory_bytes is not None:
            self.available_memory_bytes = min(
                self.total_memory_bytes,
                self.available_memory_bytes + alloc.memory_bytes,
            )


_GLOBAL_POOL: ResourcePool | None = None
_ACTIVE_CONTEXT: ContextVar["ContextContainer | None"] = ContextVar(
    "dryml_active_context",
    default=None,
)


def get_global_resource_pool() -> ResourcePool:
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        _GLOBAL_POOL = ResourcePool()
    return _GLOBAL_POOL


def active_context() -> "ContextContainer | None":
    return _ACTIVE_CONTEXT.get()


@dataclass(slots=True)
class WorkerBootstrap:
    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class ComputeContext:
    name = "base"

    def __init__(
        self,
        resource_request: Mapping[str, Any] | ResourceSpec | None = None,
        *,
        pool: ResourcePool | None = None,
    ):
        self.resource_request = ResourceSpec.from_user(resource_request)
        self.pool = get_global_resource_pool() if pool is None else pool
        self.allocation: ResourceAllocation | None = None

    # parent process
    def acquire_context(self) -> None:
        if self.allocation is not None:
            raise ContextAlreadyActiveError(f"{self.name} context is already active")
        self.allocation = self.pool.request(self.resource_request)

    def release_context(self) -> None:
        if self.allocation is None:
            return
        self.pool.release(self.allocation)
        self.allocation = None

    # worker process
    def child_env(self) -> dict[str, str]:
        return {}

    def child_setup(self) -> None:
        pass

    def child_teardown(self) -> None:
        pass


def _plain_context_loader():
    from .plain.context import PlainComputeContext
    return PlainComputeContext


def _tf_context_loader():
    from .tf.context import TFComputeContext
    return TFComputeContext


def _torch_context_loader():
    from .torch.context import TorchComputeContext
    return TorchComputeContext


def _jax_context_loader():
    from .jax.context import JAXComputeContext
    return JAXComputeContext


_CONTEXT_LOADERS: dict[str, Callable[[], Type[ComputeContext]]] = {
    "plain": _plain_context_loader,
    "tf": _tf_context_loader,
    "torch": _torch_context_loader,
    "jax": _jax_context_loader,
}
_CONTEXT_CACHE: dict[str, Type[ComputeContext]] = {}


def get_context_class(ctx_name: str) -> Type[ComputeContext]:
    if ctx_name not in _CONTEXT_CACHE:
        _CONTEXT_CACHE[ctx_name] = _CONTEXT_LOADERS[ctx_name]()
    return _CONTEXT_CACHE[ctx_name]


class ContextContainer:
    def __init__(self, resource_requests: Mapping[str, Any] | None = None):
        self.resource_requests = normalize_compute_reqs(resource_requests)
        self.contexts: dict[str, ComputeContext] = {}
        self._token = None

    def acquire_context(self) -> None:
        if active_context() is not None:
            raise ContextAlreadyActiveError("Another context is already active")

        acquired: list[str] = []
        try:
            for ctx_name, spec in self.resource_requests.items():
                ctx_cls = get_context_class(ctx_name)
                ctx = ctx_cls(resource_request=spec)
                ctx.acquire_context()
                self.contexts[ctx_name] = ctx
                acquired.append(ctx_name)
        except Exception:
            for ctx_name in reversed(acquired):
                self.contexts[ctx_name].release_context()
            self.contexts.clear()
            raise

        self._token = _ACTIVE_CONTEXT.set(self)

    def release_context(self) -> None:
        for ctx_name in reversed(list(self.contexts.keys())):
            self.contexts[ctx_name].release_context()
        self.contexts.clear()

        if self._token is not None:
            _ACTIVE_CONTEXT.reset(self._token)
            self._token = None

    def worker_bootstrap(self) -> WorkerBootstrap:
        result = WorkerBootstrap()
        for ctx_name in self.contexts:
            env = self.contexts[ctx_name].child_env()
            for key, val in env.items():
                if key in result.env and result.env[key] != val:
                    raise ContextIncompatibilityError(
                        f"Conflicting bootstrap env for {key}: "
                        f"{result.env[key]!r} vs {val!r}"
                    )
                result.env[key] = val
        return result


def set_context(
    resource_requests: Mapping[str, Any] | None = None,
    *,
    replace: bool = False,
) -> ContextContainer:
    """
    Imperatively activate a context and leave it active until cleared.

    Useful for notebooks:
        ctx = set_context({"torch": {"num_gpus": 1}})
        ...
        clear_context()
    """
    current = active_context()
    if current is not None:
        if not replace:
            raise ContextAlreadyActiveError("Another context is already active")
        current.release_context()

    mgr = ContextContainer(resource_requests)
    mgr.acquire_context()
    return mgr


def clear_context() -> None:
    """
    Clear the currently active context, if any.
    """
    current = active_context()
    if current is not None:
        current.release_context()


@contextmanager
def use_context(resource_requests: Mapping[str, Any] | None = None):
    mgr = ContextContainer(resource_requests)
    mgr.acquire_context()
    try:
        yield mgr
    finally:
        mgr.release_context()

def context_check(
    ctx_reqs: Mapping[str, Mapping[str, Any] | ResourceSpec],
) -> None:
    """
    Check that the currently active context satisfies the requested context specs.

    Example:
        context_check({"torch": {"num_gpus": 1}})
        context_check({"plain": {"num_cpus": 4}})
        context_check({
            "plain": {"num_cpus": 4},
            "tf": {"num_gpus": 1},
        })
    """
    mgr = active_context()
    if mgr is None:
        raise NoContextError("No active context")

    normalized = normalize_compute_reqs(ctx_reqs)

    for req_name, req_spec in normalized.items():
        req_cls = get_context_class(req_name)

        found_satisfier = False
        for active_name, active_ctx in mgr.contexts.items():
            active_cls = get_context_class(active_name)

            if req_cls in active_cls.mro():
                alloc = active_ctx.allocation
                if alloc is not None and alloc.satisfies(req_spec):
                    found_satisfier = True
                    break

        if not found_satisfier:
            raise ContextIncompatibilityError(
                f"Current context does not satisfy {req_name!r}: {req_spec}"
            )

def get_context_requirements(objs):
    cdefs = get_unique_concrete_definitions(objs)

    ctx_reqs: dict[str, list[ResourceSpec]] = {}
    for cdef in cdefs:
        raw_req = getattr(cdef.cls, "__compute_reqs__", None)
        if not raw_req:
            continue

        for ctx_name, spec in normalize_compute_reqs(raw_req).items():
            ctx_reqs.setdefault(ctx_name, []).append(spec)

    return {
        ctx_name: combine_resource_specs(specs)
        for ctx_name, specs in ctx_reqs.items()
    }
