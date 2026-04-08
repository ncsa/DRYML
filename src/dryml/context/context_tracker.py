from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Type
import multiprocessing
import threading

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

        self._lock = threading.RLock()

        for i in range(num_cpus):
            key = f"cpu/{i}"
            self.capacity[key] = 1.0
            self.available[key] = 1.0

        for i in range(num_gpus):
            key = f"gpu/{i}"
            self.capacity[key] = 1.0
            self.available[key] = 1.0

    def _ordered_candidate_keys(
        self,
        prefix: str,
        already_assigned: set[str],
        prefer_keys: list[str] | None = None,
    ) -> list[str]:
        prefer_keys = prefer_keys or []

        preferred = [
            key for key in prefer_keys
            if key.startswith(prefix)
            and key not in already_assigned
            and self.available.get(key, 0.0) >= 1.0
        ]

        others = sorted(
            key for key, avail in self.available.items()
            if key.startswith(prefix)
            and key not in already_assigned
            and key not in preferred
            and avail >= 1.0
        )

        return preferred + others

    def _reserve_exact_locked(self, alloc: ResourceAllocation | None) -> None:
        if alloc is None:
            return

        for key, value in alloc.assigned.items():
            have = self.available.get(key, 0.0)
            if value > have:
                raise InsufficientResourcesError(
                    f"Unable to re-reserve {value} of {key}, only {have} available"
                )
            self.available[key] = have - value

        if alloc.memory_bytes is not None and self.available_memory_bytes is not None:
            if alloc.memory_bytes > self.available_memory_bytes:
                raise InsufficientResourcesError(
                    f"Unable to re-reserve {alloc.memory_bytes} bytes, "
                    f"only {self.available_memory_bytes} available"
                )
            self.available_memory_bytes -= alloc.memory_bytes

    def _release_locked(self, alloc: ResourceAllocation | None) -> None:
        if alloc is None:
            return

        for key, value in alloc.assigned.items():
            self.available[key] = min(
                self.capacity[key],
                self.available.get(key, 0.0) + value,
            )

        if alloc.memory_bytes is not None and self.available_memory_bytes is not None:
            self.available_memory_bytes = min(
                self.total_memory_bytes,
                self.available_memory_bytes + alloc.memory_bytes,
            )

    def _request_locked(
        self,
        spec: Mapping[str, Any] | ResourceSpec | None,
        *,
        prefer_keys: list[str] | None = None,
    ) -> ResourceAllocation:
        spec = ResourceSpec.from_user(spec)
        alloc = ResourceAllocation()

        try:
            # specific fractional requests first
            for key, need in sorted(spec.specific.items()):
                have = self.available.get(key, 0.0)
                if need > have:
                    raise InsufficientResourcesError(
                        f"Requested {need} of {key}, only {have} available"
                    )
                alloc.add(key, need)
                self.available[key] = have - need

            # full GPUs
            gpu_keys = self._ordered_candidate_keys(
                "gpu/",
                already_assigned=set(alloc.assigned),
                prefer_keys=prefer_keys,
            )
            if len(gpu_keys) < spec.num_gpus:
                raise InsufficientResourcesError(
                    f"Requested {spec.num_gpus} GPUs, only {len(gpu_keys)} available"
                )
            for key in gpu_keys[:spec.num_gpus]:
                alloc.add(key, 1.0)
                self.available[key] = 0.0

            # full CPUs
            cpu_keys = self._ordered_candidate_keys(
                "cpu/",
                already_assigned=set(alloc.assigned),
                prefer_keys=prefer_keys,
            )
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
            self._release_locked(alloc)
            raise

    def request(self, spec: Mapping[str, Any] | ResourceSpec | None) -> ResourceAllocation:
        with self._lock:
            return self._request_locked(spec)

    def release(self, alloc: ResourceAllocation | None) -> None:
        with self._lock:
            self._release_locked(alloc)

    def revise(
        self,
        alloc: ResourceAllocation | None,
        spec: Mapping[str, Any] | ResourceSpec | None,
    ) -> ResourceAllocation:
        """
        Atomically revise an existing allocation to satisfy `spec`.

        Properties:
        - serialized by the pool lock
        - prefers to keep currently-held CPU/GPU ids when possible
        - rolls back to the old allocation if the new request fails
        - mutates the passed-in `alloc` in place when alloc is not None
        """
        spec = ResourceSpec.from_user(spec)

        with self._lock:
            if alloc is None:
                return self._request_locked(spec)

            old_alloc = ResourceAllocation(
                assigned=dict(alloc.assigned),
                memory_bytes=alloc.memory_bytes,
            )
            prefer_keys = list(old_alloc.assigned.keys())

            # Temporarily release old resources back into the pool so the new
            # request can be satisfied against the pool's full visible state.
            self._release_locked(old_alloc)

            try:
                new_alloc = self._request_locked(spec, prefer_keys=prefer_keys)
            except Exception:
                # Roll back exactly to the original allocation.
                self._reserve_exact_locked(old_alloc)
                raise

            alloc.assigned.clear()
            alloc.assigned.update(new_alloc.assigned)
            alloc.memory_bytes = new_alloc.memory_bytes
            return alloc
    

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
        self._applied = False
        self.check_compatible_env()

    def check_compatible_env(self):
        pass

    def acquire_resources(self) -> None:
        if self.allocation is not None:
            raise ContextAlreadyActiveError(f"{self.name} context is already active")
        self.allocation = self.pool.request(self.resource_request)

    def release_resources(self) -> None:
        if self.allocation is None:
            return
        self.pool.release(self.allocation)
        self.allocation = None

    def revise_resources(
        self,
        resource_request: Mapping[str, Any] | ResourceSpec | None = None,
    ) -> None:
        spec = ResourceSpec.from_user(resource_request)
        if self.allocation is None:
            self.allocation = self.pool.request(spec)
        else:
            self.allocation = self.pool.revise(self.allocation, spec)
        self.resource_request = spec

    def validate_current(self) -> None:
        """
        Raise if this context cannot be safely applied to the current runtime.
        Framework-specific contexts should override this.
        """
        self.check_compatible_env()

    def apply_current(self) -> None:
        """
        Apply best-effort runtime effects to the current process/thread.
        Framework-specific contexts should override this.
        """
        self._applied = True

    def unapply_current(self) -> None:
        """
        Undo best-effort runtime effects from apply_current().
        Framework-specific contexts should override this.
        """
        self._applied = False

    def bootstrap_env(self) -> dict[str, str]:
        """
        Environment required for a fresh process that wants to construct and
        apply an equivalent context for itself.
        """
        return {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"resource_request={self.resource_request!r}, "
            f"allocation={self.allocation!r}, "
            f"applied={self._applied})"
        )

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
        self._applied_order: list[str] = []

    def acquire_context(self) -> None:
        """
        Activate this container in the current runtime.

        Transaction:
        1) acquire resources
        2) validate current runtime feasibility
        3) apply current runtime effects
        4) install as active context
        """
        if active_context() is not None:
            raise ContextAlreadyActiveError("Another context is already active")

        acquired: list[str] = []
        applied: list[str] = []

        try:
            # 1) construct and acquire resources
            for ctx_name, spec in self.resource_requests.items():
                ctx_cls = get_context_class(ctx_name)
                ctx = ctx_cls(resource_request=spec)
                ctx.acquire_resources()
                self.contexts[ctx_name] = ctx
                acquired.append(ctx_name)

            # 2) validate final configuration
            for ctx_name in self.resource_requests:
                self.contexts[ctx_name].validate_current()

            # 3) mark active, then apply in declared order
            self._token = _ACTIVE_CONTEXT.set(self)
            for ctx_name in self.resource_requests:
                self.contexts[ctx_name].apply_current()
                applied.append(ctx_name)

            self._applied_order = list(applied)

        except Exception:
            # undo partial apply
            for ctx_name in reversed(applied):
                try:
                    self.contexts[ctx_name].unapply_current()
                except Exception:
                    pass
            self._applied_order.clear()

            # clear active token if we set it
            if self._token is not None:
                _ACTIVE_CONTEXT.reset(self._token)
                self._token = None

            # release acquired resources
            for ctx_name in reversed(acquired):
                try:
                    self.contexts[ctx_name].release_resources()
                except Exception:
                    pass

            self.contexts.clear()
            raise

    def release_context(self) -> None:
        """
        Deactivate this container in the current runtime.

        Transaction:
        1) unapply current runtime effects
        2) release resources
        3) clear active context token
        """
        try:
            for ctx_name in reversed(self._applied_order):
                try:
                    self.contexts[ctx_name].unapply_current()
                except Exception:
                    pass
            self._applied_order.clear()

            for ctx_name in reversed(list(self.contexts.keys())):
                try:
                    self.contexts[ctx_name].release_resources()
                except Exception:
                    pass
            self.contexts.clear()

        finally:
            if self._token is not None:
                _ACTIVE_CONTEXT.reset(self._token)
                self._token = None

    def bootstrap_env(self) -> dict[str, str]:
        """
        Merge bootstrap environment requirements for a fresh process that wants
        to recreate this logical context for itself.
        """
        result: dict[str, str] = {}

        for ctx_name in self.resource_requests:
            ctx = self.contexts.get(ctx_name)
            if ctx is None:
                # allow bootstrap before acquire_context() if needed
                ctx_cls = get_context_class(ctx_name)
                ctx = ctx_cls(resource_request=self.resource_requests[ctx_name])

            env = ctx.bootstrap_env()
            for key, val in env.items():
                if key in result and result[key] != val:
                    raise ContextIncompatibilityError(
                        f"Conflicting bootstrap env for {key}: "
                        f"{result[key]!r} vs {val!r}"
                    )
                result[key] = val

        return result

    def update_contexts(self, reqs=None, *, mode="add"):
        reqs = normalize_compute_reqs(reqs)

        if mode == "add":
            overlap = set(reqs) & set(self.contexts)
            if overlap:
                raise ContextAlreadyActiveError(
                    f"Contexts already active: {sorted(overlap)}"
                )
            target = {**self.resource_requests, **reqs}

        elif mode == "combine":
            from .resource_spec import combine_compute_reqs
            target = combine_compute_reqs(self.resource_requests, reqs)

        elif mode == "replace":
            target = dict(self.resource_requests)
            target.update(reqs)

        else:
            raise ValueError(f"Unknown mode {mode!r}")

        self._reconfigure_to(target)

    # -------------------------------------------------------------------------
    # Deprecated compatibility API
    # -------------------------------------------------------------------------

    def worker_bootstrap(self) -> WorkerBootstrap:
        warnings.warn(
            "ContextContainer.worker_bootstrap() is deprecated; use bootstrap_env().",
            DeprecationWarning,
            stacklevel=2,
        )
        result = WorkerBootstrap()
        result.env.update(self.bootstrap_env())
        return result

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ContextContainer:{self.contexts}"

    def _ctx_apply(self, ctx: ComputeContext) -> None:
        ctx.apply_current()

    def _ctx_unapply(self, ctx: ComputeContext) -> None:
        ctx.unapply_current()

    def _ctx_validate(self, ctx: ComputeContext) -> None:
        ctx.validate_current()

    def _ctx_acquire_resources(self, ctx: ComputeContext) -> None:
        ctx.acquire_resources()

    def _ctx_release_resources(self, ctx: ComputeContext) -> None:
        ctx.release_resources()

    def _ctx_revise_resources(
        self,
        ctx: ComputeContext,
        spec: Mapping[str, Any] | ResourceSpec | None,
    ) -> None:
        ctx.revise_resources(spec)

    @staticmethod
    def _copy_alloc(alloc: ResourceAllocation | None) -> ResourceAllocation | None:
        if alloc is None:
            return None
        return ResourceAllocation(
            assigned=dict(alloc.assigned),
            memory_bytes=alloc.memory_bytes,
        )

    def _reconfigure_to(self, target) -> None:
        target = normalize_compute_reqs(target)
        target_order = list(target.keys())

        old_resource_requests = dict(self.resource_requests)
        old_contexts = dict(self.contexts)
        old_order = list(self._applied_order)
        old_allocs = {
            name: self._copy_alloc(ctx.allocation)
            for name, ctx in old_contexts.items()
        }

        if old_contexts:
            pool = next(iter(old_contexts.values())).pool
        else:
            pool = get_global_resource_pool()

        prefix_len = 0
        max_prefix = min(len(old_order), len(target_order))
        while prefix_len < max_prefix:
            old_name = old_order[prefix_len]
            new_name = target_order[prefix_len]
            if old_name != new_name:
                break
            if old_resource_requests.get(old_name) != target.get(new_name):
                break
            prefix_len += 1

        if prefix_len == len(old_order) and prefix_len == len(target_order):
            return

        preserved_prefix = old_order[:prefix_len]
        old_suffix = old_order[prefix_len:]
        target_suffix = target_order[prefix_len:]

        working_contexts = dict(old_contexts)
        applied_suffix: list[str] = []

        with pool._lock:
            try:
                # 1) unapply only the old suffix
                for name in reversed(old_suffix):
                    if name in working_contexts:
                        self._ctx_unapply(working_contexts[name])

                # 2) remove contexts that disappear entirely
                for name in list(working_contexts.keys()):
                    if name in preserved_prefix:
                        continue
                    if name not in target:
                        ctx = working_contexts.pop(name)
                        self._ctx_release_resources(ctx)

                # 3) materialize the full target context mapping explicitly
                target_contexts: dict[str, ComputeContext] = {}

                for name in target_order:
                    spec = target[name]

                    if name in preserved_prefix:
                        # unchanged prefix: preserve exactly as-is
                        ctx = working_contexts[name]

                    elif name in working_contexts:
                        # existing context in affected suffix
                        if old_resource_requests.get(name) != spec:
                            self._ctx_revise_resources(working_contexts[name], spec)
                        ctx = working_contexts[name]

                    else:
                        # brand new context
                        ctx_cls = get_context_class(name)
                        ctx = ctx_cls(resource_request=spec, pool=pool)
                        self._ctx_acquire_resources(ctx)
                        working_contexts[name] = ctx

                    target_contexts[name] = ctx

                # sanity check: every target name must now exist
                missing = [name for name in target_order if name not in target_contexts]
                if missing:
                    raise RuntimeError(
                        f"Internal error: missing target contexts after reconfigure build: {missing}"
                    )

                # 4) validate only affected suffix
                for name in target_suffix:
                    self._ctx_validate(target_contexts[name])

                # 5) apply only affected suffix
                for name in target_suffix:
                    self._ctx_apply(target_contexts[name])
                    applied_suffix.append(name)

                # 6) commit
                self.resource_requests = dict(target)
                self.contexts = {name: target_contexts[name] for name in target_order}
                self._applied_order = list(target_order)

            except Exception:
                for name in reversed(applied_suffix):
                    try:
                        self._ctx_unapply(working_contexts[name])
                    except Exception:
                        pass

                affected_names = set(old_suffix) | set(target_suffix)
                seen_ids = set()
                for name, ctx in working_contexts.items():
                    if name not in affected_names:
                        continue
                    if id(ctx) in seen_ids:
                        continue
                    seen_ids.add(id(ctx))
                    try:
                        self._ctx_release_resources(ctx)
                    except Exception:
                        pass

                restored_contexts = dict(old_contexts)
                for name in old_suffix:
                    ctx = restored_contexts[name]
                    ctx.resource_request = old_resource_requests[name]
                    ctx.allocation = None

                    old_alloc = old_allocs[name]
                    if old_alloc is not None:
                        pool._reserve_exact_locked(old_alloc)
                        ctx.allocation = self._copy_alloc(old_alloc)

                reapplied_suffix: list[str] = []
                try:
                    for name in old_suffix:
                        self._ctx_apply(restored_contexts[name])
                        reapplied_suffix.append(name)
                except Exception:
                    kept_order = preserved_prefix + reapplied_suffix
                    self.resource_requests = dict(old_resource_requests)
                    self.contexts = {
                        name: restored_contexts[name]
                        for name in kept_order
                    }
                    self._applied_order = list(kept_order)
                    raise

                self.resource_requests = dict(old_resource_requests)
                self.contexts = {
                    name: restored_contexts[name]
                    for name in old_order
                }
                self._applied_order = list(old_order)
                raise

def set_context(
        resource_requests: Mapping[str, Any] | None = None,
        *,
        replace=False
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


def add_context(resource_requests=None, *, mode="add"):
    current = active_context()
    if current is None:
        return set_context(resource_requests)

    current.update_contexts(resource_requests, mode=mode)
    return current


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
