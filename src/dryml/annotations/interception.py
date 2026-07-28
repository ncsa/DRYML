"""Trusted direct-call enforcement for hard annotation fragments.

Only wrappers registered here participate in enforcement and source unwrapping.
Public ``__wrapped__`` attributes are intentionally not an authority boundary.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
import types
import warnings
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import update_wrapper
from typing import Any, Callable

from dryml.environments import inspect_current
from dryml.runtime import RuntimeEnforcement, RuntimeMode, active_runtime
from dryml.runtime.allocation import is_no_allocation
from dryml.runtime.errors import RuntimeTransitionError
from dryml.runtime.publication import publication

from .errors import AnnotationResolutionError
from .merge import resolve_fragments


_WRAPPERS: weakref.WeakKeyDictionary[types.FunctionType, types.FunctionType] = weakref.WeakKeyDictionary()
_WRAPPER_OWNERS: weakref.WeakKeyDictionary[types.FunctionType, tuple[type, str]] = weakref.WeakKeyDictionary()
_WRAPPER_LOCK = threading.RLock()
_INSTRUMENTED_CLASSES: weakref.WeakSet[type] = weakref.WeakSet()
_BYPASS: contextvars.ContextVar["_Bypass | None"] = contextvars.ContextVar(
    "dryml_annotation_direct_call_bypass", default=None
)
_BYPASS_LIFECYCLE: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "dryml_annotation_direct_call_bypass_lifecycle", default=None
)


class _Bypass:
    """Private, lifecycle-bound authority for one trusted wrapper invocation."""

    __slots__ = ("target", "lifecycle", "thread", "task", "active", "used", "lock")

    def __init__(self, target: types.FunctionType, lifecycle: object) -> None:
        self.target = target
        self.lifecycle = lifecycle
        self.thread = threading.get_ident()
        self.task = _current_task()
        self.active = True
        self.used = False
        self.lock = threading.Lock()

    def admits(self, target: types.FunctionType) -> bool:
        with self.lock:
            if (
                not self.active
                or self.used
                or self.target is not target
                or self.lifecycle is not _BYPASS_LIFECYCLE.get()
                or self.thread != threading.get_ident()
                or self.task is not _current_task()
            ):
                return False
            # A copied context may retain the ContextVar value, but it cannot
            # reuse authority after the owning lifecycle has admitted its one
            # exact wrapper invocation.
            self.used = True
            return True


def trusted_original(target: Any) -> Any:
    """Return the registry-backed original body for a known direct wrapper."""

    candidate = _descriptor_function(target)
    if type(candidate) is not types.FunctionType:
        return candidate
    with _WRAPPER_LOCK:
        return _WRAPPERS.get(candidate, candidate)


def is_trusted_wrapper(target: Any) -> bool:
    """Return whether *target* is a wrapper installed by this module."""

    candidate = _descriptor_function(target)
    if type(candidate) is not types.FunctionType:
        return False
    with _WRAPPER_LOCK:
        return candidate in _WRAPPERS


@contextmanager
def _direct_call_bypass(target: Any) -> Iterator[None]:
    """Bypass only one registered target during trusted analysis/tracing.

    The context token is bound to its owning thread and asyncio task.  Copied
    contexts, foreign threads, sibling tasks, and stale tokens therefore cannot
    authorize an unrelated direct call.
    """

    wrapper = _descriptor_function(target)
    if not is_trusted_wrapper(wrapper):
        yield
        return
    lifecycle = object()
    lifecycle_token = _BYPASS_LIFECYCLE.set(lifecycle)
    authority = _Bypass(wrapper, lifecycle)
    token = _BYPASS.set(authority)
    try:
        yield
    finally:
        authority.active = False
        _BYPASS.reset(token)
        _BYPASS_LIFECYCLE.reset(lifecycle_token)


def install_requirement_wrapper(target: Any, fragment: Any) -> Any:
    """Install/reuse enforcement on a supported hard-decorated target.

    Class decoration preserves identity and shadows supported inherited methods;
    descriptors outside ordinary functions, static/class methods, and DRYML's
    managed method descriptor deliberately remain untouched.
    """

    if isinstance(target, type):
        _instrument_class(target)
        return target
    if type(target) is types.FunctionType:
        return _wrap_function(target)
    if type(target) in {staticmethod, classmethod}:
        return _wrap_builtin_descriptor(target, fragment)
    if _is_managed_descriptor(target):
        _attach_fragment(_descriptor_function(target), fragment)
        target.__func__ = _wrap_function(target.__func__)
    return target


def _instrument_class(cls: type) -> None:
    for name in _supported_method_names(cls):
        raw = _static_class_attribute(cls, name)
        # Class decoration never intercepts construction.  These are supported
        # only when their own function was explicitly hard-decorated earlier.
        if name in {"__new__", "__init__", "__call__"} and not is_trusted_wrapper(_descriptor_function(raw)):
            continue
        replacement = _instrument_descriptor(raw, cls, name)
        if replacement is not raw or name not in type.__getattribute__(cls, "__dict__"):
            setattr(cls, name, replacement)
    _install_subclass_instrumentation(cls)


def _install_subclass_instrumentation(cls: type) -> None:
    """Preserve a decorated base's hard requirements on later overrides."""

    if cls in _INSTRUMENTED_CLASSES:
        return
    original = type.__getattribute__(cls, "__dict__").get("__init_subclass__")

    def instrumented_init_subclass(subclass: type, **kwargs: Any) -> None:
        if original is None:
            super(cls, subclass).__init_subclass__(**kwargs)
        else:
            original.__get__(None, subclass)(**kwargs)
        _instrument_class(subclass)

    setattr(cls, "__init_subclass__", classmethod(instrumented_init_subclass))
    _INSTRUMENTED_CLASSES.add(cls)


def _supported_method_names(cls: type) -> tuple[str, ...]:
    names: dict[str, None] = {}
    for base in reversed(type.__getattribute__(cls, "__mro__")):
        if base is object:
            continue
        for name, value in type.__getattribute__(base, "__dict__").items():
            if _is_supported_descriptor(value):
                names[name] = None
    return tuple(names)


def _instrument_descriptor(raw: Any, owner: type, name: str) -> Any:
    if type(raw) is types.FunctionType:
        return _wrap_function(raw, owner=owner, name=name)
    if type(raw) in {staticmethod, classmethod}:
        return type(raw)(_wrap_function(raw.__func__, owner=owner, name=name))
    if _is_managed_descriptor(raw):
        # ManagedMethod owns binding and lifecycle dispatch.  Its direct body is
        # still the only supported enforcement boundary.
        raw.__func__ = _wrap_function(raw.__func__, owner=owner, name=name)
    return raw


def _wrap_builtin_descriptor(descriptor: staticmethod | classmethod, fragment: Any) -> Any:
    wrapped = _wrap_function(descriptor.__func__)
    _attach_fragment(wrapped, fragment)
    replacement = type(descriptor)(wrapped)
    _copy_own_fragments(descriptor, replacement)
    return replacement


def _wrap_function(target: types.FunctionType, *, owner: type | None = None, name: str | None = None) -> types.FunctionType:
    with _WRAPPER_LOCK:
        if target in _WRAPPERS:
            if owner is not None and name is not None:
                _WRAPPER_OWNERS[target] = (owner, name)
            return target
        original = trusted_original(target)
        if inspect.isasyncgenfunction(original):
            wrapped = _async_generator_wrapper(target)
        elif inspect.isgeneratorfunction(original):
            wrapped = _generator_wrapper(target)
        elif inspect.iscoroutinefunction(original):
            wrapped = _coroutine_wrapper(target)
        else:
            wrapped = _sync_wrapper(target)
        update_wrapper(wrapped, target)
        _WRAPPERS[wrapped] = original
        if owner is not None and name is not None:
            _WRAPPER_OWNERS[wrapped] = (owner, name)
        return wrapped


def _sync_wrapper(target: types.FunctionType) -> types.FunctionType:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        publication.current()
        if _is_bypassed(wrapped):
            return target(*args, **kwargs)
        with publication.lease() as generation:
            _check_before_body(wrapped, args, generation)
            return target(*args, **kwargs)

    return wrapped


def _coroutine_wrapper(target: types.FunctionType) -> types.FunctionType:
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        publication.current()
        if _is_bypassed(wrapped):
            return await target(*args, **kwargs)
        with publication.lease() as generation:
            _check_before_body(wrapped, args, generation)
            return await target(*args, **kwargs)

    return wrapped


def _generator_wrapper(target: types.FunctionType) -> types.FunctionType:
    def wrapped(*args: Any, **kwargs: Any):
        def iterate():
            publication.current()
            if _is_bypassed(wrapped):
                yield from target(*args, **kwargs)
                return
            with publication.lease() as generation:
                _check_before_body(wrapped, args, generation)
                yield from target(*args, **kwargs)

        return iterate()

    return wrapped


def _async_generator_wrapper(target: types.FunctionType) -> types.FunctionType:
    def wrapped(*args: Any, **kwargs: Any):
        if _is_bypassed(wrapped):
            return target(*args, **kwargs)

        return _LeasedAsyncGenerator(wrapped, target, args, kwargs)

    return wrapped


class _LeasedAsyncGenerator:
    """Forward an async generator protocol while pinning one generation."""

    __slots__ = ("_wrapper", "_target", "_args", "_kwargs", "_lease", "_inner", "_closed")

    def __init__(self, wrapper: types.FunctionType, target: types.FunctionType, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self._wrapper = wrapper
        self._target = target
        self._args = args
        self._kwargs = kwargs
        self._lease: Any = None
        self._inner: Any = None
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._advance("__anext__")

    async def asend(self, value: Any):
        return await self._advance("asend", value)

    async def athrow(self, *args: Any):
        return await self._advance("athrow", *args)

    async def aclose(self):
        if self._inner is None:
            self._closed = True
            return None
        try:
            return await self._inner.aclose()
        finally:
            self._finish()

    def __del__(self) -> None:
        """Schedule normal source finalization when an active proxy is abandoned."""

        if self._closed or self._inner is None:
            return
        try:
            asyncio.get_running_loop().create_task(self.aclose())
        except RuntimeError:
            # No event loop can drive async-generator cleanup during interpreter
            # teardown; releasing the process-generation lease is still safe.
            self._finish()

    async def _advance(self, method: str, *args: Any):
        self._start()
        try:
            return await getattr(self._inner, method)(*args)
        except BaseException:
            if self._is_finished():
                self._finish()
            raise

    def _start(self) -> None:
        if self._closed:
            raise StopAsyncIteration
        if self._inner is not None:
            return
        publication.current()
        lease = publication.lease()
        try:
            generation = lease.__enter__()
            _check_before_body(self._wrapper, self._args, generation)
            self._inner = self._target(*self._args, **self._kwargs)
            self._lease = lease
        except BaseException:
            lease.__exit__(None, None, None)
            self._closed = True
            raise

    def _is_finished(self) -> bool:
        return self._inner is not None and getattr(self._inner, "ag_frame", None) is None

    def _finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        lease, self._lease = self._lease, None
        if lease is not None:
            lease.__exit__(None, None, None)


def _check_before_body(wrapper: types.FunctionType, args: tuple[Any, ...], generation: Any) -> None:
    runtime = active_runtime()
    if runtime.enforcement is RuntimeEnforcement.OFF:
        return
    resolution = _resolve_wrapper_requirements(wrapper, args)
    if not _has_hard_requirements(resolution):
        return
    if runtime.mode is RuntimeMode.ORCHESTRATOR:
        raise RuntimeTransitionError(
            "hard-annotated direct workload calls are unavailable in orchestrator mode; dispatch the operation instead",
            context=_diagnostic_context(generation, runtime, resolution),
        )
    issues = _compatibility_issues(runtime, generation, resolution)
    if runtime.enforcement is RuntimeEnforcement.WARN:
        if issues:
            warnings.warn(
                f"dryml direct requirement warning: {_diagnostic_context(generation, runtime, resolution, issues)}",
                RuntimeWarning,
                stacklevel=3,
            )
        return
    if issues:
        raise AnnotationResolutionError(
            "direct call requirements are incompatible with the current managed runtime",
            context=_diagnostic_context(generation, runtime, resolution, issues),
        )


def _resolve_wrapper_requirements(wrapper: types.FunctionType, args: tuple[Any, ...]):
    owner, name = _runtime_method_target(wrapper, args)
    if owner is not None and name is not None:
        from .collect import fragments_for_method

        return resolve_fragments(fragments_for_method(owner, name))
    from .collect import own_fragments

    return resolve_fragments(own_fragments(wrapper))


def _runtime_method_target(wrapper: types.FunctionType, args: tuple[Any, ...]) -> tuple[type | None, str | None]:
    pinned = _WRAPPER_OWNERS.get(wrapper)
    name = pinned[1] if pinned is not None else wrapper.__name__
    if args:
        receiver = args[0]
        if isinstance(receiver, type):
            if _receiver_uses_wrapper(receiver, name, wrapper):
                return receiver, name
        # Static methods have no receiver; their owner is recovered from the
        # class instrumentation registry only when it is unambiguous below.
        if not isinstance(receiver, (str, bytes, int, float, bool, type(None))):
            receiver_type = type(receiver)
            if _receiver_uses_wrapper(receiver_type, name, wrapper):
                return receiver_type, name
    return (pinned[0] if pinned is not None else _owner_for_wrapper(wrapper)), name


def _receiver_uses_wrapper(receiver: type, name: str, wrapper: types.FunctionType) -> bool:
    """Return whether ordinary descriptor lookup would invoke this wrapper."""

    try:
        return _descriptor_function(_static_class_attribute(receiver, name)) is wrapper
    except AttributeError:
        return False


def _owner_for_wrapper(wrapper: types.FunctionType) -> type | None:
    module = inspect.getmodule(wrapper)
    if module is None:
        return None
    qualname = wrapper.__qualname__.rsplit(".", 1)[0]
    if not qualname or "<locals>" in qualname:
        return None
    current: Any = module
    for part in qualname.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, type) else None


def _compatibility_issues(runtime: Any, generation: Any, resolution: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    configuration = generation.metadata.get("session_configuration")
    if is_no_allocation(runtime.allocation):
        issues.append({"kind": "current_allowance", "message": "active runtime has no direct workload allocation"})
    elif resolution.world_requirement is not None:
        process = None if configuration is None or configuration.allocation is None else configuration.allocation.process
        issues.extend(_world_issues(runtime.allocation, resolution.world_requirement, process=process))
    environment = resolution.environment_requirement
    if configuration is not None and getattr(configuration, "mode", "python") != "python":
        environment = configuration.environment.merge(environment) if environment is not None else configuration.environment
    if environment is not None:
        report = environment.check(inspect_current(), policy="strict")
        for issue in report.issues:
            issues.append({"kind": "environment", "code": issue.code, "message": issue.message})
    return issues


def _world_issues(allocation: Any, requirement: Any, *, process: Any = None) -> list[dict[str, Any]]:
    if len(requirement.roles) != 1:
        return [{"kind": "current_allowance", "message": "one process cannot prove a multi-role world requirement"}]
    _, role = next(iter(requirement.roles.items()))
    if not role.replicas.satisfied_by(1):
        return [{"kind": "current_allowance", "message": "one process cannot prove the required replica count"}]
    resources = role.resources
    devices = dict(getattr(process, "devices", {})) if process is not None else dict(getattr(allocation, "metadata", {}).get("devices", {}))
    named = dict(getattr(process, "metadata", {})) if process is not None else dict(getattr(allocation, "metadata", {}))
    observed = {
        "cpus": len(allocation.cpus),
        "memory": allocation.memory or 0,
        "accelerators": {kind: len(devices) for kind, devices in allocation.accelerators.items()},
        "devices": devices,
        "named": named,
    }
    checks = [("cpus", resources.cpus, observed["cpus"]), ("memory", resources.memory, observed["memory"])]
    issues = [
        {"kind": "current_allowance", "path": path, "expected": constraint.to_data(), "actual": actual}
        for path, constraint, actual in checks if not constraint.satisfied_by(actual)
    ]
    for kind, constraint in resources.accelerators.items():
        actual = observed["accelerators"].get(kind, 0)
        if not constraint.satisfied_by(actual):
            issues.append({"kind": "current_allowance", "path": f"accelerators.{kind}", "expected": constraint.to_data(), "actual": actual})
    for kind, constraint in resources.devices.items():
        actual = _resource_count(observed["devices"].get(kind))
        if not constraint.satisfied_by(actual):
            issues.append({"kind": "current_allowance", "path": f"devices.{kind}", "expected": constraint.to_data(), "actual": actual})
    for kind, constraint in resources.named.items():
        actual = _resource_count(observed["named"].get(kind))
        if not constraint.satisfied_by(actual):
            issues.append({"kind": "current_allowance", "path": f"named.{kind}", "expected": constraint.to_data(), "actual": actual})
    return issues


def _resource_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (str, bytes)):
        return 1
    if isinstance(value, Mapping):
        return len(value)
    try:
        return len(value)
    except TypeError:
        return 1


def _diagnostic_context(generation: Any, runtime: Any, resolution: Any, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    configuration = generation.metadata.get("session_configuration")
    return {
        "generation": generation.number,
        "mode": runtime.mode.value,
        "current_allowance": _allocation_data(runtime.allocation),
        "worker_intent": None if configuration is None or configuration.requested_world is None else configuration.requested_world.to_data(),
        "environment_requirement": None if resolution.environment_requirement is None else resolution.environment_requirement.to_data(),
        "controls": {} if configuration is None else dict(configuration.controls),
        "issues": issues or [],
    }


def _allocation_data(allocation: Any) -> Any:
    if is_no_allocation(allocation):
        return None
    return {
        "cpus": list(allocation.cpus),
        "memory": allocation.memory,
        "accelerators": {key: list(value) for key, value in allocation.accelerators.items()},
    }


def _has_hard_requirements(resolution: Any) -> bool:
    return any((resolution.environment_requirement, resolution.world_requirement, resolution.runtime_requirement))


def _is_bypassed(wrapper: types.FunctionType) -> bool:
    authority = _BYPASS.get()
    return authority is not None and authority.admits(wrapper)


def _current_task() -> asyncio.Task[Any] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _descriptor_function(value: Any) -> Any:
    if type(value) in {staticmethod, classmethod}:
        return value.__func__
    candidate = getattr(value, "__func__", value)
    return candidate if type(candidate) is types.FunctionType else value


def _is_managed_descriptor(value: Any) -> bool:
    try:
        from dryml.managed.descriptor import ManagedMethod
    except ImportError:
        return False
    return isinstance(value, ManagedMethod)


def _is_supported_descriptor(value: Any) -> bool:
    return type(value) is types.FunctionType or type(value) in {staticmethod, classmethod} or _is_managed_descriptor(value)


def _static_class_attribute(cls: type, name: str) -> Any:
    for base in type.__getattribute__(cls, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    raise AttributeError(name)


def _attach_fragment(target: Any, fragment: Any) -> None:
    from .decorators import FRAGMENT_ATTR

    own = tuple(getattr(target, "__dict__", {}).get(FRAGMENT_ATTR, ()))
    if fragment not in own:
        setattr(target, FRAGMENT_ATTR, own + (fragment,))


def _copy_own_fragments(source: Any, target: Any) -> None:
    from .decorators import FRAGMENT_ATTR

    fragments = getattr(source, "__dict__", {}).get(FRAGMENT_ATTR)
    if fragments:
        setattr(target, FRAGMENT_ATTR, fragments)


__all__ = ["install_requirement_wrapper", "is_trusted_wrapper", "trusted_original"]
