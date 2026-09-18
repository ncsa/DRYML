"""Atomic process-local immutable configuration for Dispatch."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from types import MappingProxyType
from typing import Literal

from dryml.environments import EnvironmentRequirement
from dryml.environments.specs import EnvironmentSpec
from dryml.execute.config import BackendConfig
from dryml.worlds import WorldRequirement

from .models import DispatchView, InProcess, ProbeOptions

_INHERIT = "inherit"


@dataclass(frozen=True, slots=True)
class _Defaults:
    """Retain immutable process defaults and no live execution resources."""

    environment: EnvironmentRequirement | None = None
    world: WorldRequirement | None = None
    python: EnvironmentSpec | None = None
    backend: BackendConfig | InProcess | None = None
    core: object | None = None
    probe: ProbeOptions = ProbeOptions()
    backend_label: str | None = None
    probe_label: str | None = None


@dataclass(frozen=True, slots=True)
class _EffectiveOptions:
    """Hold one call-entry configuration snapshot for private preflight."""

    environment: EnvironmentRequirement | None
    world: WorldRequirement | None
    python: EnvironmentSpec | None
    backend: BackendConfig | InProcess | None
    core: object | None
    probe: ProbeOptions
    backend_label: str | None
    probe_label: str | None


_lock = RLock()
_defaults = _Defaults()
_backends: MappingProxyType[str, BackendConfig] = MappingProxyType({})


def _validate_core(value: object | None) -> None:
    """Validate a core override lazily without touching Store authority."""

    if value is None:
        return
    from dryml.core.execute import CoreOptions

    if not isinstance(value, CoreOptions):
        raise TypeError("dispatch core must be CoreOptions or None")


def _validate_name(name: object) -> str:
    """Return one nonempty backend registry name."""

    if type(name) is not str:
        raise TypeError("backend name must be a string")
    if not name:
        raise ValueError("backend name must be nonempty")
    return name


def _bind_backend(
    backend: BackendConfig | InProcess | str | None,
) -> tuple[BackendConfig | InProcess | None, str | None]:
    """Resolve a registry name while holding the state lock."""

    if backend is None:
        return None, None
    if type(backend) is str:
        name = _validate_name(backend)
        try:
            return _backends[name], name
        except KeyError:
            raise KeyError(f"unknown Dispatch backend {name!r}") from None
    if isinstance(backend, BackendConfig):
        return backend, type(backend).__name__
    if type(backend) is InProcess:
        return backend, None
    raise TypeError(
        "dispatch backend must be an Execute configuration, InProcess, name, "
        "or None"
    )


def _bind_probe(probe: ProbeOptions) -> tuple[ProbeOptions, str | None]:
    """Resolve a probe backend name without changing probe fields."""

    if type(probe) is not ProbeOptions:
        raise TypeError("probe must be ProbeOptions")
    backend = probe.backend
    if type(backend) is str:
        config, label = _bind_backend(backend)
        assert isinstance(config, BackendConfig)
        return replace(probe, backend=config), label
    if backend is None:
        return probe, None
    if not isinstance(backend, BackendConfig):
        raise TypeError(
            "probe backend must be an Execute configuration, name, or None"
        )
    return probe, type(backend).__name__


def register_backend(
    name: str, config: BackendConfig, *, replace: bool = False
) -> None:
    """Register one inert Execute configuration by a unique process-local name.

    Args:
        name: Nonempty name for later Dispatch binding.
        config: Inert Execute backend configuration, never a live backend.
        replace: Permit replacement of an existing name when true.

    Raises:
        TypeError: If arguments have unsupported types.
        ValueError: If the name already exists without explicit replacement.

    Side Effects:
        Atomically replaces the detached registry mapping. It does not
        construct, discover, start, reserve, or stop a backend.
    """

    global _backends
    name = _validate_name(name)
    if not isinstance(config, BackendConfig):
        raise TypeError("registered backend must be a BackendConfig")
    if type(replace) is not bool:
        raise TypeError("replace must be bool")
    with _lock:
        if name in _backends and not replace:
            raise ValueError(
                f"Dispatch backend {name!r} is already registered"
            )
        _backends = MappingProxyType({**_backends, name: config})


def unregister_backend(name: str) -> None:
    """Remove one backend name without invalidating already bound values.

    Args:
        name: Existing registry name.

    Raises:
        KeyError: If no backend has this name.

    Side Effects:
        Atomically replaces the detached registry mapping only.
    """

    global _backends
    name = _validate_name(name)
    with _lock:
        if name not in _backends:
            raise KeyError(name)
        _backends = MappingProxyType(
            {key: value for key, value in _backends.items() if key != name}
        )


def backends() -> MappingProxyType[str, BackendConfig]:
    """Return a detached immutable snapshot of registered configurations.

    Returns:
        A read-only mapping of names to inert backend configurations.

    Raises:
        None.

    Side Effects:
        Reads process-local state only. It does not inspect, create, or start a
        backend.
    """

    with _lock:
        return MappingProxyType(dict(_backends))


def set_worker_environment_default(
    environment: EnvironmentRequirement | None,
) -> None:
    """Atomically replace the worker environment requirement default.

    Args:
        environment: Hard worker software requirement, or ``None`` to clear it.

    Raises:
        TypeError: If the value is not an environment requirement or ``None``.

    Side Effects:
        Replaces process-local configuration without inspecting or changing the
        current process or a worker environment.
    """

    global _defaults
    if environment is not None and not isinstance(
        environment, EnvironmentRequirement
    ):
        raise TypeError(
            "worker environment must be an EnvironmentRequirement or None"
        )
    with _lock:
        _defaults = replace(_defaults, environment=environment)


def set_worker_world_default(world: WorldRequirement | None) -> None:
    """Atomically replace the worker world requirement default.

    Args:
        world: Hard worker resource requirement, or ``None`` to clear it.

    Raises:
        TypeError: If the value is not a world requirement or ``None``.

    Side Effects:
        Replaces process-local configuration without allocating resources or
        changing the caller's current-process allocation.
    """

    global _defaults
    if world is not None and not isinstance(world, WorldRequirement):
        raise TypeError("worker world must be a WorldRequirement or None")
    with _lock:
        _defaults = replace(_defaults, world=world)


def set_worker_python_default(python: EnvironmentSpec | None) -> None:
    """Atomically replace the exact existing worker selector default.

    Args:
        python: Existing environment selector, or ``None`` to clear the pin.

    Raises:
        TypeError: If the value is not an environment selector or ``None``.

    Side Effects:
        Replaces process-local configuration only. Resolution and launch remain
        owned by environments and Execute.
    """

    global _defaults
    if python is not None and not isinstance(python, EnvironmentSpec):
        raise TypeError("worker python must be an EnvironmentSpec or None")
    with _lock:
        _defaults = replace(_defaults, python=python)


def set_execute_backend_default(
    backend: BackendConfig | InProcess | str | None,
    *,
    core: object | None = None,
) -> None:
    """Atomically bind the default workload route and optional core controls.

    Args:
        backend: Execute configuration, registered name, :class:`InProcess`, or
            ``None`` to clear the route.
        core: Optional core Execute controls for later backend-hosted calls.

    Raises:
        TypeError: If controls have unsupported types.
        ValueError: If a supplied name is empty.
        KeyError: If a supplied registered name is unknown.

    Side Effects:
        Replaces process-local configuration. It validates inert values only.
        It never creates an Execute backend or changes caller session/runtime
        state.
    """

    global _defaults
    _validate_core(core)
    with _lock:
        bound, label = _bind_backend(backend)
        _defaults = replace(
            _defaults, backend=bound, core=core, backend_label=label
        )


def set_probe_default(probe: ProbeOptions) -> None:
    """Atomically replace the independent immutable probe policy.

    Args:
        probe: Valid immutable declaration-probe policy.

    Raises:
        TypeError: If ``probe`` is not a :class:`ProbeOptions` value.
        KeyError: If its registered backend name is unknown.

    Side Effects:
        Binds a named backend now and replaces process-local configuration; it
        does not inspect targets or create an Execute backend.
    """

    global _defaults
    with _lock:
        bound, label = _bind_probe(probe)
        _defaults = replace(_defaults, probe=bound, probe_label=label)


def with_options(
    *,
    env: EnvironmentRequirement | None | Literal["inherit"] = _INHERIT,
    world: WorldRequirement | None | Literal["inherit"] = _INHERIT,
    python: EnvironmentSpec | None | Literal["inherit"] = _INHERIT,
    backend: (
        BackendConfig | InProcess | str | None | Literal["inherit"]
    ) = _INHERIT,
    core: object | None | Literal["inherit"] = _INHERIT,
    probe: ProbeOptions | Literal["inherit"] = _INHERIT,
    _parent: DispatchView | None = None,
) -> DispatchView:
    """Create one immutable non-mutating Dispatch override view.

    Args:
        env: Worker environment requirement, ``None`` to clear, or
            ``"inherit"``.
        world: Worker world requirement, ``None`` to clear, or
            ``"inherit"``.
        python: Existing worker selector, ``None`` to clear, or ``"inherit"``.
        backend: Execute route, :class:`InProcess`, name, ``None``, or inherit.
        core: Core Execute override, ``None`` to clear, or ``"inherit"``.
        probe: Independent probe policy or ``"inherit"``.

    Returns:
        An immutable view retaining the supplied override layer.

    Raises:
        TypeError, ValueError: If an override is malformed.
        KeyError: If a supplied backend name is unknown.

    Side Effects:
        Omitted options inherit at operation entry. Explicit ``None`` clears
        nullable selections, while names for ``backend`` or ``probe.backend``
        bind to the registry now. No backend is created.
    """

    if _parent is not None and type(_parent) is not DispatchView:
        raise TypeError("dispatch view parent is invalid")
    if (
        env != _INHERIT
        and env is not None
        and not isinstance(env, EnvironmentRequirement)
    ):
        raise TypeError(
            "dispatch env must be an EnvironmentRequirement, None, "
            "or 'inherit'"
        )
    if (
        world != _INHERIT
        and world is not None
        and not isinstance(world, WorldRequirement)
    ):
        raise TypeError(
            "dispatch world must be a WorldRequirement, None, or 'inherit'"
        )
    if (
        python != _INHERIT
        and python is not None
        and not isinstance(python, EnvironmentSpec)
    ):
        raise TypeError(
            "dispatch python must be an EnvironmentSpec, None, or 'inherit'"
        )
    if core != _INHERIT:
        _validate_core(core)
    with _lock:
        if backend == _INHERIT:
            bound_backend, backend_label = _INHERIT, None
        else:
            bound_backend, backend_label = _bind_backend(backend)
        if probe == _INHERIT:
            bound_probe, probe_label = _INHERIT, None
        else:
            bound_probe, probe_label = _bind_probe(probe)
    return DispatchView(
        _parent,
        env,
        world,
        python,
        bound_backend,
        core,
        bound_probe,
        backend_label,
        probe_label,
    )


def _effective_options(view: DispatchView | None) -> _EffectiveOptions:
    """Capture defaults and every inherited view layer under one short lock."""

    if view is not None and type(view) is not DispatchView:
        raise TypeError("dispatch view is invalid")
    with _lock:
        defaults = _defaults
        layers: list[DispatchView] = []
        current = view
        while current is not None:
            layers.append(current)
            current = current._parent
        values: dict[str, object] = {
            "environment": defaults.environment,
            "world": defaults.world,
            "python": defaults.python,
            "backend": defaults.backend,
            "core": defaults.core,
            "probe": defaults.probe,
            "backend_label": defaults.backend_label,
            "probe_label": defaults.probe_label,
        }
        for layer in reversed(layers):
            for name in (
                "environment",
                "world",
                "python",
                "backend",
                "core",
                "probe",
            ):
                value = getattr(layer, f"_{name}")
                if value != _INHERIT:
                    values[name] = value
                    if name == "backend":
                        values["backend_label"] = layer._backend_label
                    elif name == "probe":
                        values["probe_label"] = layer._probe_label
        return _EffectiveOptions(**values)  # type: ignore[arg-type]


def _reset_for_testing() -> None:
    """Clear focused-test process state.

    This is not a public configuration API.
    """

    global _defaults, _backends
    with _lock:
        _defaults = _Defaults()
        _backends = MappingProxyType({})


__all__ = [
    "backends",
    "register_backend",
    "set_execute_backend_default",
    "set_probe_default",
    "set_worker_environment_default",
    "set_worker_python_default",
    "set_worker_world_default",
    "unregister_backend",
    "with_options",
]
