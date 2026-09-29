"""Immutable Dispatch configuration carriers for the later public facade."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from dryml.environments import EnvironmentRequirement
from dryml.environments.specs import EnvironmentSpec
from dryml.execute.config import BackendConfig
from dryml.requirements import RequirementResult
from dryml.worlds import WorldRequirement


@dataclass(frozen=True, slots=True)
class InProcess:
    """Select direct blocking execution in the current process.

    The marker is separate from :class:`BackendConfig`: construction neither
    selects an executor nor changes current runtime state. ``run`` performs
    fresh local admission and invokes the original synchronous callable once;
    ``submit`` rejects this route before probing.

    Raises:
        None.

    Side Effects:
        Construction has none. It neither allocates resources nor starts a
        backend.
    """


#: Select an Execute configuration, direct :class:`InProcess` route, or a
#: registered Execute configuration name. A supplied name binds to the current
#: registry value when a Dispatch default, view, or probe policy accepts it, so
#: later registry changes cannot retarget that bound selection. Unknown names
#: raise :class:`KeyError` from those consuming configuration APIs. This alias
#: has no initialization, discovery, resource, or runtime-state side effects.
BackendChoice: TypeAlias = BackendConfig | InProcess | str


class DispatchCoverageWarning(RuntimeWarning):
    """Warn that valid static requirement collection was incomplete.

    The warning carries no workload value, source, backend credentials, or
    reservation. It is emitted by ``run`` and ``submit`` only; ``explain``
    keeps the same fact in its immutable report.
    """


@dataclass(frozen=True, slots=True)
class DispatchReport:
    """Provide one bounded, non-reserving Dispatch preflight observation.

    Args:
        workload_placement: Selected in-process or Execute workload route.
        workload_backend: Redacted workload backend identifier when applicable.
        supported_methods: Operations supported by the selected route.
        probe_placement: Probe route, or ``None`` when setup failed first.
        probe_backend: Redacted selected probe backend identifier.
        probe_reason: Bounded safe explanation of probe placement or failure.
        coverage: Static coverage status when a probe completed.
        environment: Combined environment requirement result when produced.
        world: Combined world requirement result when produced.
        eligible: Whether this observation passed Dispatch preflight checks.
        diagnostics: Bounded stable diagnostic categories.
        warnings: Bounded stable warning categories.

    Raises:
        TypeError: If report fields are malformed.
        ValueError: If diagnostic bounds are exceeded.

    Side Effects:
        None. A report has no callable, arguments, live backend, Store,
        resource reservation, selector evidence, or reusable admission
        authority.
    """

    workload_placement: Literal["in_process", "execute"] | None
    workload_backend: str | None
    supported_methods: frozenset[Literal["run", "submit"]]
    probe_placement: Literal["in_process", "execute"] | None
    probe_backend: str | None
    probe_reason: str
    coverage: Literal["complete", "incomplete"] | None
    environment: RequirementResult[EnvironmentRequirement] | None
    world: RequirementResult[WorldRequirement] | None
    eligible: bool
    diagnostics: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and detach bounded public diagnostics."""

        if self.workload_placement not in {None, "in_process", "execute"}:
            raise ValueError("dispatch workload placement is invalid")
        if self.probe_placement not in {None, "in_process", "execute"}:
            raise ValueError("dispatch probe placement is invalid")
        if self.coverage not in {None, "complete", "incomplete"}:
            raise ValueError("dispatch coverage is invalid")
        if not isinstance(
            self.supported_methods, frozenset
        ) or not self.supported_methods <= {"run", "submit"}:
            raise TypeError("dispatch supported methods are invalid")
        if type(self.eligible) is not bool:
            raise TypeError("dispatch eligibility must be bool")
        for name in ("workload_backend", "probe_backend"):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not str or not value or len(value) > 512
            ):
                raise ValueError(f"dispatch {name} is invalid")
        if (
            type(self.probe_reason) is not str
            or not self.probe_reason
            or len(self.probe_reason) > 512
        ):
            raise ValueError("dispatch probe reason is invalid")
        for name in ("diagnostics", "warnings"):
            values = tuple(getattr(self, name))
            if len(values) > 64 or any(
                type(value) is not str or not value or len(value) > 512
                for value in values
            ):
                raise ValueError(f"dispatch {name} are invalid")
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class DispatchView:
    """Hold immutable Dispatch overrides without owning a live backend.

    Views retain values explicitly supplied through :func:`with_options`.
    Fields set to ``"inherit"`` read lower-precedence views and process
    defaults when an operation starts. Registered names resolve to inert
    configurations, so later registry replacement cannot retarget this view.
    """

    _parent: "DispatchView | None" = field(repr=False)
    _environment: object = field(repr=False)
    _world: object = field(repr=False)
    _python: object = field(repr=False)
    _backend: object = field(repr=False)
    _core: object = field(repr=False)
    _probe: object = field(repr=False)
    _backend_label: str | None = field(repr=False)
    _probe_label: str | None = field(repr=False)

    def with_options(self, **options: object) -> "DispatchView":
        """Create a child non-mutating override view.

        Args:
            **options: The documented Dispatch option fields.

        Returns:
            A child immutable view.

        Raises:
            TypeError, ValueError, KeyError: If a supplied option is invalid.

        Side Effects:
            Resolves explicitly supplied registry names but starts no backend.
        """

        from .api import with_options

        return with_options(_parent=self, **options)

    def explain(self, fn: Any, /, *args: Any, **kwargs: Any) -> DispatchReport:
        """Return this view's bounded non-submitting preflight report.

        Args:
            fn: Supported synchronous workload root.
            *args: Caller-owned workload positional data.
            **kwargs: Caller-owned workload data, never Dispatch controls.

        Returns:
            An immutable, non-reserving Dispatch report.

        Raises:
            TypeError, ValueError: If configuration or workload modality is
                malformed.

        Side Effects:
            May run bounded declaration discovery, but never invokes ``fn`` or
            accepts workload execution.
        """

        from .api import _explain

        return _explain(fn, args, kwargs, self)

    def run(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run one synchronous workload through this view's selected route.

        Args:
            fn: Supported synchronous workload root.
            *args: Workload positional data.
            **kwargs: Workload data, including names matching Dispatch options.

        Returns:
            The direct result for :class:`InProcess` or recovered core result
            for an Execute route.

        Raises:
            DispatchError: If preflight, admission, or target-drift validation
                fails before workload acceptance.
            BaseException: Existing workload, backend, and cleanup failures.

        Side Effects:
            Performs bounded preflight and invokes once only after the selected
            route admits the workload. It never retries or falls back.
        """

        from .api import _run

        return _run(fn, args, kwargs, self)

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Submit one synchronous workload through this view's Execute route.

        Args:
            fn: Supported synchronous workload root.
            *args: Workload positional data.
            **kwargs: Workload data, including names matching Dispatch options.

        Returns:
            The existing core execution future after backend acceptance.

        Raises:
            ValueError: If this view selects :class:`InProcess`.
            DispatchError: If preflight or target validation fails first.
            BaseException: Existing backend acceptance failures.

        Side Effects:
            Performs bounded preflight and accepts one backend-owned call. It
            neither creates a local thread nor retries or changes routes.
        """

        from .api import _submit

        return _submit(fn, args, kwargs, self)


@dataclass(frozen=True, kw_only=True, slots=True)
class ProbeOptions:
    """Describe independent, bounded placement for a declaration probe.

    Args:
        placement: ``"auto"`` chooses compatible inline inspection or a local
            subprocess; ``"in_process"`` requires inline compatibility; and
            ``"execute"`` requires generic Execute ownership.
        backend: Optional exact Execute configuration or registry name. Later
            Dispatch facade resolves a supplied name and selects Execute.
        environment: Optional bootstrap environment requirement for the probe
            worker. It is not a discovered workload requirement.
        world: Optional bootstrap world requirement for the probe worker. It is
            not a discovered workload requirement.
        environment_spec: Optional exact existing environment selector for the
            probe worker.
        execution_timeout: Finite positive deadline supplied to Execute or
            checked cooperatively at inline analysis boundaries.
        max_targets: Positive maximum static traversal targets, including root.
        max_depth: Positive maximum static traversal depth.

    Raises:
        TypeError: If a field has an unsupported type.
        ValueError: If a placement, duration, or traversal bound is invalid.

    Side Effects:
        Construction only validates values. It performs no inspection, imports,
        backend initialization, environment resolution, or execution.
    """

    placement: Literal["auto", "in_process", "execute"] = "auto"
    backend: BackendConfig | str | None = None
    environment: EnvironmentRequirement | None = None
    world: WorldRequirement | None = None
    environment_spec: EnvironmentSpec | None = None
    execution_timeout: float = 30.0
    max_targets: int = 256
    max_depth: int = 32

    def __post_init__(self) -> None:
        """Validate the inert policy without resolving external state."""

        if self.placement not in ("auto", "in_process", "execute"):
            raise ValueError("probe placement is invalid")
        if self.backend is not None and not isinstance(
            self.backend, (BackendConfig, str)
        ):
            raise TypeError(
                "probe backend must be an Execute configuration, name, or None"
            )
        if type(self.backend) is str and not self.backend:
            raise ValueError("probe backend name must be nonempty")
        if self.environment is not None and not isinstance(
            self.environment, EnvironmentRequirement
        ):
            raise TypeError(
                "probe environment must be an EnvironmentRequirement or None"
            )
        if self.world is not None and not isinstance(
            self.world, WorldRequirement
        ):
            raise TypeError("probe world must be a WorldRequirement or None")
        if self.environment_spec is not None and not isinstance(
            self.environment_spec, EnvironmentSpec
        ):
            raise TypeError(
                "probe environment_spec must be an EnvironmentSpec or None"
            )
        if (
            isinstance(self.execution_timeout, bool)
            or not isinstance(self.execution_timeout, (int, float))
            or not math.isfinite(self.execution_timeout)
            or self.execution_timeout <= 0
        ):
            raise ValueError(
                "probe execution_timeout must be finite and positive"
            )
        for name, value in (
            ("max_targets", self.max_targets),
            ("max_depth", self.max_depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"probe {name} must be an integer, not bool")
            if value <= 0:
                raise ValueError(f"probe {name} must be positive")


__all__ = [
    "BackendChoice",
    "DispatchCoverageWarning",
    "DispatchReport",
    "DispatchView",
    "InProcess",
    "ProbeOptions",
]
