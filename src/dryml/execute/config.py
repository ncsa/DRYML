"""Inert, immutable common configuration for future Execute backends."""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from dryml.environments.specs import (
    CondaEnvironmentSpec,
    ContainerEnvironmentSpec,
    CurrentEnvironmentSpec,
    EnvironmentSpec,
    PythonExecutableSpec,
)

if TYPE_CHECKING:
    from .backend import Backend


_MAX_FRAME_LENGTH = (1 << 64) - 1
_MAX_HEADER_LENGTH = (1 << 32) - 1
_ENVIRONMENT_SPEC_TYPES = (
    CurrentEnvironmentSpec,
    PythonExecutableSpec,
    CondaEnvironmentSpec,
    ContainerEnvironmentSpec,
)


def _positive_duration(name: str, value: object, *, allow_none: bool = False) -> None:
    """Validate one finite positive duration without performing backend work."""
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number of seconds")
    if isinstance(value, int):
        if value > sys.float_info.max:
            raise ValueError(f"{name} must be representable as finite seconds")
    elif not math.isfinite(value):
        raise ValueError(f"{name} must be a finite positive number of seconds")
    if value <= 0:
        raise ValueError(f"{name} must be a finite positive number of seconds")


def _positive_limit(name: str, value: object, *, allow_zero: bool = False) -> None:
    """Validate a representable non-boolean operational count or byte limit."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    if value > _MAX_FRAME_LENGTH:
        raise ValueError(f"{name} exceeds the framing format range")


@dataclass(frozen=True, kw_only=True)
class BackendConfig(ABC):
    """Describe backend policy without discovery, filesystem access, or launch.

    All duration, transport, output, discovery, spool, and one-off cleanup
    controls are public so
    callers can choose operational bounds. Construction validates values and
    freezes nested containers only; a future executor validates live resources
    and joins a process-wide spool budget when it first preflights work.

    ``one_off_cleanup_attempts`` is the positive bounded number of automatic
    reconciliation attempts retained one-off owners make per cleanup budget.
    ``one_off_cleanup_retry_interval`` schedules later attempts while a retained
    owner remains incomplete. An owner remains inspectable when attempts fail.

    Raises:
        TypeError: If a field has an unsupported type.
        ValueError: If a field is non-finite, out of range, or incompatible with
            another configured limit.
    """

    admission_timeout: float = 30.0
    discovery_timeout: float = 30.0
    termination_timeout: float = 5.0
    one_off_cleanup_attempts: int = 2
    one_off_cleanup_retry_interval: float = 0.1
    execution_timeout: float | None = None
    output_final_timeout: float = 5.0
    spool_directory: Path | None = None
    spool_limit_bytes: int = 4_294_967_296
    spool_file_limit: int = 128
    preflight_limit: int = 8
    invocation_limit_bytes: int = 67_108_864
    result_limit_bytes: int = 67_108_864
    control_header_limit_bytes: int = 1_048_576
    owner_envelope_limit_bytes: int = 16_777_216
    admission_message_limit_bytes: int = 83_886_080
    output_frame_limit_bytes: int = 65_536
    output_limit_bytes: int = 1_048_576
    live_output_queue_limit_bytes: int = 262_144
    diagnostic_text_limit_bytes: int = 65_536
    diagnostic_issue_limit: int = 64
    discovery_candidate_limit: int = 128
    discovery_directory_entry_limit: int = 1024
    process_read_chunk_bytes: int = 8192
    process_poll_interval: float = 0.005
    environment_search_depth: int = 2
    stream_output: bool = False
    automatic_environment_discovery: bool = True
    conda_executable: str = "conda"
    conda_launch_mode: str = "direct"
    environment_candidates: tuple[EnvironmentSpec, ...] = ()
    environment_search_roots: tuple[Path, ...] = ()
    working_directory: Path | None = None
    env_vars: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate relationships and defensively detach caller containers."""
        for name in ("admission_timeout", "discovery_timeout", "termination_timeout", "output_final_timeout", "one_off_cleanup_retry_interval", "process_poll_interval"):
            _positive_duration(name, getattr(self, name))
        _positive_duration("execution_timeout", self.execution_timeout, allow_none=True)
        for name in (
            "one_off_cleanup_attempts", "spool_limit_bytes", "spool_file_limit", "preflight_limit", "invocation_limit_bytes",
            "result_limit_bytes", "control_header_limit_bytes", "owner_envelope_limit_bytes",
            "admission_message_limit_bytes", "output_frame_limit_bytes", "output_limit_bytes",
            "live_output_queue_limit_bytes", "diagnostic_text_limit_bytes", "diagnostic_issue_limit",
            "discovery_candidate_limit", "discovery_directory_entry_limit", "process_read_chunk_bytes",
        ):
            _positive_limit(name, getattr(self, name))
        if self.control_header_limit_bytes > _MAX_HEADER_LENGTH:
            raise ValueError("control_header_limit_bytes exceeds the 4-byte framing header range")
        _positive_limit("environment_search_depth", self.environment_search_depth, allow_zero=True)
        for name in ("stream_output", "automatic_environment_discovery"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.conda_executable, str) or not self.conda_executable:
            raise TypeError("conda_executable must be a non-empty string")
        if self.conda_launch_mode not in {"direct", "conda-run"}:
            raise ValueError("conda_launch_mode must be direct or conda-run")
        for name in ("spool_directory", "working_directory"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{name} must be pathlib.Path or None")
        if self.spool_limit_bytes < self.invocation_limit_bytes + self.result_limit_bytes:
            raise ValueError("spool_limit_bytes must cover invocation_limit_bytes plus result_limit_bytes")
        if self.spool_file_limit < 2:
            raise ValueError("spool_file_limit must reserve invocation and result files")
        if self.output_frame_limit_bytes > self.live_output_queue_limit_bytes:
            raise ValueError("output_frame_limit_bytes must not exceed live_output_queue_limit_bytes")
        if self.control_header_limit_bytes > self.admission_message_limit_bytes:
            raise ValueError("control_header_limit_bytes must not exceed admission_message_limit_bytes")
        if not isinstance(self.env_vars, Mapping):
            raise TypeError("env_vars must be a mapping")
        frozen_env: dict[str, str] = {}
        for key, value in self.env_vars.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("env_vars keys and values must be strings")
            frozen_env[key] = value
        try:
            candidates = tuple(self.environment_candidates)
            roots = tuple(self.environment_search_roots)
        except TypeError as exc:
            raise TypeError("environment candidates and roots must be iterable") from exc
        if not all(isinstance(candidate, _ENVIRONMENT_SPEC_TYPES) for candidate in candidates):
            raise TypeError("environment_candidates entries must be EnvironmentSpec values")
        if not all(isinstance(root, Path) for root in roots):
            raise TypeError("environment_search_roots entries must be pathlib.Path")
        object.__setattr__(self, "env_vars", MappingProxyType(frozen_env))
        object.__setattr__(self, "environment_candidates", candidates)
        object.__setattr__(self, "environment_search_roots", roots)

    @abstractmethod
    def create_backend(self) -> "Backend":
        """Create this configuration's inert backend without changing this config.

        Returns:
            The concrete, unstarted backend associated with this configuration.

        Side Effects:
            None. The returned backend must not initialize native SDKs, inspect
            the filesystem, or launch work until its explicit ``start`` call.
        """


__all__ = ["BackendConfig"]
