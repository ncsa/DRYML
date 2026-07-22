"""Structured progress reporting for DRYML orchestration boundaries."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready


ReportLevel = Literal["quiet", "steps", "details", "debug"]
EventLevel = Literal["step", "detail", "debug"]
ReportStream = Literal["stdout", "stderr", "none", "logging"]
ReportFormat = Literal["text", "json"]

_REPORT_LEVELS = {"quiet", "steps", "details", "debug"}
_EVENT_LEVELS = {"step", "detail", "debug"}
_STREAMS = {"stdout", "stderr", "none", "logging"}
_FORMATS = {"text", "json"}
_EVENT_RANK = {"step": 1, "detail": 2, "debug": 3}
_REPORT_RANK = {"quiet": 0, "steps": 1, "details": 2, "debug": 3}


@dataclass(frozen=True, slots=True)
class DrymlEvent:
    """One structured DRYML progress event."""

    name: str
    message: str
    level: EventLevel
    phase: str | None = None
    operation_id: str | None = None
    environment_id: str | None = None
    world_id: str | None = None
    runtime_id: str | None = None
    record_id: str | None = None
    provider_id: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("reporting event name must be a non-empty string")
        if not isinstance(self.message, str):
            raise ValueError("reporting event message must be a string")
        if self.level not in _EVENT_LEVELS:
            raise ValueError(f"reporting event level must be one of {sorted(_EVENT_LEVELS)}")
        object.__setattr__(self, "data", _freeze_json_mapping(self.data, "data"))

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready event data."""

        return {
            "name": self.name,
            "message": self.message,
            "level": self.level,
            "phase": self.phase,
            "operation_id": self.operation_id,
            "environment_id": self.environment_id,
            "world_id": self.world_id,
            "runtime_id": self.runtime_id,
            "record_id": self.record_id,
            "provider_id": self.provider_id,
            "data": json_ready(self.data),
            "timestamp": self.timestamp,
        }


class Reporter:
    """Base reporter interface for DRYML progress events."""

    def emit(self, event: DrymlEvent, config: "ReportingConfig") -> None:
        """Handle one event."""

        raise NotImplementedError


class NullReporter(Reporter):
    """Reporter that discards all events."""

    def emit(self, event: DrymlEvent, config: "ReportingConfig") -> None:
        return None


class StdoutReporter(Reporter):
    """Reporter that renders events to stdout or stderr."""

    def emit(self, event: DrymlEvent, config: "ReportingConfig") -> None:
        stream = sys.stderr if config.stream == "stderr" else sys.stdout
        print(_format_event(event, config), file=stream)


class LoggingReporter(Reporter):
    """Reporter that forwards events to Python logging."""

    def __init__(self, logger_name: str = "dryml.reporting") -> None:
        self.logger = logging.getLogger(logger_name)
        if not any(isinstance(handler, logging.NullHandler) for handler in self.logger.handlers):
            self.logger.addHandler(logging.NullHandler())

    def emit(self, event: DrymlEvent, config: "ReportingConfig") -> None:
        level = logging.DEBUG if event.level == "debug" else logging.INFO
        self.logger.log(level, _format_event(event, config))


class CaptureReporter(Reporter):
    """Reporter useful for tests and notebooks that need captured events."""

    def __init__(self) -> None:
        self.events: list[DrymlEvent] = []

    def emit(self, event: DrymlEvent, config: "ReportingConfig") -> None:
        self.events.append(event)

    def clear(self) -> None:
        """Remove all captured events."""

        self.events.clear()


@dataclass(frozen=True, slots=True)
class ReportingConfig:
    """Process-local DRYML progress reporting configuration."""

    level: ReportLevel = "quiet"
    stream: ReportStream = "stdout"
    format: ReportFormat = "text"
    include_ids: bool = True
    include_timing: bool = True
    strict: bool = False
    reporter: Reporter | None = None

    def __post_init__(self) -> None:
        level = _normalize_token(self.level, "level", _REPORT_LEVELS)
        stream = _normalize_token(self.stream, "stream", _STREAMS)
        fmt = _normalize_token(self.format, "format", _FORMATS)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "stream", stream)
        object.__setattr__(self, "format", fmt)
        object.__setattr__(self, "include_ids", _coerce_bool(self.include_ids, "include_ids"))
        object.__setattr__(self, "include_timing", _coerce_bool(self.include_timing, "include_timing"))
        object.__setattr__(self, "strict", _coerce_bool(self.strict, "strict"))

    @classmethod
    def from_value(cls, value: Any = None, *, base: "ReportingConfig | None" = None) -> "ReportingConfig":
        """Coerce strings, mappings, reporters, and configs into a config."""

        if value is None:
            return base or cls()
        if isinstance(value, ReportingConfig):
            return value
        if isinstance(value, Reporter):
            current = base or cls(level="debug")
            level = "debug" if current.level == "quiet" else current.level
            return cls(current.level if current.level != "quiet" else level, current.stream, current.format, current.include_ids, current.include_timing, current.strict, value)
        if isinstance(value, str):
            current = base or cls()
            return cls(value, current.stream, current.format, current.include_ids, current.include_timing, current.strict, current.reporter)
        if isinstance(value, Mapping):
            current = base or cls()
            unknown = set(value) - {"level", "stream", "format", "include_ids", "include_timing", "strict", "reporter"}
            if unknown:
                raise ValueError(f"unknown reporting config fields: {sorted(unknown)}")
            return cls(
                level=value.get("level", current.level),
                stream=value.get("stream", current.stream),
                format=value.get("format", current.format),
                include_ids=value.get("include_ids", current.include_ids),
                include_timing=value.get("include_timing", current.include_timing),
                strict=value.get("strict", current.strict),
                reporter=value.get("reporter", current.reporter),
            )
        raise ValueError(f"unsupported reporting config value {value!r}")

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready public config data."""

        return {
            "level": self.level,
            "stream": self.stream,
            "format": self.format,
            "include_ids": self.include_ids,
            "include_timing": self.include_timing,
            "strict": self.strict,
            "reporter": None if self.reporter is None else type(self.reporter).__name__,
        }


def config_from_env() -> ReportingConfig:
    """Build the default reporting config from environment variables."""

    return ReportingConfig(
        level=os.environ.get("DRYML_REPORT", "quiet"),
        stream=os.environ.get("DRYML_REPORT_STREAM", "stdout"),
        format=os.environ.get("DRYML_REPORT_FORMAT", "text"),
    )


def emit(name: str, message: str, *, level: EventLevel = "step", **fields: Any) -> DrymlEvent | None:
    """Emit one structured progress event if enabled."""

    cfg = _current_reporting_config()
    if not _enabled(cfg, level):
        return None
    try:
        data = fields.pop("data", {}) or {}
        if fields:
            known = {"phase", "operation_id", "environment_id", "world_id", "runtime_id", "record_id", "provider_id"}
            event_fields = {key: fields.pop(key) for key in tuple(fields) if key in known}
            data = {**fields, **dict(data)}
        else:
            event_fields = {}
        timestamp = time.time() if cfg.include_timing else None
        event = DrymlEvent(name=name, message=message, level=level, data=data, timestamp=timestamp, **event_fields)
        _reporter_for(cfg).emit(event, cfg)
        return event
    except Exception:
        if cfg.strict:
            raise
        logging.getLogger("dryml.reporting").debug("reporting event dropped", exc_info=True)
        return None


def step(name: str, message: str, **fields: Any) -> DrymlEvent | None:
    """Emit a lifecycle step event."""

    return emit(name, message, level="step", **fields)


def detail(name: str, message: str, **fields: Any) -> DrymlEvent | None:
    """Emit a compact detail event."""

    return emit(name, message, level="detail", **fields)


def debug(name: str, message: str, **fields: Any) -> DrymlEvent | None:
    """Emit a verbose debug event."""

    return emit(name, message, level="debug", **fields)


def _current_reporting_config() -> ReportingConfig:
    try:
        from dryml.core.session import get_config

        return get_config().reporting
    except Exception:
        return config_from_env()


def _enabled(config: ReportingConfig, level: EventLevel) -> bool:
    return _REPORT_RANK[config.level] >= _EVENT_RANK[level]


def _reporter_for(config: ReportingConfig) -> Reporter:
    if config.reporter is not None:
        return config.reporter
    if config.level == "quiet" or config.stream == "none":
        return NullReporter()
    if config.stream == "logging":
        return _logging_reporter()
    return StdoutReporter()


def _format_event(event: DrymlEvent, config: ReportingConfig) -> str:
    data = event.to_data()
    if not config.include_ids:
        for key in ("operation_id", "environment_id", "world_id", "runtime_id", "record_id", "provider_id"):
            data.pop(key, None)
    if not config.include_timing:
        data.pop("timestamp", None)
    if config.format == "json":
        return json.dumps(data, sort_keys=True, separators=(",", ":"))
    lines = [f"DRYML: {event.message}"]
    if config.level in {"details", "debug"}:
        details = dict(data.get("data") or {})
        for key in ("operation_id", "environment_id", "world_id", "runtime_id", "record_id", "provider_id"):
            if data.get(key) is not None:
                details.setdefault(key, data[key])
        for key in sorted(details):
            lines.append(f"  {key}: {details[key]}")
    return "\n".join(lines)


def _normalize_token(value: Any, field_name: str, allowed: set[str]) -> str:
    token = str(value).strip().lower().replace("-", "_")
    aliases = {"step": "steps", "detail": "details", "verbose": "details", "none": "quiet"} if field_name == "level" else {}
    token = aliases.get(token, token)
    if token not in allowed:
        raise ValueError(f"reporting {field_name} must be one of {sorted(allowed)}, got {value!r}")
    return token


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"reporting {field_name} must be a boolean, got {value!r}")


_LOGGING_REPORTER: LoggingReporter | None = None


def _logging_reporter() -> LoggingReporter:
    global _LOGGING_REPORTER
    if _LOGGING_REPORTER is None:
        _LOGGING_REPORTER = LoggingReporter()
    return _LOGGING_REPORTER


def _freeze_json_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"reporting {path} must be a mapping")
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise ValueError(f"reporting {path} is not JSON-ready: {exc}") from exc
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = [
    "CaptureReporter",
    "DrymlEvent",
    "EventLevel",
    "LoggingReporter",
    "NullReporter",
    "ReportFormat",
    "ReportLevel",
    "ReportStream",
    "Reporter",
    "ReportingConfig",
    "StdoutReporter",
    "config_from_env",
    "debug",
    "detail",
    "emit",
    "step",
]
