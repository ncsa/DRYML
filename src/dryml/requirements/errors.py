"""Bounded, redacted failures for shared requirement contracts."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .barrier import AdmissionReport
    from .model import RequirementReport

_ASSIGNMENT = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|credential)\s*=\s*[^\s,]+")
_ABSOLUTE_PATH = re.compile(r"(?<![\w:])(?:/[^\s,?#]+|[A-Za-z]:[\\/][^\s,?#]+|\\\\[^\s,?#]+)")
_URI = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*):\/\/([^\s/?#@]+@)?([^\s/?#]+)(?:\/[^\s?#]*)?(?:\?[^\s#]*)?(?:#[^\s]*)?")
_FILE_URI = re.compile(r"(?i)\bfile:\/\/[^\s,]+")
_SECRET_KEY = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|credential)")
_MAX_TEXT = 512
_MAX_DEPTH = 8
_MAX_ENTRIES = 256
_MAX_NODES = 1024
_MAX_INT_BITS = 4096


def _project_text(value: object, *, limit: int = _MAX_TEXT) -> str:
    """Return one bounded text diagnostic without formatting unknown values."""

    if type(value) is not str:
        return "<unsupported>"
    text = _ASSIGNMENT.sub(r"\1=<redacted>", value)
    text = _FILE_URI.sub("file://<redacted>", text)
    text = _URI.sub(_redact_uri, text)
    text = _ABSOLUTE_PATH.sub("<local-path>", text)
    return text[:limit]


def _redact_uri(match: re.Match[str]) -> str:
    """Project a URI without credentials, paths, queries, or fragments."""

    scheme, userinfo, host = match.group(1), match.group(2), match.group(3)
    if scheme.lower() == "file":
        return "file://<redacted>"
    prefix = "<redacted>@" if userinfo else ""
    return f"{scheme}://{prefix}{host}"


def _project_value(value: Any, *, strict: bool = False) -> Any:
    """Freeze one bounded built-in diagnostic value without user callbacks."""

    state = [0]

    def project(item: Any, depth: int, key: str | None = None) -> Any:
        state[0] += 1
        if strict and (depth > _MAX_DEPTH or state[0] > _MAX_NODES):
            raise ValueError("diagnostic context exceeds its structural bound")
        if key is not None and _SECRET_KEY.search(key):
            return "<redacted>"
        if item is None or type(item) in (bool, float):
            return item
        if type(item) is int:
            if item.bit_length() > _MAX_INT_BITS:
                if strict:
                    raise ValueError("diagnostic integer exceeds its bit bound")
                return "<unsupported>"
            return item
        if type(item) is str:
            return _project_text(item)
        if type(item) in (list, tuple):
            if strict and len(item) > _MAX_ENTRIES:
                raise ValueError("diagnostic sequence exceeds its entry bound")
            return tuple(project(child, depth + 1) for child in item[:_MAX_ENTRIES])
        if type(item) in (dict, MappingProxyType):
            if strict and len(item) > _MAX_ENTRIES:
                raise ValueError("diagnostic mapping exceeds its entry bound")
            projected: dict[str, Any] = {}
            for index, (name, child) in enumerate(tuple(item.items())[:_MAX_ENTRIES], start=1):
                name_text = _project_text(name) if type(name) is str else f"<key-{index}>"
                projected[name_text] = project(child, depth + 1, name_text)
            return MappingProxyType(projected)
        return "<unsupported>"

    return project(value, 0)


class RequirementError(RuntimeError):
    """Raised when shared requirement input or protocol data is invalid.

    Args:
        message: A human-readable diagnostic projected to avoid secrets and local
            paths.
        context: Optional built-in mapping of bounded diagnostic details.

    Raises:
        RequirementError: If context exceeds the safe diagnostic structure bound.

    Side Effects:
        None. Constructing the error does not inspect targets or alter runtime,
        session, or global state.
    """

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        if context is not None and type(context) not in (dict, MappingProxyType):
            raise RequirementError("invalid requirement error context")
        try:
            projected = MappingProxyType({}) if context is None else _project_value(context, strict=True)
        except ValueError:
            raise RequirementError("invalid requirement error context") from None
        super().__init__(_project_text(message))
        self.context = projected


class RequirementCombinationError(RequirementError):
    """Raised when declaration orchestration cannot produce a valid result.

    Args:
        message: A fixed, safely projected explanation of the orchestration
            failure.
        report: The report associated with a validated unsuccessful combination.

    Side Effects:
        None. No partial domain requirement is exposed.
    """

    def __init__(self, message: str, *, report: "RequirementReport") -> None:
        super().__init__(message)
        self.report = report


class RequirementBarrierError(RequirementError):
    """Raised when an explicit admission report denies a hard requirement.

    Args:
        message: A fixed, safely projected admission failure explanation.
        report: The exact domain report whose admission decision was false.
        operation: Optional bounded operation label retained for diagnostics.

    Side Effects:
        None. The barrier does not invoke protected work or mutate ambient state.
    """

    def __init__(self, message: str, *, report: "AdmissionReport", operation: str | None = None) -> None:
        super().__init__(message)
        self.report = report
        if operation is not None and (
            type(operation) is not str
            or len(operation) > _MAX_TEXT
            or any(ord(char) < 32 or ord(char) == 127 for char in operation)
        ):
            raise RequirementError("invalid admission operation")
        self.operation = None if operation is None else _project_text(operation)


__all__ = ["RequirementBarrierError", "RequirementCombinationError", "RequirementError"]
