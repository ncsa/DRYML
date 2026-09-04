"""Stable, redacted failures raised before code analysis begins."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from .facts import FactValue


AnalysisErrorCode: TypeAlias = Literal[
    "target.invalid",
    "target.import_failed",
    "source.unavailable",
    "source.invalid",
    "graph.invalid",
    "kernel.invalid",
    "kernel.dependency",
    "kernel.execution",
    "kernel.output_type",
    "analysis.missing_output",
    "trace.unsupported",
    "trace.hook_active",
    "trace.limit",
    "trace.invocation",
    "trace.cleanup",
]

_CODES = frozenset(
    {
        "target.invalid", "target.import_failed", "source.unavailable", "source.invalid",
        "graph.invalid", "kernel.invalid", "kernel.dependency", "kernel.execution",
        "kernel.output_type", "analysis.missing_output", "trace.unsupported",
        "trace.hook_active", "trace.limit", "trace.invocation", "trace.cleanup",
    }
)


def _is_fact_value(value: object) -> bool:
    """Return whether *value* belongs to the closed diagnostic value grammar."""

    value_type = type(value)
    if value is None or value_type in (bool, int, str, bytes):
        return True
    if value_type is float:
        return math.isfinite(value)
    if value_type is not tuple:
        return False
    if all(type(item) is tuple and len(item) == 2 and type(item[0]) is str for item in value):
        keys = tuple(item[0] for item in value)
        return keys == tuple(sorted(set(keys))) and all(_is_fact_value(item[1]) for item in value)
    return all(_is_fact_value(item) for item in value)


class CodeAnalysisError(RuntimeError):
    """A classified, redacted pre-analysis failure.

    Args:
        message: Fixed framework-authored explanation that contains no target
            representation, exception text, path, or secret-bearing value.
        code: Stable machine-readable failure category.
        context: Optional immutable scalar/tuple evidence using the closed
            :class:`~dryml.code.facts.FactValue` grammar.

    Raises:
        ValueError: If ``code`` or ``context`` is outside the public closed
            error contract.

    Side Effects:
        None. The error retains only the supplied immutable diagnostic context.
    """

    code: AnalysisErrorCode
    context: tuple[tuple[str, FactValue], ...]

    def __init__(
        self,
        message: str,
        *,
        code: AnalysisErrorCode,
        context: Mapping[str, FactValue] | tuple[tuple[str, FactValue], ...] = (),
    ) -> None:
        """Create a validated classified error without formatting caller values."""

        if type(message) is not str or code not in _CODES:
            raise ValueError("code analysis error arguments are invalid")
        if isinstance(context, Mapping):
            items = tuple(sorted(context.items()))
        else:
            items = context
        if type(items) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not _is_fact_value(item[1])
            for item in items
        ):
            raise ValueError("code analysis error context is invalid")
        keys = tuple(item[0] for item in items)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("code analysis error context is invalid")
        super().__init__(message)
        self.code = code
        self.context = items


class InvalidTargetError(CodeAnalysisError):
    """A target cannot be admitted by the closed static target whitelist.

    Args:
        message: Fixed framework-authored failure explanation.
        code: ``"target.invalid"`` or the explicit import-failure category.
        context: Optional closed immutable diagnostic evidence.

    Raises:
        ValueError: If the supplied error contract is invalid.

    Side Effects:
        None.
    """

    def __init__(
        self,
        message: str = "unsupported target",
        *,
        code: AnalysisErrorCode = "target.invalid",
        context: Mapping[str, FactValue] | tuple[tuple[str, FactValue], ...] = (),
    ) -> None:
        """Create a target-resolution failure with a stable category."""

        if code not in ("target.invalid", "target.import_failed"):
            raise ValueError("invalid target error code")
        super().__init__(message, code=code, context=context)


class SourceUnavailableError(CodeAnalysisError):
    """Source is unavailable or malformed before any graph is constructed.

    Args:
        message: Fixed framework-authored failure explanation.
        code: ``"source.unavailable"`` or ``"source.invalid"``.
        context: Optional closed immutable diagnostic evidence.

    Raises:
        ValueError: If the supplied error contract is invalid.

    Side Effects:
        None.
    """

    def __init__(
        self,
        message: str = "source is unavailable",
        *,
        code: AnalysisErrorCode = "source.unavailable",
        context: Mapping[str, FactValue] | tuple[tuple[str, FactValue], ...] = (),
    ) -> None:
        """Create a source failure with a stable category."""

        if code not in ("source.unavailable", "source.invalid"):
            raise ValueError("invalid source error code")
        super().__init__(message, code=code, context=context)


class InvalidKernelError(CodeAnalysisError):
    """A consumer kernel declaration is invalid before execution.

    Args:
        message: Fixed framework-authored failure explanation.
        context: Optional closed immutable diagnostic evidence.

    Raises:
        ValueError: If the supplied error contract is invalid.

    Side Effects:
        None.
    """

    def __init__(self, message: str = "invalid kernel declaration", *, context: Mapping[str, FactValue] | tuple[tuple[str, FactValue], ...] = ()) -> None:
        """Create a stable kernel-declaration error."""

        super().__init__(message, code="kernel.invalid", context=context)


class KernelDependencyError(CodeAnalysisError):
    """A kernel dependency graph cannot be admitted.

    Args:
        message: Fixed framework-authored failure explanation.
        context: Optional closed immutable diagnostic evidence.

    Raises:
        ValueError: If the supplied error contract is invalid.

    Side Effects:
        None.
    """

    def __init__(self, message: str = "invalid kernel dependency", *, context: Mapping[str, FactValue] | tuple[tuple[str, FactValue], ...] = ()) -> None:
        """Create a stable dependency error."""

        super().__init__(message, code="kernel.dependency", context=context)


class KernelExecutionError(CodeAnalysisError):
    """A kernel execution failure crosses a public analysis boundary.

    Args:
        message: Fixed framework-authored failure explanation.
        context: Optional closed immutable diagnostic evidence.

    Raises:
        ValueError: If the supplied error contract is invalid.

    Side Effects:
        None.
    """

    def __init__(self, message: str = "kernel execution failed", *, context: Mapping[str, FactValue] | tuple[tuple[str, FactValue], ...] = ()) -> None:
        """Create a stable execution error."""

        super().__init__(message, code="kernel.execution", context=context)


class MissingOutputError(CodeAnalysisError):
    """A required analysis output is absent or incomplete.

    Args:
        message: Fixed framework-authored failure explanation.
        context: Optional closed immutable diagnostic evidence.

    Raises:
        ValueError: If the supplied error contract is invalid.

    Side Effects:
        None.
    """

    def __init__(self, message: str = "required analysis output is missing", *, context: Mapping[str, FactValue] | tuple[tuple[str, FactValue], ...] = ()) -> None:
        """Create a stable missing-output error."""

        super().__init__(message, code="analysis.missing_output", context=context)


__all__ = [
    "AnalysisErrorCode",
    "CodeAnalysisError",
    "InvalidKernelError",
    "InvalidTargetError",
    "KernelDependencyError",
    "KernelExecutionError",
    "MissingOutputError",
    "SourceUnavailableError",
]
