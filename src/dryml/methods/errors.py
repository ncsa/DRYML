"""Bounded diagnostics for Method declaration and invocation contracts."""

from __future__ import annotations

from typing import Literal

SelectionFailureReason = Literal["no_candidate", "ambiguous", "unknown_traits", "conflict"]
SelectionTraitName = Literal["backend", "batch_mode"]


class MethodError(Exception):
    """Base error for bounded Method contract failures.

    Method APIs raise this family rather than leaking implementation-specific
    lookup or declaration exceptions to callers.
    """


class ImplementationDeclarationError(MethodError):
    """Report malformed or ambiguous authored implementation evidence.

    This error is raised while a Method class is declared or its static catalog
    is inspected, before an authored target is bound or invoked.
    """


class ImplementationSelectionError(MethodError):
    """Report a bounded implementation-selection failure.

    Args:
        reason: The machine-readable reason that selection could not produce one
            implementation.
        unknown_traits: Required trait dimensions that remain unknown. This is
            non-empty only when ``reason`` is ``"unknown_traits"``.
        message: Optional bounded diagnostic text.

    ``unknown_traits`` is populated only for the ``"unknown_traits"`` reason.
    """

    def __init__(
        self,
        reason: SelectionFailureReason,
        unknown_traits: tuple[SelectionTraitName, ...] = (),
        message: str | None = None,
    ) -> None:
        """Initialize the typed selection diagnostic without inspecting runtime values."""

        self.reason = reason
        self.unknown_traits = unknown_traits
        super().__init__(message or f"Method implementation selection failed: {reason}.")


class PreparedCallMismatchError(MethodError):
    """Report an exact cached-call signature mismatch.

    Args:
        expected: The retained normalized call signature.
        observed: The normalized signature supplied by the mismatching call.

    The expected and observed payloads are immutable Method call signatures when
    normalization succeeded. A malformed cached runtime call may report the
    retained signature as both payloads while still failing before user code.
    """

    def __init__(self, expected: object, observed: object) -> None:
        """Retain typed immutable diagnostic payloads for cached-call handling."""

        self.expected = expected
        self.observed = observed
        super().__init__("Prepared Method call does not match the cached signature.")


__all__ = [
    "MethodError",
    "ImplementationDeclarationError",
    "ImplementationSelectionError",
    "PreparedCallMismatchError",
    "SelectionFailureReason",
    "SelectionTraitName",
]
