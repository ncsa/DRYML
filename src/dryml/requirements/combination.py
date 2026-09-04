"""Generic bounded orchestration for domain-owned requirement combination."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar

from .errors import RequirementCombinationError
from .model import RequirementDeclaration, RequirementReport, RequirementResult, _ordinalized_declaration

R = TypeVar("R")
_MAX_DECLARATIONS = 256


class RequirementCombiner(Protocol[R]):
    """Domain-owned semantic combination behavior used by the shared orchestrator.

    Methods:
        combine: Receives ordered, ordinalized exact declarations and returns a
            legal shared result without invoking unrelated domain policy.
    """

    def combine(self, declarations: tuple[RequirementDeclaration[R], ...]) -> RequirementResult[R]:
        """Combine one nonempty ordered declaration tuple into a shared outcome."""


def combine_requirements(
    declarations: Iterable[RequirementDeclaration[R]], *, combiner: RequirementCombiner[R]
) -> RequirementResult[R]:
    """Snapshot, validate, and delegate one bounded domain declaration sequence.

    Args:
        declarations: An ordered iterable of at most 256 exact declarations.
        combiner: A domain-owned object with a callable ``combine`` method.

    Returns:
        An empty success without invoking ``combiner`` when no declarations are
        supplied, otherwise the validated domain result.

    Raises:
        RequirementCombinationError: If input, combiner behavior, capacities, or
            the returned result violate the shared contract.

    Side Effects:
        Consumes ``declarations`` at most once. It does not mutate targets,
        runtime, session, or global state.
    """

    try:
        combine = combiner.combine
    except Exception:
        raise RequirementCombinationError("invalid requirement combiner", report=RequirementReport()) from None
    if not callable(combine):
        raise RequirementCombinationError("invalid requirement combiner", report=RequirementReport())
    try:
        iterator = iter(declarations)
    except Exception:
        raise RequirementCombinationError("invalid requirement declarations", report=RequirementReport()) from None
    values: list[RequirementDeclaration[R]] = []
    try:
        for _ in range(_MAX_DECLARATIONS + 1):
            try:
                declaration = next(iterator)
            except StopIteration:
                break
            if type(declaration) is not RequirementDeclaration:
                raise RequirementCombinationError("invalid requirement declaration", report=RequirementReport())
            values.append(declaration)
    except RequirementCombinationError:
        raise
    except Exception:
        raise RequirementCombinationError("invalid requirement declarations", report=RequirementReport()) from None
    if len(values) > _MAX_DECLARATIONS:
        raise RequirementCombinationError("requirement declaration limit exceeded", report=RequirementReport())
    if not values:
        return RequirementResult()
    try:
        ordinalized = tuple(_ordinalized_declaration(value, ordinal) for ordinal, value in enumerate(values, start=1))
    except Exception:
        raise RequirementCombinationError("invalid requirement declarations", report=RequirementReport()) from None
    try:
        result = combine(ordinalized)
    except Exception:
        raise RequirementCombinationError("requirement combination failed", report=RequirementReport()) from None
    if type(result) is not RequirementResult or type(result.report) is not RequirementReport:
        raise RequirementCombinationError("invalid requirement combination result", report=RequirementReport())
    if result.value is None and result.report.ok:
        raise RequirementCombinationError("nonempty requirement combination has no value", report=result.report)
    return result


__all__ = ["RequirementCombiner", "combine_requirements"]
