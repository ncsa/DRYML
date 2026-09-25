"""Inert named numerical Fold factories."""

from __future__ import annotations

from dryml.core import AutoRef, Ref, function
from dryml.data.reduction_methods import (
    MeanFinalize,
    MeanInitial,
    MeanUpdate,
    ReductionMode,
    ReservoirInitial,
    ReservoirQuantile,
    ReservoirUpdate,
)

from .fold import Fold


@function
def mean(src: Ref[AutoRef], *, mode: ReductionMode) -> Fold:
    """Declare an uncomputed exact streaming mean Fold.

    Args:
        src: Non-materializing Dataset reference accepted by :class:`Fold`.
        mode: ``"global"`` for all scalar coordinates or ``"coordinate"`` for
            independent non-batch coordinates.

    Returns:
        An inert Fold whose native sum/count carry is allocated only by compute.

    Raises:
        ValueError: If ``mode`` is not a supported population definition.

    Side Effects:
        None. The source is not loaded and no Method is selected or invoked.
    """

    return Fold(src, initial_state=MeanInitial(mode=mode), accumulator=MeanUpdate(mode=mode), finalize=MeanFinalize())


@function
def quantile(
    src: Ref[AutoRef],
    q: float | tuple[float, ...],
    *,
    mode: ReductionMode,
    capacity: int,
    seed: int = 0,
) -> Fold:
    """Declare an uncomputed bounded Algorithm R quantile Fold.

    Args:
        src: Non-materializing Dataset reference accepted by :class:`Fold`.
        q: One finite request or ordered non-empty tuple of requests in ``[0, 1]``.
        mode: ``"global"`` or ``"coordinate"`` population definition.
        capacity: Positive exact maximum retained rows per reduction domain.
        seed: Exact nonnegative 63-bit deterministic reservoir seed.

    Returns:
        An inert Fold with fixed-capacity native reservoir state. Quantiles are
        exact while the population fits ``capacity`` and sample estimates after.

    Raises:
        TypeError: If request types are invalid.
        ValueError: If q, mode, capacity, or seed is outside the supported contract.

    Side Effects:
        None. Construction does not read the source or consume random draws.
    """

    return Fold(
        src,
        initial_state=ReservoirInitial(mode=mode, capacity=capacity, seed=seed),
        accumulator=ReservoirUpdate(mode=mode, capacity=capacity),
        finalize=ReservoirQuantile(q),
    )


__all__ = ["mean", "quantile"]
