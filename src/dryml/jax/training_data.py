"""JAX-native batch iteration for prepared DRYML Dataset pipelines."""

from __future__ import annotations

from dryml.data.native import PreparedDataset


def iter_training_batches(data: PreparedDataset, preparation, *, epoch: int = 0):
    """Return one closeable iterator yielding JAX-native prepared ``(x, y)`` batches.

    Args:
        data: Dependency-light prepared Dataset pipeline.
        preparation: Retained x/y handoff prepared for JAX.
        epoch: Logical Dataset epoch used for a supported seed-aware source.

    Returns:
        A closeable cursor whose yielded trees retain authored structure, dtype,
        shape, batch axis, and order.

    Raises:
        TypeError: If ``data`` is not a :class:`PreparedDataset`.
        DatasetExhaustedError: During iteration if a finite source ends early.

    Side Effects:
        Opens no source until advancement. The selected preparation may import JAX
        to execute its retained conversion edge, but cursor reads never update
        training progress, exposure, or checkpoint state.
    """

    if not isinstance(data, PreparedDataset):
        raise TypeError("JAX training input requires a PreparedDataset.")
    return data.training_batches(preparation, epoch=epoch)


__all__ = ["iter_training_batches"]
