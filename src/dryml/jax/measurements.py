"""Native JAX parameter measurement without model execution or initialization."""

from __future__ import annotations

from dryml.models.measurements import ParameterCounts, parameter_counts_from_parameters


def parameter_sets(parameters, trainable_parameters=None) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Flatten declared JAX parameter trees without including auxiliary state.

    Args:
        parameters: Pytree containing model parameter arrays only.
        trainable_parameters: Optional effective trainable parameter pytree. When
            omitted, all declared parameter leaves are trainable.

    Returns:
        Flattened total and effective-trainable parameter collections.

    Side Effects:
        Reads pytree metadata only. It does not invoke a model or allocate inputs.
    """

    import jax

    total = tuple(jax.tree_util.tree_leaves(parameters))
    trainable = total if trainable_parameters is None else tuple(jax.tree_util.tree_leaves(trainable_parameters))
    return total, trainable


def parameter_counts(parameters, trainable_parameters=None) -> ParameterCounts:
    """Count scalar JAX model parameters while excluding buffers, slots, and RNG.

    Args:
        parameters: Pytree containing model parameter arrays only.
        trainable_parameters: Optional effective trainable parameter pytree.

    Returns:
        Identity-deduplicated total and trainable scalar counts.

    Side Effects:
        Reads declared parameter-tree metadata only; no model is invoked.
    """

    return parameter_counts_from_parameters(*parameter_sets(parameters, trainable_parameters))


__all__ = ["parameter_counts", "parameter_sets"]
