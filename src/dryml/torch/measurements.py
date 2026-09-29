"""Native PyTorch parameter measurements without DRYML lifecycle setup."""

from __future__ import annotations

from dryml.models.measurements import MeasurementUnavailableError, ParameterCounts, parameter_counts_from_parameters


def parameter_sets(model) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Return native Torch total and effective-trainable parameter collections.

    Args:
        model: Native ``torch.nn.Module`` or compatible object exposing
            ``parameters()``.

    Returns:
        Tuples containing all parameters and those with ``requires_grad``.

    Raises:
        MeasurementUnavailableError: If parameter enumeration is unavailable.

    Side Effects:
        Enumerates module metadata only. It does not invoke, initialize, move,
        save, or mutate the model.
    """

    try:
        parameters = tuple(model.parameters())
    except (AttributeError, TypeError, ValueError) as error:
        raise MeasurementUnavailableError("Torch parameters are unavailable.") from error
    return parameters, tuple(parameter for parameter in parameters if parameter.requires_grad)


def parameter_counts(model) -> ParameterCounts:
    """Count native PyTorch scalar parameters by object identity.

    Args:
        model: A native ``torch.nn.Module`` or compatible parameter container.

    Returns:
        Distinct total and effective-trainable scalar parameter counts.

    Raises:
        MeasurementUnavailableError: If a lazy parameter has no concrete shape.

    Side Effects:
        Reads parameter metadata only; no DRYML runtime or model invocation occurs.
    """

    parameters, trainables = parameter_sets(model)
    return parameter_counts_from_parameters(parameters, trainables)


__all__ = ["parameter_counts", "parameter_sets"]
