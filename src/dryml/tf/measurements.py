"""Native TensorFlow/Keras parameter measurements without DRYML lifecycle setup."""

from __future__ import annotations

from dryml.models.measurements import MeasurementUnavailableError, ParameterCounts, parameter_counts_from_parameters


def parameter_sets(model) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Return native Keras total and effective-trainable parameter collections.

    Args:
        model: Built native ``tf.keras.Model`` or compatible object exposing
            ``variables`` and ``trainable_variables``.

    Returns:
        Tuples of all variables and effective trainable variables.

    Raises:
        MeasurementUnavailableError: If Keras reports the model is unbuilt.

    Side Effects:
        Reads native metadata only. It does not call, build, compile, save, or
        otherwise mutate the model.
    """

    if getattr(model, "built", None) is False:
        raise MeasurementUnavailableError("TensorFlow model is unbuilt.")
    try:
        return tuple(model.variables), tuple(model.trainable_variables)
    except (AttributeError, TypeError, ValueError) as error:
        raise MeasurementUnavailableError("TensorFlow parameters are unavailable.") from error


def parameter_counts(model) -> ParameterCounts:
    """Count native TensorFlow/Keras scalar parameters by object identity.

    Args:
        model: A built native ``tf.keras.Model`` or compatible Keras object.

    Returns:
        Distinct total and effective-trainable scalar parameter counts.

    Raises:
        MeasurementUnavailableError: If the model is unbuilt or has unknown
            parameter shapes.

    Side Effects:
        Reads model metadata only; no DRYML runtime or model invocation occurs.
    """

    parameters, trainables = parameter_sets(model)
    return parameter_counts_from_parameters(parameters, trainables)


__all__ = ["parameter_counts", "parameter_sets"]
