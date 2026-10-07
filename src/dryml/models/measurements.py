"""Dependency-light model and training measurement values.

This module deliberately does not import optional machine-learning frameworks.
Backend modules provide native parameter discovery while this module owns shared
identity deduplication, availability errors, and Dataset cardinality reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable

from dryml.core.cardinality import Cardinality


class MeasurementUnavailableError(RuntimeError):
    """Raise when a truthful measurement cannot be obtained without execution.

    This error distinguishes an unbuilt, lazy, or unknown-shape model from a
    built model that legitimately has zero parameters. Callers must retain the
    measurement as unavailable rather than substituting zero.
    """


@dataclass(frozen=True, slots=True)
class ParameterCounts:
    """Scalar parameter counts for one model graph.

    Args:
        total: Nonnegative number of distinct native parameter scalars.
        trainable: Nonnegative number of distinct scalars exposed by at least
            one effective trainable path.

    Raises:
        ValueError: If either count is invalid or trainable exceeds total.

    ``total`` excludes optimizer slots and backend buffers. Shared native
    parameter objects count once by identity; frozen objects contribute only to
    ``total`` unless another path exposes the same object as trainable.
    """

    total: int
    trainable: int

    def __post_init__(self) -> None:
        if type(self.total) is not int or type(self.trainable) is not int:
            raise TypeError("Parameter counts must be exact integers.")
        if self.total < 0 or self.trainable < 0 or self.trainable > self.total:
            raise ValueError("Parameter counts must be nonnegative with trainable <= total.")


@dataclass(frozen=True, slots=True)
class TrainingObservation:
    """Immutable retained facts staged at a training safe point.

    Args:
        examples_seen: Successful optimizer-update exposure represented by the
            checkpoint.
        loss_numerator: Sum of per-example loss contributions since the prior
            observation.
        loss_denominator: Number of contributing examples in that loss window.
        epoch: Next logical epoch position at the safe point.
        next_batch: Next unprocessed batch within ``epoch``.
        sequence: Monotonic safe-point sequence for the current training state.

    The observation is pickle-portable state only; U9 owns association and
    publication of observations to ExperimentData.
    """

    examples_seen: int
    loss_numerator: float
    loss_denominator: int
    epoch: int
    next_batch: int
    sequence: int

    @property
    def training_loss(self) -> float | None:
        """Return the weighted mean loss for this observation window.

        Returns:
            ``None`` when no successful update contributed to the window, else
            the numerator divided by the contributing-example denominator.
        """

        if self.loss_denominator == 0:
            return None
        return self.loss_numerator / self.loss_denominator


def parameter_counts_from_parameters(
    parameters: Iterable[object], trainable_parameters: Iterable[object],
) -> ParameterCounts:
    """Count distinct scalar parameters from total and effective-trainable paths.

    Args:
        parameters: Native parameter objects exposed by all model paths.
        trainable_parameters: Native parameter objects exposed by effective
            trainable paths. Objects may overlap ``parameters`` or one another.

    Returns:
        Identity-deduplicated total and trainable scalar counts.

    Raises:
        MeasurementUnavailableError: If a parameter has an unavailable or
            unknown shape.

    Side Effects:
        None. The function neither invokes a model nor initializes a backend.
    """

    total_by_id = {id(parameter): parameter for parameter in parameters}
    trainable_by_id = {id(parameter): parameter for parameter in trainable_parameters}
    total_by_id.update(trainable_by_id)
    return ParameterCounts(
        total=sum(_parameter_size(parameter) for parameter in total_by_id.values()),
        trainable=sum(_parameter_size(parameter) for parameter in trainable_by_id.values()),
    )


def dataset_size(dataset) -> Cardinality:
    """Return a Dataset's declared example cardinality without opening a cursor.

    Args:
        dataset: Final canonical Dataset persisted by an Experiment.

    Returns:
        A finite, unknown, or infinite :class:`Cardinality`.

    Side Effects:
        Calls only declared cardinality accessors. It never iterates or consumes
        data. It delegates wholly to the Dataset's public example contract.
    """

    cardinality = getattr(dataset, "example_cardinality", None)
    if cardinality is None:
        return Cardinality.UNKNOWN
    return cardinality()


def model_parameter_counts(model, *, repo=None) -> ParameterCounts:
    """Measure a native-backed DRYML model or composite model graph.

    Args:
        model: A DRYML backend model wrapper or composite containing one backend.
        repo: Optional Repo authority for composite traversal. A temporary Repo
            is used only while traversing when omitted.

    Returns:
        Distinct native scalar parameter counts for the model graph.

    Raises:
        MeasurementUnavailableError: If a native model is unbuilt, lazy, or has
            an unknown parameter shape.
        TypeError: If no supported native backend is present or multiple backend
            types occur in the graph.

    Side Effects:
        Composite traversal temporarily uses Repo graph authority but never
        invokes, builds, saves, or changes a model/runtime.
    """

    backend = getattr(model, "native_backend", None)
    if backend in {"tf", "torch"}:
        _, parameters, trainables = _model_parameter_set(model)
        return parameter_counts_from_parameters(parameters, trainables)
    if backend is not None:
        raise TypeError(f"Unsupported native measurement backend {backend!r}.")

    if repo is None:
        from dryml.core.repo import manage_repo

        with manage_repo() as temporary_repo:
            return model_parameter_counts(model, repo=temporary_repo)

    parameter_sets = []
    backends = set()
    for result in repo.apply_graph(
        model,
        lambda obj: _model_parameter_set(obj),
        missing="raise",
        order="post",
    ).values():
        if result is not None:
            backend, parameters, trainables = result
            backends.add(backend)
            parameter_sets.append((parameters, trainables))
    if len(backends) != 1:
        raise TypeError("Model measurement requires exactly one supported native backend.")
    parameters = tuple(parameter for values, _ in parameter_sets for parameter in values)
    trainables = tuple(parameter for _, values in parameter_sets for parameter in values)
    return parameter_counts_from_parameters(parameters, trainables)


def _model_parameter_set(model):
    backend = getattr(model, "native_backend", None)
    if backend not in {"tf", "torch"}:
        return None
    parameters, trainables = _backend_parameter_set(backend, getattr(model, "obj", model))
    effective = getattr(model, "trainable_parameters", None)
    if callable(effective):
        try:
            trainables = tuple(
                parameter
                for parameter in effective(backend)
                if _parameter_is_trainable(parameter)
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise MeasurementUnavailableError("Effective trainable parameters are unavailable.") from error
    return backend, parameters, trainables


def _backend_parameter_set(backend: str, model) -> tuple[tuple[object, ...], tuple[object, ...]]:
    if backend == "tf":
        from dryml.tf.measurements import parameter_sets
    elif backend == "torch":
        from dryml.torch.measurements import parameter_sets
    else:
        raise TypeError(f"Unsupported native measurement backend {backend!r}.")
    return parameter_sets(model)


def _parameter_size(parameter: object) -> int:
    try:
        shape = parameter.shape
        dimensions = shape.as_list() if hasattr(shape, "as_list") else tuple(shape)
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise MeasurementUnavailableError("Parameter shape is unavailable.") from error
    if dimensions is None or any(dimension is None for dimension in dimensions):
        raise MeasurementUnavailableError("Parameter shape is unavailable.")
    try:
        sizes = tuple(int(dimension) for dimension in dimensions)
    except (TypeError, ValueError) as error:
        raise MeasurementUnavailableError("Parameter shape is unavailable.") from error
    if any(size < 0 for size in sizes):
        raise MeasurementUnavailableError("Parameter shape is unavailable.")
    return prod(sizes)


def _parameter_is_trainable(parameter: object) -> bool:
    """Honor backend-native trainability while retaining wrapper path policy."""

    if hasattr(parameter, "requires_grad"):
        return bool(parameter.requires_grad)
    return bool(getattr(parameter, "trainable", True))


__all__ = [
    "MeasurementUnavailableError",
    "ParameterCounts",
    "TrainingObservation",
    "dataset_size",
    "model_parameter_counts",
    "parameter_counts_from_parameters",
]
