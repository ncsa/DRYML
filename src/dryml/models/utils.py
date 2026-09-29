from __future__ import annotations

from dataclasses import dataclass
from dryml.context import check_context, NoContextError, \
    WrongContextError, ContextIncompatibilityError
from typing import Any

from dryml.core.tensor_spec import iter_specs
from dryml.core.utils.recurse import iter_leaves
from dryml.data import Shuffle, Take, Unbatch
from dryml.models.train_spec import TrainState

expected_context_errors = (NoContextError, WrongContextError, ContextIncompatibilityError)


@dataclass(frozen=True, slots=True)
class TrainingPreparation:
    """Once-planned declared x/y handoffs for one native training consumer.

    The carrier executes the selected edges retained in its owning trainer's
    Method graph. Runtime preparation never selects or plans a route.
    """

    producer_specs: tuple[object, object]
    consumer_specs: tuple[object, object]
    conversion_edges: tuple[object | None, object | None]

    @classmethod
    def from_specs(cls, owner, x_spec, y_spec, backend: str) -> "TrainingPreparation":
        """Plan supported direct handoffs from declared x/y specs once.

        Args:
            owner: TrainFunction Method that owns the inspectable graph facts.
            x_spec: Complete declared feature tensor-spec tree.
            y_spec: Complete declared target tensor-spec tree.
            backend: Native consumer backend for both values.

        Returns:
            An immutable carrier retaining producer/consumer facts and direct
            conversion edges, if either declared source differs from ``backend``.

        Raises:
            TypeError: If a source tree has no one complete supported backend or
                lacks a direct dense handoff to ``backend``.

        Side Effects:
            Selects no runtime values and imports no optional backend; it only
            records conversion planning facts on ``owner.method_graph()``.
        """

        from dryml.core.backend import Backend
        from dryml.methods.conversion import make_edge

        target = Backend(backend)

        def plan(spec):
            source_backends = {tensor_spec.backend for tensor_spec in iter_specs(spec)}
            if len(source_backends) != 1 or None in source_backends:
                raise TypeError("Training values require one complete declared backend.")
            source = next(iter(source_backends))
            return None if source is target else make_edge(spec, target)

        edges = (plan(x_spec), plan(y_spec))
        try:
            owner._retain_preparation_graph((x_spec, y_spec), edges)
        except AttributeError as error:
            raise TypeError("Training preparation requires a Method graph owner.") from error
        return cls(
            producer_specs=(x_spec, y_spec),
            consumer_specs=tuple(
                spec if edge is None else edge.consumer_spec
                for spec, edge in zip((x_spec, y_spec), edges)
            ),
            conversion_edges=edges,
        )

    def prepare(self, x, y):
        """Execute retained x/y conversion edges for one yielded batch.

        Args:
            x: Runtime feature batch matching ``producer_specs[0]``.
            y: Runtime target batch matching ``producer_specs[1]``.

        Returns:
            The prepared ``(x, y)`` pair matching ``consumer_specs``.

        Raises:
            TypeError: If a runtime value violates its retained dense handoff.
            RuntimeError: If a Torch gradient-bearing value would cross a
                framework boundary.

        Side Effects:
            Imports only adapters selected during planning and copies only values
            whose retained edge requires a backend handoff.
        """

        from dryml.methods.conversion import convert

        return tuple(
            value if edge is None else convert(edge, value)
            for value, edge in zip((x, y), self.conversion_edges)
        )


def validate_num_examples(num_examples: int | None) -> None:
    if num_examples is not None and num_examples < 0:
        raise ValueError("num_examples must be non-negative or None.")


def dataset_is_batched(dataset) -> bool:
    try:
        return any(spec.batched for spec in iter_specs(dataset.spec))
    except ValueError:
        return False


def finite_dataset_len(dataset) -> int | None:
    try:
        cardinality = dataset.__len__()
    except Exception:
        return None

    if hasattr(cardinality, "is_finite"):
        if cardinality.is_finite:
            return cardinality.require_finite()
        return None
    return int(cardinality)


def training_cardinality(dataset):
    """Return declared training cardinality without acquiring a cursor.

    Args:
        dataset: Prepared Dataset whose cardinality is queried.

    Returns:
        A finite, unknown, or infinite ``Cardinality`` declaration. Plain integer
        lengths are normalized to finite cardinality.

    Side Effects:
        Calls only ``dataset.__len__``; it does not open, scan, or consume a
        Dataset iterator.
    """

    from dryml.core.cardinality import Cardinality

    try:
        cardinality = dataset.__len__()
    except (NotImplementedError, TypeError):
        return Cardinality.UNKNOWN
    return cardinality if isinstance(cardinality, Cardinality) else Cardinality.finite(int(cardinality))


def require_bounded_safe_points(dataset, callbacks) -> None:
    """Reject streams that cannot expose truthful final update safe points.

    Args:
        dataset: Prepared Dataset used by one trainer invocation.
        callbacks: Prevalidated DRYML safe-point callbacks.

    Raises:
        ValueError: If the Dataset is infinite without a finite bound, or its
            cardinality is unknown while callbacks require final-batch recovery.

    Side Effects:
        Reads only declared cardinality and never opens or consumes the Dataset.
    """

    cardinality = training_cardinality(dataset)
    if cardinality.is_infinite:
        raise ValueError("Training on an infinite dataset requires an explicit finite bound.")
    if callbacks and not cardinality.is_finite:
        raise ValueError(
            "Training callbacks require a finite deterministic batch count for final safe-point recovery."
        )


def prepare_training_data(
    train_data,
    *,
    num_examples: int | None = None,
    shuffle: bool = False,
    shuffle_seed=None,
    shuffle_buffer_size: int | None = None,
):
    if train_data is None:
        raise ValueError("Experiment has no train_data.")
    validate_num_examples(num_examples)

    if dataset_is_batched(train_data):
        train_data = Unbatch(train_data)

    if shuffle:
        buffer_size = shuffle_buffer_size or finite_dataset_len(train_data)
        if buffer_size is None:
            raise ValueError("shuffle_buffer_size is required when train_data length is unknown.")
        train_data = Shuffle(train_data, buffer_size, seed=shuffle_seed)

    if num_examples is not None:
        train_data = Take(train_data, num_examples)

    return train_data


def advance_train_state(exp, *, epochs: int = 0, steps: int = 0, phase: str = TrainState.trained):
    if epochs:
        exp.state.advance_epoch(epochs)
    if steps:
        exp.state.advance_step(steps)
    exp.state.phase = phase


def validate_training_callbacks(callbacks) -> tuple:
    """Freeze and validate every DRYML safe-point callback before training work.

    Args:
        callbacks: Iterable of zero-argument process-local callbacks.

    Returns:
        An immutable callback sequence in caller order.

    Raises:
        TypeError: If the value is not iterable or any member is not callable.

    Side Effects:
        None.  In particular, this check neither opens data nor changes model,
        optimizer, cursor, mode, or retained training state.
    """

    try:
        frozen = tuple(callbacks)
    except TypeError as error:
        raise TypeError("training callbacks must be an iterable of callables.") from error
    if not all(callable(callback) for callback in frozen):
        raise TypeError("training callbacks must be callable.")
    return frozen


def record_train_update(
    exp,
    batch,
    loss: float,
    *,
    batched: bool,
    callbacks=(),
    examples: int | None = None,
    complete_epoch: bool = False,
) -> None:
    """Retain one successful update before invoking process-local safe points.

    Args:
        exp: Experiment whose TrainState receives retained accounting.
        batch: Prepared training value used to determine examples in this update.
        loss: Finite mean loss for the completed update.
        batched: Whether ``batch`` has a leading example dimension.
        callbacks: Zero-argument callbacks run after retained accounting.
        examples: Exact completed-update examples when an owned backend step
            reports them.  When omitted, derive them from ``batch``.
        complete_epoch: Whether this completed update exhausted its logical
            epoch and must normalize state before safe-point callbacks.

    Raises:
        TypeError: If callbacks are not callable.
        ValueError: If a batched update has no examples.

    Side Effects:
        Advances TrainState optimizer/exposure/loss-window state, then invokes
        callbacks. A callback failure does not roll back the completed update.
    """

    derive_examples = examples is None
    if derive_examples:
        examples = 1
    if derive_examples and batched:
        try:
            first = next(iter_leaves(batch))
            examples = len(first)
        except (StopIteration, TypeError):
            raise ValueError("A batched training update must expose a leading example dimension.") from None
    exp.state.record_update(examples=examples, loss=float(loss))
    if complete_epoch:
        exp.state.finish_epoch(postlude_pending=True)
    for callback in callbacks:
        callback()


def signature_discovery(obj: Any, **kwargs):
    try:
        check_context('tf')
        from .tf.utils import tf_signature_discovery
        return tf_signature_discovery(obj, **kwargs)
    except expected_context_errors:
        pass

    raise ValueError("Unable to guess a signature based on the object.")


__all__ = [
    "advance_train_state",
    "dataset_is_batched",
    "finite_dataset_len",
    "require_bounded_safe_points",
    "prepare_training_data",
    "record_train_update",
    "signature_discovery",
    "TrainingPreparation",
    "training_cardinality",
    "validate_num_examples",
    "validate_training_callbacks",
]
