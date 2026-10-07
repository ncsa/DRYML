from __future__ import annotations

from dataclasses import dataclass
from dryml.context import check_context, NoContextError, \
    WrongContextError, ContextIncompatibilityError
from typing import Any

from dryml.core.tensor_spec import iter_specs
from dryml.core.utils.recurse import iter_leaves
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


def finite_dataset_len(dataset) -> int | None:
    """Return a Dataset's finite yield count without opening a cursor."""

    cardinality = dataset.yield_cardinality()
    return cardinality.require_finite() if cardinality.is_finite else None


def training_cardinality(dataset):
    """Return declared training cardinality without acquiring a cursor.

    Args:
        dataset: Canonical Dataset whose yield cardinality is queried.

    Returns:
        A finite, unknown, or infinite ``Cardinality`` declaration.

    Side Effects:
        Calls only ``dataset.yield_cardinality()``; it does not open, scan, or
        consume a Dataset iterator.
    """

    return dataset.yield_cardinality()


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


def require_supervised_dataset(dataset, *, batched: bool) -> None:
    """Validate a canonical Dataset pair for a supplied trainer.

    Args:
        dataset: Dataset that must yield exactly ``(inputs, targets)``. Each
            branch may be a nested TensorSpec tree.
        batched: Whether both branches must carry an explicit leading example
            dimension for a native update loop.

    Raises:
        ValueError: If data is absent, noncanonical, or has incompatible batch
            declarations.
        TypeError: If a Dataset specification has no TensorSpec leaves.

    Side Effects:
        Reads public Dataset specification metadata only. It does not inspect
        Dataset operators, open a cursor, or alter selection/order/batching.
    """

    from dryml.data import Dataset

    if dataset is None:
        raise ValueError("Experiment has no train_data.")
    if not isinstance(dataset, Dataset):
        raise TypeError("Training data must be a Dataset.")
    try:
        inputs, targets = dataset.spec
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "Training data must be a canonical Dataset yielding (inputs, targets); "
            "use data.as_supervised(...)."
        ) from error
    if not isinstance(dataset.spec, tuple) or len(dataset.spec) != 2:
        raise ValueError(
            "Training data must be a canonical Dataset yielding (inputs, targets); "
            "use data.as_supervised(...)."
        )
    branch_batches = []
    for branch in (inputs, targets):
        specs = tuple(iter_specs(branch))
        if not specs:
            raise TypeError("Training data branches require TensorSpec leaves.")
        flags = {spec.batched for spec in specs}
        if len(flags) != 1:
            raise ValueError("Training data branches must be uniformly batched or unbatched.")
        branch_batches.append(flags.pop())
    if branch_batches[0] != branch_batches[1]:
        raise ValueError("Training inputs and targets must have matching batching declarations.")
    if batched and not branch_batches[0]:
        raise ValueError(
            "Native training requires explicitly batched (inputs, targets) data; "
            "author Batch(dataset, 1) or another Dataset batch before training."
        )


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
    epoch_metrics: dict[str, float] | None = None,
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
        epoch_metrics: Finite completed-epoch metrics to retain before callbacks
            when ``complete_epoch`` is true.

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
        exp.state.finish_epoch(postlude_pending=True, metrics=epoch_metrics)
    elif epoch_metrics is not None:
        raise ValueError("epoch_metrics require a completed epoch update.")
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
    "finite_dataset_len",
    "record_train_update",
    "require_bounded_safe_points",
    "require_supervised_dataset",
    "signature_discovery",
    "TrainingPreparation",
    "training_cardinality",
    "validate_training_callbacks",
]
